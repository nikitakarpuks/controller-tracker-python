"""accel_tools.py -- windowed gravity-consistency accelerometer-bias measurement (telescoping at the VELOCITY level).

Model (absolute inertial/mocap-room frame, additive-gravity convention of src/imu_data.py):
    f_meas(t) = R(t)^T (a_w(t) - g) + lever(t) + b_a + noise            (f: specific force, body frame, factory-corrected)
    =>  a_w = R (f_meas - lever) + g - R b_a
    =>  Q(window) := int (R f_c + g) dt - (v(t_e) - v(t_s))  =  ( int R dt ) b_a  + noise,        f_c = f_meas - lever
so   b_a = (int R dt)^-1 Q.  Every quantity is a plain sum over the window (cumulative sums make any window O(1)); the
window noise is dominated by the two end velocities (~0.05-0.1 m/s), NOT by per-sample noise -- exactly the telescoping
structure that makes long windows informative while a per-frame position residual is not.
Requirements: absolute orientation R(t) (vision-lifted to the inertial frame, or mocap) and positions for the end velocities.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from common import *
from src.imu_data import _lever_arm_correction, MOCAP_ROOM_G_WORLD

G_ABS = MOCAP_ROOM_G_WORLD


class AbsTrack:
    """Absolute (inertial-frame) orientation + LED-origin position track at frame times, with SLERP/lerp helpers."""

    def __init__(self, ts, R, P, max_gap_s=0.10):
        self.ts, self.R, self.P = np.asarray(ts, dtype=np.int64), np.asarray(R), np.asarray(P)
        self.slerp = Slerp(self.ts.astype(float), Rotation.from_matrix(self.R))
        self.max_gap = max_gap_s * 1e9

    def valid(self, t):
        t = np.asarray(t)
        k = np.clip(np.searchsorted(self.ts, t), 1, len(self.ts) - 1)
        inside = (t >= self.ts[0]) & (t <= self.ts[-1])
        return inside & ((self.ts[k] - self.ts[k - 1]) <= self.max_gap)

    def R_at(self, t):
        t = np.clip(np.asarray(t, dtype=float), self.ts[0], self.ts[-1])
        return self.slerp(t).as_matrix()

    def v_at(self, t, half_s=0.06, min_frames=3):
        """local linear-fit velocity around t using frames within +-half_s (None if too few frames)."""
        lo, hi = np.searchsorted(self.ts, int(t - half_s * 1e9)), np.searchsorted(self.ts, int(t + half_s * 1e9))
        if hi - lo < min_frames:
            return None
        tt = (self.ts[lo:hi] - t) / 1e9
        A = np.vstack([np.ones_like(tt), tt]).T
        coef, *_ = np.linalg.lstsq(A, self.P[lo:hi], rcond=None)
        return coef[1]


def track_vision(run, ctrl, prep):
    """Vision-derived absolute track (strong frames): R = prep['R_wc'], P = R_wh p_hc + p_wh (mocap headset ego -- dev only)."""
    ts, ok = prep["ts"], prep["ok"]
    idx = np.flatnonzero(ok)
    P, keep = [], []
    for i in idx:
        Th = run.T_wh(ts[i])
        keep.append(Th is not None)
        P.append(np.zeros(3) if Th is None else Th.R @ prep["p_hc"][i] + Th.t)
    idx = idx[np.array(keep)]
    P = np.array(P)[np.array(keep)]
    return AbsTrack(ts[idx], prep["R_wc"][idx], P)


def track_mocap(run, ctrl):
    d = run.ctrl[ctrl]; ts = d["vision"].ts
    R, P, tt = [], [], []
    for t in ts:
        T = world_pose(d["mocap"], int(t))
        if T is None: continue
        Tl = T.compose(d["bridge"].inverse())
        R.append(Tl.R); P.append(Tl.t); tt.append(t)
    return AbsTrack(np.array(tt), np.array(R), np.array(P))


class AccelWindows:
    """Precomputed per-IMU-sample cumulative sums for O(1) window measurements of b_a."""

    def __init__(self, run, ctrl, track, gyro=None, accel=None):
        d = run.ctrl[ctrl]
        t = d["t"]; gy = d["gyro"] if gyro is None else gyro; ac = d["accel"] if accel is None else accel
        lev = _lever_arm_correction(t, t, gy, d["lever"])
        fc = ac - lev
        m = track.valid(t)
        Rk = np.zeros((len(t), 3, 3)); Rk[m] = track.R_at(t[m])
        dt = np.zeros(len(t)); dt[1:] = np.diff(t) / 1e9
        dt = np.where(m & np.roll(m, 1), np.minimum(dt, 0.02), 0.0)   # ignore samples in coverage holes / huge IMU gaps
        q = np.einsum("nij,nj->ni", Rk, fc) + G_ABS
        self.t = t
        self.cQ = np.vstack([np.zeros(3), np.cumsum(q * dt[:, None], axis=0)])
        self.cM = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(Rk * dt[:, None, None], axis=0)])
        self.cT = np.concatenate([[0.0], np.cumsum(dt)])
        # scale/misalignment regressor: F[i, 3a+b] = sum R[i,a] f_c[b] dt
        Fk = np.einsum("nia,nb->niab", Rk, fc).reshape(len(t), 3, 9) * dt[:, None, None]
        self.cF = np.concatenate([np.zeros((1, 3, 9)), np.cumsum(Fk, axis=0)])
        # lever-arm regressor: columns of Omega_n (lever correction is LINEAR in r): O[:, :, k] = R_n * lever(t_n; r=e_k) * dt
        Ok = np.stack([np.einsum("nij,nj->ni", Rk, _lever_arm_correction(t, t, gy, np.eye(3)[k])) * dt[:, None] for k in range(3)], axis=2)
        self.cO = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(Ok, axis=0)])
        self.track = track

    def _fit_v(self, lo, hi):
        """LS-line velocity over frames lo..hi-1; returns (t_center_ns, v) with t_center = mean fit time."""
        tt = self.track.ts[lo:hi].astype(float)
        tc = tt.mean()
        A = np.vstack([np.ones(hi - lo), (tt - tc) / 1e9]).T
        coef, *_ = np.linalg.lstsq(A, self.track.P[lo:hi], rcond=None)
        return int(tc), coef[1]

    def components(self, t_s, t_e, fit_span_s=0.06, min_seg_s=0.5):
        """(Q (3,), M (3,3), F (3,9), covered_fraction, n_seg) summed over hole-free segments, or None. See measure()."""
        ts = self.track.ts
        lo_w, hi_w = np.searchsorted(ts, t_s), np.searchsorted(ts, t_e)
        if hi_w - lo_w < 8:
            return None
        gaps = np.diff(ts[lo_w:hi_w]) > self.track.max_gap
        cuts = np.flatnonzero(gaps) + 1
        bounds = np.concatenate([[0], cuts, [hi_w - lo_w]]) + lo_w
        M_tot = np.zeros((3, 3)); Q_tot = np.zeros(3); F_tot = np.zeros((3, 9)); O_tot = np.zeros((3, 3)); covered = 0.0; nseg = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 8 or (ts[b - 1] - ts[a]) / 1e9 < min_seg_s:
                continue
            n0 = 3
            while a + n0 < b and (ts[a + n0 - 1] - ts[a]) / 1e9 < fit_span_s: n0 += 1
            n1 = 3
            while b - n1 > a and (ts[b - 1] - ts[b - n1]) / 1e9 < fit_span_s: n1 += 1
            tc0, v0 = self._fit_v(a, a + n0); tc1, v1 = self._fit_v(b - n1, b)
            if tc1 - tc0 < min_seg_s * 1e9:
                continue
            i0, i1 = np.searchsorted(self.t, tc0), np.searchsorted(self.t, tc1)
            Q_tot += (self.cQ[i1] - self.cQ[i0]) - (v1 - v0)
            M_tot += self.cM[i1] - self.cM[i0]
            F_tot += self.cF[i1] - self.cF[i0]
            O_tot += self.cO[i1] - self.cO[i0]
            covered += (tc1 - tc0) / 1e9; nseg += 1
        if nseg == 0:
            return None
        self.last_O = O_tot
        return Q_tot, M_tot, F_tot, covered / ((t_e - t_s) / 1e9), nseg

    def measure(self, t_s, t_e, fit_span_s=0.06, ridge=1e-3, min_seg_s=0.5):
        """Segment-wise window measurement. Vision-coverage holes (> max_gap) break the telescoping of Q - dv, so the window is
        split into hole-free segments, each with its OWN end velocities (one-sided LS fits at the segment ends, integrated
        between the fit centres), and the normal equations are summed:  b = (sum M_seg)^-1 sum (Q_seg - dv_seg).
        Returns (b (3,), covered_fraction, n_segments) or None."""
        ts = self.track.ts
        lo_w, hi_w = np.searchsorted(ts, t_s), np.searchsorted(ts, t_e)
        if hi_w - lo_w < 8:
            return None
        gaps = np.diff(ts[lo_w:hi_w]) > self.track.max_gap
        cuts = np.flatnonzero(gaps) + 1
        bounds = np.concatenate([[0], cuts, [hi_w - lo_w]]) + lo_w
        M_tot = np.zeros((3, 3)); Q_tot = np.zeros(3); covered = 0.0; nseg = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 8 or (ts[b - 1] - ts[a]) / 1e9 < min_seg_s:
                continue
            # start / end fit windows (>=3 frames spanning >= fit_span_s)
            n0 = 3
            while a + n0 < b and (ts[a + n0 - 1] - ts[a]) / 1e9 < fit_span_s: n0 += 1
            n1 = 3
            while b - n1 > a and (ts[b - 1] - ts[b - n1]) / 1e9 < fit_span_s: n1 += 1
            tc0, v0 = self._fit_v(a, a + n0); tc1, v1 = self._fit_v(b - n1, b)
            if tc1 - tc0 < min_seg_s * 1e9:
                continue
            i0, i1 = np.searchsorted(self.t, tc0), np.searchsorted(self.t, tc1)
            Q_tot += (self.cQ[i1] - self.cQ[i0]) - (v1 - v0)
            M_tot += self.cM[i1] - self.cM[i0]
            covered += (tc1 - tc0) / 1e9; nseg += 1
        if nseg == 0:
            return None
        return np.linalg.solve(M_tot + ridge * np.eye(3), Q_tot), covered / ((t_e - t_s) / 1e9), nseg
