"""simlib.py -- synthetic IMU + vision simulator built from REAL mocap trajectories.

Truth is defined by smoothing real mocap (so IMU can be generated analytically and exactly
self-consistently), then IMU/vision streams are synthesized at the REAL sample timestamps.

Conventions (identical to src/imu_data.py, verified by tests in run_tests.py):
  * R = R_world_body; dR/dt = R [omega_body]x
  * accel_body = R^T (a_world - g_world) + alpha x r + omega x (omega x r) + b_a + noise
    (project: a_world = R (f - lever) + g_world, g_world = gravity vector, e.g. [0,-9.81,0])
  * gyro_body  = omega_body + b_g + noise           (bias ADDED to truth; estimator subtracts)
  * the vision world frame is the HEADSET rig frame: T_hc = T_wh^-1 T_wc.
"""
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as Rot, Slerp

REPO = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
sys.path.insert(0, str(REPO))
from src.imu_data import load_imu_csv, MOCAP_ROOM_G_WORLD  # noqa: E402
from src.mocap_data import load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker  # noqa: E402

REC_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
RECORDINGS = {
    "static_dark": "euroc_recording_20260826173103_static_dark",
    "walk_medium": "euroc_recording_20260826175510_walk_medium",
}
EVAL_ROOT = REPO / "visualization" / "evaluate_2026-09-22"
CALIB_DIR = REC_ROOT / "mocap_calibrations_for_each_device"
_DISK = {"headset": "headset", "left": "ctrlleft", "right": "ctrlright"}
_CALIB = {"headset": "mocap_calibration_headset.json", "left": "controller_left_calib.json",
          "right": "controller_right_calib.json"}
_IMU = {"left": "imu1", "right": "imu2"}
LAG_NS = {"left": -7_650_000, "right": -7_600_000}       # config: controller IMU stamp -> camera clock
LEVER_ARM = {"left": np.array([0.034635, -0.077529, 0.002809]),   # accel - gyro (body), README finding 5
             "right": np.array([-0.033227, -0.080239, 0.002945])}
G_WORLD = MOCAP_ROOM_G_WORLD.copy()                        # [0,-9.81,0]


# --------------------------------------------------------------------------- SO(3) helpers
def Exp(v):
    return Rot.from_rotvec(v).as_matrix()


def Log(R):
    return Rot.from_matrix(R).as_rotvec()


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


# --------------------------------------------------------------------------- truth
class Truth:
    """Smooth ground-truth trajectory of one rigid body (body = IMU frame). Times are seconds
    since T0_ns (camera clock)."""

    def __init__(self, t_s, p_grid, w_mid, R0, fs=120.0, fc=8.0, dense_dt=1e-3):
        # position: low-pass + cubic spline (analytic v, a)
        sos = butter(4, fc / (fs / 2), output="sos")
        p_f = sosfiltfilt(sos, p_grid, axis=0, padtype="odd")
        self.t0, self.t1 = t_s[0], t_s[-1]
        self.p_spl = CubicSpline(t_s, p_f)
        self.v_spl = self.p_spl.derivative(1)
        self.a_spl = self.p_spl.derivative(2)
        # angular velocity (body): low-pass + cubic spline
        t_mid = 0.5 * (t_s[:-1] + t_s[1:])
        w_f = sosfiltfilt(sos, w_mid, axis=0, padtype="odd")
        self.w_spl = CubicSpline(t_mid, w_f)
        self.al_spl = self.w_spl.derivative(1)
        # rotation: integrate the smooth omega exactly on a dense grid, slerp in between
        n = int(np.floor((self.t1 - self.t0) / dense_dt))
        self._td = self.t0 + dense_dt * np.arange(n + 1)
        wm = self.w_spl(self._td[:-1] + 0.5 * dense_dt)
        dRs = Rot.from_rotvec(wm * dense_dt).as_matrix()
        Rs = np.empty((n + 1, 3, 3))
        Rs[0] = R0
        for j in range(n):
            Rs[j + 1] = Rs[j] @ dRs[j]
        self._Rs = Rs
        self._slerp = Slerp(self._td, Rot.from_matrix(Rs))

    def R(self, t):
        t = np.clip(np.atleast_1d(t), self._td[0], self._td[-1])
        return self._slerp(t).as_matrix()

    def p(self, t): return self.p_spl(np.atleast_1d(t))
    def v(self, t): return self.v_spl(np.atleast_1d(t))
    def a(self, t): return self.a_spl(np.atleast_1d(t))
    def w(self, t): return self.w_spl(np.atleast_1d(t))
    def alpha(self, t): return self.al_spl(np.atleast_1d(t))


def load_truth(rec_name, dev, T0_ns, fs=120.0, fc=8.0):
    """Build a Truth from real mocap (marker pose -> IMU pose via T_imu_marker, camera-clock =
    mocap - fine_offset). Pose chain identical to src.mocap_data.world_pose."""
    root = REC_ROOT / RECORDINGS[rec_name]
    d = root / "mocap_filtered" / _DISK[dev]
    t, pos, quat = load_mocap_csv(d / "data.csv")
    fine = load_mocap_fine_offset_ns(d / "drift_check" / "1chunk" / "drift_check.json")
    Tim = load_T_imu_marker(CALIB_DIR / _CALIB[dev])
    R_wm = Rot.from_quat(quat.astype(np.float64)).as_matrix()
    R_wi = R_wm @ Tim.R.T
    p_wi = pos.astype(np.float64) - np.einsum("nij,j->ni", R_wi, Tim.t)
    t_s = (t.astype(np.float64) - fine - T0_ns) / 1e9
    tg = np.arange(np.ceil(t_s[0] * fs) / fs, t_s[-1], 1.0 / fs)
    pg = np.stack([np.interp(tg, t_s, p_wi[:, i]) for i in range(3)], axis=1)
    Rg = Slerp(t_s, Rot.from_matrix(R_wi))(tg).as_matrix()
    dR = np.einsum("nji,njk->nik", Rg[:-1], Rg[1:])          # R_k^T R_{k+1}
    w_mid = Rot.from_matrix(dR).as_rotvec() * fs
    return Truth(tg, pg, w_mid, Rg[0], fs=fs, fc=fc)


# --------------------------------------------------------------------------- scenario
@dataclass
class Scenario:
    name: str = "base"
    seed: int = 0
    # sensor noise (per-sample white); factory Noise + real quiet-segment measurement
    noise_g: float = 7e-4
    noise_a: float = 6.5e-3
    # bias truth
    b_g0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_a0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rw_g: float = 0.0                # rad/s per sqrt(s)
    rw_a: float = 0.0                # m/s^2 per sqrt(s)
    warm_g: np.ndarray = field(default_factory=lambda: np.zeros(3))   # exponential warm-up amplitude
    warm_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    warm_tau: float = 40.0
    # vision
    sig_rot_deg: float = 0.4
    sig_pos_mm: float = 4.0
    outlier_frac: float = 0.01
    corr_rot_deg: float = 0.0        # AR(1) correlated vision error
    corr_pos_mm: float = 0.0
    corr_tau: float = 0.5
    strong_only: bool = True
    swap_window: tuple = None        # (t_a, t_b) seconds: vision reports OTHER controller's pose
    # model errors (truth-side; estimator assumes ideal)
    timing_ms: float = 0.0           # IMU sampled at t_rec - timing (estimator believes stamp)
    jitter_ms: float = 0.0
    axis_mis_deg: float = 0.0        # sensor frame rotated vs assumed (random axis)
    scale_g: float = 0.0
    scale_a: float = 0.0
    headset_mode: str = "mocap"      # mocap | ignored | vio_drift
    headset_drift: float = 1e-3      # rad/s slope for vio_drift (sawtooth, reset every 10 s)
    # estimator-side model errors
    lever_err: float = 0.0           # estimator lever arm scaled (1+lever_err)
    grav_tilt_deg: float = 0.0
    grav_mag_err: float = 0.0        # fraction


@dataclass
class SimData:
    dev: str
    scen: Scenario
    T0_ns: int
    # IMU (recorded stamps, camera clock)
    t_imu_ns: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    b_g_true: np.ndarray             # (N,3) at sample times
    b_a_true: np.ndarray
    # vision nodes (measurements as the estimator sees them, world-lifted)
    t_v_ns: np.ndarray
    strong: np.ndarray
    R_wc_meas: np.ndarray
    p_wc_meas: np.ndarray
    R_hc_meas: np.ndarray
    p_hc_meas: np.ndarray
    R_wh_est: np.ndarray
    p_wh_est: np.ndarray
    # truth at vision nodes
    R_wc_true: np.ndarray
    p_wc_true: np.ndarray
    g_est: np.ndarray                # gravity vector the estimator believes
    lever_est: np.ndarray
    truth: Truth = None
    truth_h: Truth = None
    sig_rot: float = 0.0
    sig_pos: float = 0.0

    @property
    def t_imu_s(self): return (self.t_imu_ns - self.T0_ns) / 1e9
    @property
    def t_v_s(self): return (self.t_v_ns - self.T0_ns) / 1e9


def load_vision_rows(rec_name, dev, strong_thr=None):
    f = EVAL_ROOT / rec_name / "vision_pose.csv"
    name = f"{dev}_controller"
    rows = [r for r in csv.reader(open(f)) if r[1] == name][1:] if False else None
    out = []
    with open(f) as fh:
        rd = csv.reader(fh); next(rd)
        for r in rd:
            if r[1] == name:
                out.append((int(r[0]), float(r[10]), float(r[11])))
    a = np.array(out)
    ts, err, ninl = a[:, 0].astype(np.int64), a[:, 1], a[:, 2]
    thr = strong_thr if strong_thr is not None else (0.15 if dev == "left" else 0.2)
    return ts, (ninl >= 8) & (err <= thr)


_TRUTH_CACHE = {}
def get_truth(rec_name, dev, T0_ns):
    k = (rec_name, dev, T0_ns)
    if k not in _TRUTH_CACHE:
        _TRUTH_CACHE[k] = load_truth(rec_name, dev, T0_ns)
    return _TRUTH_CACHE[k]


def simulate(rec_name, dev, scen: Scenario, t_max_s=None) -> SimData:
    rng = np.random.default_rng(scen.seed)
    root = REC_ROOT / RECORDINGS[rec_name] / "mav0"
    t_raw, _, _ = load_imu_csv(root / _IMU[dev] / "data.csv")
    t_rec = t_raw + LAG_NS[dev]                                  # recorded stamps, camera clock
    T0 = int(t_rec[0])
    tr = get_truth(rec_name, dev, T0)
    th = get_truth(rec_name, "headset", T0)
    lo = max(tr.t0, th.t0) + 0.5
    hi = min(tr.t1, th.t1) - 0.5
    if t_max_s is not None:
        hi = min(hi, lo + t_max_s)
    ts_all = (t_rec - T0) / 1e9
    m = (ts_all >= lo) & (ts_all <= hi)
    t_imu_ns = t_rec[m]
    ts = ts_all[m]
    N = len(ts)

    # ---- bias truth (sampled at true sensor times)
    t_samp = ts - scen.timing_ms * 1e-3 + (rng.normal(0, scen.jitter_ms * 1e-3, N) if scen.jitter_ms > 0 else 0.0)
    dts = np.diff(np.concatenate(([ts[0] - 0.005], ts)))
    def rw(sig):
        return np.cumsum(rng.normal(0, 1.0, (N, 3)) * np.sqrt(dts)[:, None] * sig, axis=0)
    tt = (ts - ts[0])[:, None]
    warm = 1.0 - np.exp(-tt / scen.warm_tau)
    b_g = scen.b_g0 + rw(scen.rw_g) + scen.warm_g * warm
    b_a = scen.b_a0 + rw(scen.rw_a) + scen.warm_a * warm

    # ---- IMU synthesis
    R = tr.R(t_samp); w = tr.w(t_samp); al = tr.alpha(t_samp); a_w = tr.a(t_samp)
    r = LEVER_ARM[dev]
    lever = np.cross(al, r) + np.cross(w, np.cross(w, r))
    f_body = np.einsum("nji,nj->ni", R, a_w - G_WORLD) + lever
    R_mis = np.eye(3)
    if scen.axis_mis_deg > 0:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        R_mis = Exp(ax * np.radians(scen.axis_mis_deg))
    gyro = (1 + scen.scale_g) * (w @ R_mis.T) + b_g + rng.normal(0, scen.noise_g, (N, 3))
    accel = (1 + scen.scale_a) * (f_body @ R_mis.T) + b_a + rng.normal(0, scen.noise_a, (N, 3))

    # ---- vision nodes
    tv_all, strong_all = load_vision_rows(rec_name, dev)
    tvs = (tv_all - T0) / 1e9
    mv = (tvs >= lo + 0.1) & (tvs <= hi - 0.1)
    tv_ns, strong = tv_all[mv], strong_all[mv]
    if scen.strong_only:
        tv_ns, strong = tv_ns[strong], strong[strong]
    tvs = (tv_ns - T0) / 1e9
    K = len(tvs)
    Rc, pc = tr.R(tvs), tr.p(tvs)
    Rh, ph = th.R(tvs), th.p(tvs)
    R_hc = np.einsum("nji,njk->nik", Rh, Rc)
    p_hc = np.einsum("nji,nj->ni", Rh, pc - ph)
    sr, sp = np.radians(scen.sig_rot_deg), scen.sig_pos_mm * 1e-3
    nr = rng.normal(0, sr, (K, 3)); npos = rng.normal(0, sp, (K, 3))
    if scen.corr_rot_deg > 0 or scen.corr_pos_mm > 0:
        er = np.zeros((K, 3)); ep = np.zeros((K, 3))
        for k in range(1, K):
            phi = np.exp(-(tvs[k] - tvs[k - 1]) / scen.corr_tau)
            s = np.sqrt(1 - phi ** 2)
            er[k] = phi * er[k - 1] + s * rng.normal(0, np.radians(scen.corr_rot_deg), 3)
            ep[k] = phi * ep[k - 1] + s * rng.normal(0, scen.corr_pos_mm * 1e-3, 3)
        nr += er; npos += ep
    R_hc_m = np.einsum("nij,njk->nik", R_hc, Rot.from_rotvec(nr).as_matrix())
    p_hc_m = p_hc + npos
    if scen.outlier_frac > 0:
        idx = rng.random(K) < scen.outlier_frac
        for k in np.flatnonzero(idx):
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            R_hc_m[k] = R_hc[k] @ Exp(ax * np.radians(rng.uniform(20, 90)))
            p_hc_m[k] = p_hc[k] + rng.normal(size=3) * 0.2
    if scen.swap_window is not None:
        other = "left" if dev == "right" else "right"
        to = get_truth(rec_name, other, T0)
        a, b = scen.swap_window
        for k in np.flatnonzero((tvs >= a) & (tvs <= b)):
            Ro, po = to.R(tvs[k])[0], to.p(tvs[k])[0]
            R_hc_m[k] = Rh[k].T @ Ro
            p_hc_m[k] = Rh[k].T @ (po - ph[k])

    # ---- headset estimate seen by the estimator + gravity/lever the estimator believes
    g_est = G_WORLD.copy()
    if scen.headset_mode == "mocap":
        Rh_e, ph_e = Rh, ph
    elif scen.headset_mode == "vio_drift":
        e = np.zeros((K, 3))
        # sawtooth rotation error growing at slope headset_drift, reset every 10 s, random axis per segment
        seg = np.floor(tvs / 10.0).astype(int)
        for s_ in np.unique(seg):
            ax = np.random.default_rng(1000 + s_).normal(size=3); ax /= np.linalg.norm(ax)
            mk = seg == s_
            e[mk] = ax * scen.headset_drift * (tvs[mk] - s_ * 10.0)[:, None]
        Rh_e = np.einsum("nij,njk->nik", Rh, Rot.from_rotvec(e).as_matrix())
        ph_e = ph
    elif scen.headset_mode == "ignored":
        # estimator's "world" = headset rig frame at the recording start; gravity expressed there
        R00 = th.R(np.array([lo]))[0]
        Rh_e = np.broadcast_to(np.eye(3), (K, 3, 3)).copy(); ph_e = np.zeros((K, 3))
        g_est = R00.T @ G_WORLD
    else:
        raise ValueError(scen.headset_mode)
    if scen.grav_tilt_deg > 0:
        ax = np.cross(g_est, [1.0, 0.3, 0.2]); ax /= np.linalg.norm(ax)
        g_est = Exp(ax * np.radians(scen.grav_tilt_deg)) @ g_est
    g_est = g_est * (1 + scen.grav_mag_err)
    R_wc_m = np.einsum("nij,njk->nik", Rh_e, R_hc_m)
    p_wc_m = ph_e + np.einsum("nij,nj->ni", Rh_e, p_hc_m)
    if scen.headset_mode == "ignored":
        # the "true" quantities in the estimator's (rig) frame
        Rc_t = np.einsum("nji,njk->nik", th.R(tvs), Rc) if False else R_hc
        R_wc_true, p_wc_true = R_hc, p_hc
    else:
        R_wc_true, p_wc_true = Rc, pc
    return SimData(dev, scen, T0, t_imu_ns, gyro, accel, b_g, b_a, tv_ns, strong[:K] if not scen.strong_only else np.ones(K, bool),
                   R_wc_m, p_wc_m, R_hc_m, p_hc_m, Rh_e, ph_e, R_wc_true, p_wc_true, g_est,
                   LEVER_ARM[dev] * (1 + scen.lever_err), tr, th, sr, sp)
