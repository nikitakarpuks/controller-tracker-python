"""fv_common.py -- shared, SELF-CONTAINED helpers for the loader + lever-arm fix validation (investigator V).

Independence: uses ONLY committed snapshots of the project math (imu_data_head.py = git HEAD:src/imu_data.py,
mocap_data_head.py = git HEAD:src/mocap_data.py, both stored next to this file) so the results cannot be
influenced by the loader/lever edits being made concurrently in the working tree. Stream variants are built
here directly from the raw CSV.

Conventions (all verified in the design review):
  * recorded CSV = driver output AFTER mix+bias+P_oxr (monado-main wmr_controller_hp.c / wmr_controller_base.c),
    body = D @ sensor with D = diag(1,-1,-1)
  * bridge maps IMU-frame points to LED-frame points (T_vision.compose(bridge) ~= relative_pose), so the
    accelerometer origin in the LED frame is bridge.t
  * lever arm r = accelerometer position in the body/LED frame; consumed by imu_data._lever_arm_correction
    (subtracts alpha x r + omega x (omega x r) from the accel reading).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
REPO = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import imu_data_head as IH          # noqa: E402  (snapshot of HEAD src/imu_data.py)
import mocap_data_head as MH        # noqa: E402  (snapshot of HEAD src/mocap_data.py)
from src.load_config import load_yaml_config, load_json_config   # noqa: E402

REC_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
EVAL = REPO / "visualization" / "evaluate_2026-09-22"
CONFIG = load_yaml_config(str(REPO / "config" / "config.yml"))
CTRLS = ("left_controller", "right_controller")
MODERATE = ["static_dark", "walk_dark", "static_easy", "static_medium", "walk_easy", "walk_medium"]
ALL8 = ["static_dark", "walk_dark", "static_easy", "static_medium", "static_hard", "walk_easy", "walk_medium", "walk_hard"]
_DISK = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_IMU = {"left_controller": "imu1", "right_controller": "imu2"}
D = np.diag([1.0, -1.0, -1.0])
SCALE = 9.80665 / 10.0
G_ABS = np.array([0.0, -9.81, 0.0])             # = imu_data.MOCAP_ROOM_G_WORLD (additive convention a_w = R f + g)


def rec_dir(name):
    return next(p for p in sorted(REC_ROOT.iterdir()) if p.is_dir() and p.name.endswith("_" + name))


def vision_csv(name):
    if name in ("static_hard", "walk_hard"):
        return EVAL / f"{name}_vision_pose.csv"
    return EVAL / name / "vision_pose.csv"


def load_mocap_device(rdir, key):
    d = rdir / "mocap_filtered" / _DISK[key]
    t, pos, quat = MH.load_mocap_csv(d / "data.csv")
    fine = MH.load_mocap_fine_offset_ns(d / "drift_check" / MH.DRIFT_CHECK_VARIANT / "drift_check.json")
    if key == "headset":
        calib, cfg = CONFIG["cameras"]["mocap_calib_path"], None
    else:
        calib, cfg = CONFIG["controllers"][key]["mocap_calib_path"], CONFIG["controllers"][key]
    voff = MH.load_vision_offset_ns(cfg)
    doff, drate = MH.load_vision_drift_params(cfg)
    return MH.DeviceMocap(t, pos, quat, fine, MH.load_T_imu_marker(calib),
                          max_interp_gap_ns=float(CONFIG["mocap"].get("max_interp_gap_ms", 30.0)) * 1e6,
                          vision_offset_ns=voff, drift_offset_ns=doff, drift_rate_ns_per_ns=drate)


def lag_ns(ctrl):
    """lag_ns = -mocap_vision_offset_ns (the shipped shared constant), as main.py uses."""
    return -int(round(MH.load_vision_offset_ns(CONFIG["controllers"][ctrl])))


def factory_calib(ctrl, entry=1):
    cfg = load_json_config(str(REPO / CONFIG["controllers"][ctrl]["config_path"]))
    return IH.create_imu_calib_from_config(cfg, entry_index=entry)


def raw_csv(rdir, ctrl):
    """(t_ns lagged onto the camera clock, gyro_csv (N,3), accel_csv (N,3)); NO factory correction, NO axis flip."""
    t, g, a = IH.load_imu_csv(rdir / "mav0" / _IMU[ctrl] / "data.csv")
    return t + lag_ns(ctrl), g.astype(np.float64), a.astype(np.float64)


def build_streams(ctrl, t_g_csv, a_csv):
    """The four loader variants (body frame = D @ ...):
       L0 OLD        : factory mix+bias applied a second time, accel scale 1
       L1 drop2nd    : NO second correction, accel scale 1
       L2 scale-only : second correction kept, accel x SCALE
       L3 NEW        : NO second correction, accel x SCALE
    Returns {name: (gyro_body, accel_body)}."""
    c = factory_calib(ctrl, 1)
    g_old = (D @ (c.gyro.mix0 @ t_g_csv.T + c.gyro.bias0[:, None])).T
    a_old = (D @ (c.accel.mix0 @ a_csv.T + c.accel.bias0[:, None])).T
    g_new = (D @ t_g_csv.T).T
    a_new_noscale = (D @ a_csv.T).T
    return {"L0_OLD": (g_old, a_old),
            "L1_drop2nd": (g_new, a_new_noscale),
            "L2_scale_only": (g_old, SCALE * a_old),
            "L3_NEW": (g_new, SCALE * a_new_noscale)}


def lever_candidates(ctrl):
    """Lever-arm candidates (metres, body/LED frame)."""
    c = factory_calib(ctrl, 1)
    side = "left" if ctrl.startswith("left") else "right"
    bridge = MH.load_mocap_bridge(str(REPO / f"data/mocap_calib/controller_{side}_mocap_bridge_basalt01.json"))
    a = c.accel.T_rt
    fac_now = c.accel.T_rt.compose(c.gyro.T_rt.inverse()).t                     # what main.py builds today
    cand_c = bridge.t.copy()                                                      # accelerometer origin in the LED frame
    cand_d = -a.R.T @ a.t                                                         # factory Rt read in the correct frame
    cand_e = -bridge.R.T @ bridge.t                                               # my earlier (wrong-convention) design vector
    return {"a_zero": np.zeros(3), "b_factory_t (main.py today)": fac_now, "c_bridge.t": cand_c,
            "d_factory -R^T t": cand_d, "e_old_design -Rb^T tb": cand_e, "f_mean(c,d)": 0.5 * (cand_c + cand_d)}


# ----------------------------------------------------------------------------------------------- vision
class Vision:
    def __init__(self, csv_path, ctrl):
        ts, R, p, err, nin = [], [], [], [], []
        with open(csv_path, newline="") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if row[1] != ctrl:
                    continue
                ts.append(int(row[0])); R.append(Rotation.from_quat([float(x) for x in row[2:6]]).as_matrix())
                p.append([float(x) for x in row[6:9]]); err.append(float(row[10])); nin.append(float(row[11]))
        o = np.argsort(ts)
        self.ts = np.array(ts, dtype=np.int64)[o]; self.R = np.array(R)[o]; self.p = np.array(p)[o]
        self.err = np.array(err)[o]; self.nin = np.array(nin)[o]

    def strong(self, min_inl=8, max_err=0.5):
        return (self.nin >= min_inl) & (self.err <= max_err) & np.isfinite(self.err)


class AbsTrack:
    """Absolute (inertial/mocap-room frame) orientation + LED-origin position track at strong-frame times."""

    def __init__(self, ts, R, P, max_gap_s=0.10):
        self.ts, self.R, self.P = np.asarray(ts, dtype=np.int64), np.asarray(R), np.asarray(P)
        self.max_gap = max_gap_s * 1e9

    def v_at(self, t, half_s=0.06, min_frames=3):
        lo, hi = np.searchsorted(self.ts, int(t - half_s * 1e9)), np.searchsorted(self.ts, int(t + half_s * 1e9))
        if hi - lo < min_frames:
            return None
        tt = (self.ts[lo:hi] - t) / 1e9
        A = np.vstack([np.ones_like(tt), tt]).T
        coef, *_ = np.linalg.lstsq(A, self.P[lo:hi], rcond=None)
        return coef[1]


def vision_track(name, ctrl, headset_mocap):
    """Strong vision frames lifted to the absolute frame with the MOCAP headset pose (dev-only ego; independent of
    the controller mocap, the bridge and every lever candidate)."""
    v = Vision(vision_csv(name), ctrl)
    ok = v.strong()
    ts, R, P = [], [], []
    for i in np.flatnonzero(ok):
        Th = MH.world_pose(headset_mocap, int(v.ts[i]))
        if Th is None:
            continue
        ts.append(v.ts[i]); R.append(Th.R @ v.R[i]); P.append(Th.R @ v.p[i] + Th.t)
    return AbsTrack(ts, R, P)


# ----------------------------------------------------------------------------------------------- stats
def block_bootstrap_ci(diff, times_ns, block_s=5.0, n=2000, seed=0, stat=np.mean):
    """95 % CI of stat(diff) resampling 5-s blocks (anchors are strongly autocorrelated)."""
    m = np.isfinite(diff); diff, t = np.asarray(diff)[m], np.asarray(times_ns)[m]
    blk = ((t - t.min()) / 1e9 // block_s).astype(int)
    ub = np.unique(blk); groups = [diff[blk == u] for u in ub]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[i] for i in pick])))
    return np.percentile(vals, [2.5, 97.5])


# ----------------------------------------------------------------------------------------------- mocap orientation / IMU-point pose
class MocapPose:
    """R_world_imu(t), IMU-point position(t) for one device from its mocap track, evaluated at CAMERA-clock times with the
    same lookup shift as DeviceMocap.pose_at (vision offset + drift + fine offset). Copy of the oracle's validated class
    (analysis/imu_bias/oracle/common.py MocapOrientation + accel_oracle.MocapPose) on the HEAD snapshot."""

    def __init__(self, dev):
        self.dev = dev
        self.slerp = Slerp(dev.t_ns.astype(np.float64), Rotation.from_quat(dev.quat_xyzw.astype(np.float64)))
        self.R_im = dev.T_imu_marker.R
        self.t_mocap = dev.t_ns

    def lookup_times(self, q_ns):
        q = np.asarray(q_ns, dtype=np.float64); d = self.dev
        return q + d.vision_offset_ns + d.drift_offset_ns + d.drift_rate_ns_per_ns * (q - d._drift_pivot_ns) + d.fine_offset_ns

    def valid(self, q_ns, max_gap_ns=30e6):
        tl = self.lookup_times(q_ns)
        ok = (tl >= self.t_mocap[0]) & (tl <= self.t_mocap[-1])
        idx = np.clip(np.searchsorted(self.t_mocap, tl, side="right") - 1, 0, len(self.t_mocap) - 2)
        return ok & ((self.t_mocap[idx + 1] - self.t_mocap[idx]) <= max_gap_ns)

    def R_world_imu(self, q_ns):
        tl = np.clip(self.lookup_times(q_ns), self.t_mocap[0], self.t_mocap[-1])
        return self.slerp(tl) * Rotation.from_matrix(self.R_im.T)

    def pos_imu(self, q_ns):
        tl = np.clip(self.lookup_times(q_ns), self.t_mocap[0], self.t_mocap[-1])
        pm = np.stack([np.interp(tl, self.t_mocap, self.dev.position[:, i]) for i in range(3)], 1)
        return pm + self.slerp(tl).apply(-(self.R_im.T @ self.dev.T_imu_marker.t))


def cum2(t_s, x):
    """double trapezoid integral of x (n,k) over t_s (n,) from 0."""
    dt = np.diff(t_s)[:, None]
    v = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(0.5 * (x[1:] + x[:-1]) * dt, 0)])
    return np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, 0)])
