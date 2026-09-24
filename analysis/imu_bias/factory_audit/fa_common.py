"""Shared helpers for the factory-calibration audit (investigator F).
Reuses the oracle study's validated machinery unchanged: MocapOrientation / load_mocap_device (common.py),
interval_terms / solve (gyro_oracle.py), window_system / project / MocapPose (accel_oracle.py).
The ONLY new thing: those routines are driven by an arbitrary candidate correction of the RAW CSV stream
instead of the loader's fixed one.  Frame note: gs = 'sensor frame' of the oracle = the CSV axes after the
candidate correction (the oracle defines gs = DIAG_FLIP @ gyro_body = M @ CSV + b in the CSV axes)."""
import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *                       # noqa: F401,F403  (REPO, CONFIG, rec_dir, load_mocap_device, MocapOrientation, ...)
import gyro_oracle as GO                   # noqa: E402
import accel_oracle as AO                  # noqa: E402
from src.imu_data import load_imu_csv, create_imu_calib_from_config  # noqa: E402

FA = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit/"
MODERATE = ["static_dark", "walk_dark", "static_easy", "static_medium", "walk_easy", "walk_medium"]
ALL8 = REC_NAMES
D = np.diag([1.0, -1.0, -1.0])


def raw_csv(rdir, ctrl):
    """(t_ns lagged with the shipped lag, gyro_csv (N,3), accel_csv (N,3)) -- NO factory correction, NO axis flip."""
    p = rdir / "mav0" / {"left_controller": "imu1", "right_controller": "imu2"}[ctrl] / "data.csv"
    t, g, a = load_imu_csv(p)
    return t + controller_imu_lag_ns(ctrl, CONFIG), g.astype(np.float64), a.astype(np.float64)


def factory(ctrl, entry):
    """(Mg, bg, Ma, ba) T=0 factory terms of the given InertialSensors entry (0=ICM20602, 1=Undefined)."""
    cfg = load_json_config(str(REPO / CONFIG["controllers"][ctrl]["config_path"]))
    c = create_imu_calib_from_config(cfg, entry_index=entry)
    return c.gyro.mix0.copy(), c.gyro.bias0.copy(), c.accel.mix0.copy(), c.accel.bias0.copy()


def apply(stream, A, beta, order="mix_then_bias"):
    """affine candidate correction of an (N,3) stream."""
    if order == "mix_then_bias":
        return (A @ stream.T).T + beta
    return (A @ (stream + beta).T).T          # bias_then_mix


def baseline_stream(stream, ctrl, entry, which):
    Mg, bg, Ma, ba = factory(ctrl, entry)
    return apply(stream, Mg, bg) if which == "gyro" else apply(stream, Ma, ba)


def gyro_terms_stream(name, ctrl, t, gs, tau=0.5):
    """Exactly gyro_oracle.build, but driven by a supplied sensor-frame stream gs."""
    rdir = rec_dir(name)
    mo = MocapOrientation(load_mocap_device(rdir, ctrl))
    dtn = int(tau * 1e9)
    grid = np.arange(t[0] + 0.5e9, t[-1] - 0.5e9, dtn).astype(np.int64)
    ok = mo.valid(grid) & mo.valid(grid + dtn) & mo.valid(grid + dtn // 2)
    grid = grid[ok]
    R0 = mo.R_world_imu(grid); R1 = mo.R_world_imu(grid + dtn)
    T0, R, JB, JKs, WB = [], [], [], [], []
    for i, t0 in enumerate(grid):
        Rg, Jb, JK, wbar = GO.interval_terms(t, gs, int(t0), int(t0) + dtn)
        Rm = (R0[i].inv() * R1[i]).as_matrix()
        R.append(Rotation.from_matrix(Rm.T @ Rg).as_rotvec()); JB.append(Jb); JKs.append(JK); T0.append(t0); WB.append(wbar)
    return dict(t0=np.array(T0), r0=np.array(R), Jb=np.array(JB), JK=np.array(JKs), wbar=np.array(WB), tau=tau)


def accel_windows_stream(name, ctrl, t, fs, W_s=2.0):
    """Exactly accel_oracle.run, but driven by a supplied sensor-frame accel stream fs."""
    rdir = rec_dir(name)
    dev = load_mocap_device(rdir, ctrl); mo = AO.MocapPose(dev)
    W = int(W_s * 1e9); wins = []
    w_dummy = np.zeros_like(fs)                 # lever arm disabled (use_lever=False) as in the oracle's headline numbers
    for T0 in np.arange(t[0] + 0.5e9, t[-1] - W - 0.5e9, W):
        r = AO.window_system(t, fs, w_dummy, mo, dev.t_ns, int(T0), W, False)
        if r is None:
            continue
        Ap, yp = AO.project(r[0], r[1], r[2])
        wins.append(dict(t0=int(T0), AtA=Ap.T @ Ap, Aty=Ap.T @ yp, yty=float(yp @ yp), n=len(yp), Am=np.abs(r[1]).mean()))
    return wins
