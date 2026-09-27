"""Accelerometer lever-arm verification, mocap-derived a_center_world instead of vision-PnP.

Same physics/design-row model as accel_lever_arm_solve.py (unmodified: skew(alpha)+skew(omega)@skew(omega)
lever-arm term, gyro-derived omega/alpha, R_k from vision's own fused pose.csv -- rotation is a direct
per-frame PnP estimate, not differentiated, so it isn't the noisy part and stays as-is). ONLY a_center_world's
source changes: mocap-derived room-frame LED-reference position (T_room_ctrlLedRef = world_pose(headset,t)
.compose(mocap_gt[t].compose(bridge.inverse()))), fit locally with a windowed quadratic (Savitzky-Golay-style)
before taking the analytic 2nd derivative -- a raw 3-point central difference at mocap's own ~8.3ms spacing and
~1mm jitter would still give noise ~1mm/(8.3ms)^2 ~= 14 m/s^2, comparable to gravity itself (verified below).

Optionally restricts design rows to frames whose gyro rate exceeds a threshold (the lever-arm term's own SNR
scales with rotation rate) and pools across recordings.
"""
import sys, argparse
from pathlib import Path
import numpy as np

REPO = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
sys.path.insert(0, str(REPO))
import os; os.chdir(REPO)

import evaluate_mocap as em
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import load_mocap_bridge, load_device_mocap_from_config, world_pose, controller_imu_files
from src.imu_data import load_and_calibrate_controller_imu
from src.transformations import Transform

DOWNLOADS = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
BASE = REPO / "visualization" / "evaluate_2026-09-27_full"
RECS = {"static_dark": "euroc_recording_20260826173103_static_dark",
        "static_easy": "euroc_recording_20260826173932_static_easy",
        "static_medium": "euroc_recording_20260826174213_static_medium",
        "static_hard": "euroc_recording_20260826174510_static_hard",
        "walk_dark": "euroc_recording_20260826173350_walk_dark",
        "walk_easy": "euroc_recording_20260826175226_walk_easy",
        "walk_medium": "euroc_recording_20260826175510_walk_medium",
        "walk_hard": "euroc_recording_20260826180039_walk_hard"}
CTRLS = ("left_controller", "right_controller")
cfg = load_yaml_config(str(REPO / "config" / "config.yml"))
_IMU_FILES = controller_imu_files(cfg)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def local_quad_accel(t_query, t_all, p_all, half_window_s):
    """Analytic 2nd derivative of a local quadratic LS fit p(t) = p0 + v0*(t-t_query) + 0.5*a0*(t-t_query)^2
    over samples within half_window_s of t_query. Returns a0 (3,) or None if too few samples."""
    m = np.abs(t_all - t_query) <= half_window_s
    n = m.sum()
    if n < 6:
        return None, n
    tt = (t_all[m] - t_query)
    A = np.stack([np.ones_like(tt), tt, 0.5 * tt ** 2], axis=1)
    # one LS solve per axis (shared design matrix)
    sol, *_ = np.linalg.lstsq(A, p_all[m], rcond=None)
    return sol[2], n  # (3,) acceleration


def noise_floor_check(t_all, half_window_s, sigma_mm=1.0, trials=200, seed=0):
    """Synthetic check: feed pure Gaussian noise (no real signal) through the SAME quadratic fit and report the
    resulting 'phantom acceleration' std -- the effective noise floor this method would see."""
    rng = np.random.default_rng(seed)
    t_query = t_all[len(t_all) // 2]
    accs = []
    for _ in range(trials):
        noise = rng.normal(0, sigma_mm / 1000.0, size=(len(t_all), 3))
        a, n = local_quad_accel(t_query, t_all, noise, half_window_s)
        if a is not None:
            accs.append(a)
    accs = np.array(accs)
    return float(np.std(accs)) if len(accs) else float("nan"), n


def build_rows(rec, ctrl, half_window_s, min_gyro_dps=0.0, R_source="fused"):
    d = BASE / rec
    bridge = load_mocap_bridge(cfg["controllers"][ctrl]["mocap_bridge_path"])
    gt = em.load_mocap_gt_csv(d / f"{rec}_mocap_gt" / f"{ctrl}_mocap_gt.csv")
    vision = em.load_pose_csv(d / (f"{rec}_pose.csv" if R_source == "fused" else f"{rec}_vision_pose.csv")).get(ctrl, {})
    rec_root = DOWNLOADS / RECS[rec]
    headset_mocap, why = load_device_mocap_from_config(rec_root, "headset", cfg)
    if headset_mocap is None:
        return [], [], [], why
    imu_rel_path, lag_ns = _IMU_FILES[ctrl]
    ctrl_json_cfg = load_json_config(cfg["controllers"][ctrl]["config_path"])
    t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(rec_root / "mav0" / imu_rel_path, ctrl_json_cfg, lag_ns=lag_ns)
    t_gyro = t_gyro.astype(np.int64)

    # dense mocap room-frame LED-reference position series, at every mocap_gt.csv timestamp
    mocap_ts, mocap_pos = [], []
    for ts, T_gt in sorted(gt.items()):
        T_wh = world_pose(headset_mocap, ts)
        if T_wh is None:
            continue
        T_room = T_wh.compose(T_gt.compose(bridge.inverse()))
        mocap_ts.append(ts); mocap_pos.append(T_room.t)
    if len(mocap_ts) < 20:
        return [], [], [], "too few mocap samples"
    mocap_ts = np.array(mocap_ts, dtype=np.float64) / 1e9  # seconds
    mocap_pos = np.array(mocap_pos)

    rows_A, rows_rhs, rows_t = [], [], []
    ts_sorted = sorted(vision.keys())
    for tk in ts_sorted:
        if tk < t_gyro[0] or tk > t_gyro[-1]:
            continue
        omega_k = np.array([np.interp(tk, t_gyro, gyro_body[:, ax]) for ax in range(3)])
        if np.degrees(np.linalg.norm(omega_k)) < min_gyro_dps:
            continue
        # alpha via a slightly wider gyro-only finite window (gyro is dense/clean, unaffected by this fix)
        dt_a = 0.02
        om_m = np.array([np.interp(tk - dt_a, t_gyro, gyro_body[:, ax]) for ax in range(3)])
        om_p = np.array([np.interp(tk + dt_a, t_gyro, gyro_body[:, ax]) for ax in range(3)])
        alpha_k = (om_p - om_m) / (2 * dt_a)

        a_center_world, n_used = local_quad_accel(tk / 1e9, mocap_ts, mocap_pos, half_window_s)
        if a_center_world is None:
            continue
        accel_meas = np.array([np.interp(tk, t_gyro, accel_body[:, ax]) for ax in range(3)])
        R_k = vision[tk].R  # kept from vision's own per-frame PnP estimate -- not differentiated, not the noisy part

        M_lever = skew(alpha_k) + skew(omega_k) @ skew(omega_k)
        A_row = np.hstack([-R_k, M_lever, np.eye(3)])
        rhs_row = accel_meas - R_k @ a_center_world
        rows_A.append(A_row); rows_rhs.append(rhs_row); rows_t.append(tk)
    return rows_A, rows_rhs, rows_t, None


def solve(rows_A, rows_rhs):
    A = np.vstack(rows_A); b = np.concatenate(rows_rhs)
    x, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    g_world, r, b_a = x[0:3], x[3:6], x[6:9]
    resid_rms = float(np.sqrt(np.mean((A @ x - b) ** 2)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return g_world, r, b_a, resid_rms, cond, rank


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", nargs="*", default=["walk_hard"])
    ap.add_argument("--window", type=float, default=0.15, help="half-window (s) for the local quadratic fit")
    ap.add_argument("--min-gyro-dps", type=float, default=0.0)
    args = ap.parse_args()

    # go/no-go: synthetic noise floor at this window, real mocap sample spacing
    d0 = BASE / args.rec[0]
    gt0 = em.load_mocap_gt_csv(d0 / f"{args.rec[0]}_mocap_gt" / "left_controller_mocap_gt.csv")
    t_all = np.array(sorted(gt0.keys()), dtype=np.float64) / 1e9
    dt_med = np.median(np.diff(t_all))
    floor, n_win = noise_floor_check(t_all, args.window)
    print(f"mocap sample spacing median={dt_med*1000:.1f}ms | noise floor @ window={args.window*1000:.0f}ms "
          f"(n~{n_win} samples/window): phantom accel std = {floor:.3f} m/s^2 (vs gravity 9.81)\n")

    for rec in args.rec:
        for ctrl in CTRLS:
            rows_A, rows_rhs, rows_t, err = build_rows(rec, ctrl, args.window, args.min_gyro_dps)
            if err:
                print(f"[{rec}/{ctrl}] SKIP: {err}"); continue
            n = len(rows_A)
            if n < 20:
                print(f"[{rec}/{ctrl}] only {n} usable rows -- skipping"); continue
            g_world, r, b_a, resid_rms, cond, rank = solve(rows_A, rows_rhs)
            print(f"[{rec}/{ctrl}] n={n} rank={rank}/9 cond={cond:.2e} resid_rms={resid_rms:.3f} m/s^2  "
                  f"|g_world|={np.linalg.norm(g_world):.3f} (want 9.81)  |r|={np.linalg.norm(r)*1000:.1f}mm (factory ref ~85mm)  "
                  f"|b_a|={np.linalg.norm(b_a):.3f}")
