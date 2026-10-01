#!/usr/bin/env python3
"""
accel_lever_arm_solve.py -- Step 3, sub-stage 2 of the IMU+vision fusion
plan: a small standalone least-squares solve for the accelerometer's lever
arm r, an additional bias refinement b_a, and world-frame gravity g_world --
with position/orientation states FIXED at vision (Step 0, world-frame,
un-bridged) values, not jointly optimized (that's Step 4).

Physics (rigid-body kinematics, accelerometer offset r from the body origin
vision tracks):
    a_point_world(t) = a_center_world(t) + R_world_body(t) @ [alpha(t) x r + omega(t) x (omega(t) x r)]
    accel_body_reading(t) = R_body_world(t) @ (a_point_world(t) - g_world) + b_a
Substituting and rearranging (this IS linear in the unknowns [g_world; r; b_a]):
    accel_body(t) - R_body_world(t) @ a_center_world(t)
        = -R_body_world(t) @ g_world + [skew(alpha(t)) + skew(omega(t))@skew(omega(t))] @ r + b_a
One 3-row block per accepted vision frame with valid neighbors on both sides;
stacked and solved via ordinary least squares (np.linalg.lstsq) -- no
nonlinear optimizer needed since the model is linear in these 9 unknowns.

Inputs, all held fixed (not solved here):
  - R_world_body(t), used both to rotate g_world into body frame and as part
    of a_center_world's world-frame estimate -- Step 0's world-frame vision
    orientation (un-bridged, see gyro_preint_check.py's module docstring).
  - a_center_world(t) -- the WORLD-frame linear acceleration of the vision-
    tracked body origin, from a non-uniform 3-point central second-difference
    of vision position (Steps 0-1's "fixed at vision+gyro values"). This is
    the noisiest input by far: mm-level PnP position noise gets amplified by
    ~1/dt^2 (dt ~ 16-33ms here) into several m/s^2 of acceleration noise,
    comparable to gravity itself -- expected, and exactly why this sub-stage
    exists in isolation: to find out whether thousands of frames' worth of
    (hopefully close to zero-mean) noise still lets a stable r fall out, or
    whether it doesn't and the whole accel-term idea needs rethinking before
    Step 4 tries to use it.
  - omega(t), alpha(t) -- angular velocity/acceleration from GYRO (not
    vision), since gyro directly measures body angular velocity far more
    cleanly than differentiating vision orientation would; alpha is a
    non-uniform 3-point central difference of gyro-interpolated omega at the
    same three frame timestamps a_center_world uses, for internal consistency.

Test: does r converge to something with plausible ~85mm magnitude (the
factory InertialSensors Rt accelerometer entry's translation magnitude --
see data/controllers/mocap_bridge_fit.txt: ~80-83mm from the mocap bridge
fit's own tz -- a sanity check on MAGNITUDE only, not direction, which is
documented elsewhere as unreliable)? Also checks stability by re-solving on
several disjoint time chunks -- if r's magnitude swings wildly between
chunks, that's the "stop and debug before the full joint solve" signal the
plan calls for, not a single global number.

Usage: python accel_lever_arm_solve.py [path/to/config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accel_sign_check import world_vision_poses, _MAX_PAIR_DT_S
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config

from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
_EXPECTED_LEVER_ARM_MM = 85.0  # factory-Rt-magnitude sanity target, see module docstring
_N_STABILITY_CHUNKS = 5


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


def nonuniform_2nd_diff(p_m, p_k, p_p, dt1, dt2):
    """Standard 3-point non-uniform-grid central second-difference estimate
    of d^2p/dt^2 at t_k, given samples at t_k-dt1, t_k, t_k+dt2."""
    return 2.0 * (dt1 * p_p - (dt1 + dt2) * p_k + dt2 * p_m) / (dt1 * dt2 * (dt1 + dt2))


def build_design_rows(ts_sorted, world_poses: dict, t_gyro, gyro_body, t_accel, accel_body,
                       max_pair_dt_s=_MAX_PAIR_DT_S):
    """One (A_row (3,9), rhs_row (3,), t_k) tuple per interior accepted frame
    with valid bounded-gap neighbors and full gyro/accel coverage. Unknown
    order: x = [g_world(3); r(3); b_a(3)]."""
    rows_A, rows_rhs, rows_t = [], [], []
    for i in range(1, len(ts_sorted) - 1):
        tm, tk, tp = ts_sorted[i - 1], ts_sorted[i], ts_sorted[i + 1]
        dt1, dt2 = (tk - tm) / 1e9, (tp - tk) / 1e9
        if dt1 <= 0 or dt2 <= 0 or dt1 > max_pair_dt_s or dt2 > max_pair_dt_s:
            continue
        if tk < t_gyro[0] or tk > t_gyro[-1] or tm < t_gyro[0] or tp > t_gyro[-1]:
            continue
        if tk < t_accel[0] or tk > t_accel[-1]:
            continue

        p_m, p_k, p_p = world_poses[tm].t, world_poses[tk].t, world_poses[tp].t
        a_center_world = nonuniform_2nd_diff(p_m, p_k, p_p, dt1, dt2)

        om = np.array([[np.interp(t, t_gyro, gyro_body[:, ax]) for ax in range(3)] for t in (tm, tk, tp)])
        omega_k = om[1]
        alpha_k = nonuniform_2nd_diff(om[0], om[1], om[2], dt1, dt2)

        accel_meas = np.array([np.interp(tk, t_accel, accel_body[:, ax]) for ax in range(3)])
        R_k = world_poses[tk].R

        M_lever = skew(alpha_k) + skew(omega_k) @ skew(omega_k)
        A_row = np.hstack([-R_k, M_lever, np.eye(3)])
        rhs_row = accel_meas - R_k @ a_center_world

        rows_A.append(A_row)
        rows_rhs.append(rhs_row)
        rows_t.append(tk)
    return rows_A, rows_rhs, rows_t


def solve(rows_A, rows_rhs):
    A = np.vstack(rows_A)
    b = np.concatenate(rows_rhs)
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    g_world, r, b_a = x[0:3], x[3:6], x[6:9]
    resid_rms = float(np.sqrt(np.mean((A @ x - b) ** 2)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return g_world, r, b_a, resid_rms, cond, rank


def plot_stability(ctrl_name: str, chunk_labels, r_mag_mm, g_mag, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(chunk_labels, r_mag_mm, color="steelblue")
    axes[0].axhline(_EXPECTED_LEVER_ARM_MM, color="crimson", linestyle="--",
                     label=f"factory-magnitude ref ({_EXPECTED_LEVER_ARM_MM:.0f}mm)")
    axes[0].set_ylabel("|r| (mm)")
    axes[0].set_title("lever arm magnitude per chunk")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(chunk_labels, g_mag, color="darkorange")
    axes[1].axhline(9.81, color="crimson", linestyle="--", label="9.81 m/s^2")
    axes[1].set_ylabel("|g_world| (m/s^2)")
    axes[1].set_title("gravity magnitude per chunk")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"{ctrl_name}: lever-arm/gravity solve stability across {len(chunk_labels) - 1} time chunks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step3")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    if not pose_csv_path or not Path(pose_csv_path).exists():
        raise SystemExit(f"debug.pose_csv not set or missing ({pose_csv_path})")

    poses, _errors = load_pose_csv(pose_csv_path)
    headset_mocap = load_device_mocap("headset")
    mav0_root = Path(config["data"]["root"])

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses or ctrl_name not in _IMU_FILES:
            continue
        imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
        imu_path = mav0_root / imu_rel_path
        if not imu_path.exists():
            print(f"[{ctrl_name}] IMU file not found ({imu_path}) -- skipping")
            continue

        ctrl_cfg = config["controllers"][ctrl_name]
        ctrl_json_cfg = load_json_config(ctrl_cfg["config_path"])
        t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)
        t_accel = t_gyro  # same sample timestamps for this IMU (single stream)

        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        ts_sorted = sorted(world_poses.keys())

        rows_A, rows_rhs, rows_t = build_design_rows(ts_sorted, world_poses, t_gyro, gyro_body, t_accel, accel_body)
        n = len(rows_A)
        if n < 20:
            print(f"[{ctrl_name}] only {n} usable frames -- skipping")
            continue

        g_world, r, b_a, resid_rms, cond, rank = solve(rows_A, rows_rhs)
        print(f"[{ctrl_name}] full-recording solve ({n} frames, design matrix rank={rank}/9, "
              f"cond={cond:.3g}, resid_rms={resid_rms:.3f} m/s^2)")
        print(f"    r = {r} m  |r|={np.linalg.norm(r) * 1000:.1f} mm  (factory-magnitude ref ~{_EXPECTED_LEVER_ARM_MM:.0f}mm)")
        print(f"    g_world = {g_world}  |g_world|={np.linalg.norm(g_world):.3f} m/s^2  (expect ~9.81)")
        print(f"    b_a = {b_a} m/s^2  |b_a|={np.linalg.norm(b_a):.4f} m/s^2")

        rows_t = np.array(rows_t)
        edges = np.linspace(rows_t.min(), rows_t.max(), _N_STABILITY_CHUNKS + 1)
        chunk_labels, r_mag_mm, g_mag = [], [], []
        for c in range(_N_STABILITY_CHUNKS):
            mask = (rows_t >= edges[c]) & (rows_t <= edges[c + 1])
            if mask.sum() < 20:
                continue
            A_c = [rows_A[i] for i in range(n) if mask[i]]
            b_c = [rows_rhs[i] for i in range(n) if mask[i]]
            g_c, r_c, ba_c, rms_c, cond_c, rank_c = solve(A_c, b_c)
            chunk_labels.append(f"chunk{c}\n(n={mask.sum()})")
            r_mag_mm.append(np.linalg.norm(r_c) * 1000)
            g_mag.append(np.linalg.norm(g_c))
            print(f"    chunk {c} (n={mask.sum()}): |r|={np.linalg.norm(r_c) * 1000:.1f}mm  "
                  f"|g_world|={np.linalg.norm(g_c):.3f}  resid_rms={rms_c:.3f}")

        if chunk_labels:
            plot_stability(ctrl_name, chunk_labels, r_mag_mm, g_mag,
                            out_dir / f"accel_lever_arm_{ctrl_name}_stability.png")
            print(f"[{ctrl_name}] saved stability plot to {out_dir}/accel_lever_arm_{ctrl_name}_stability.png\n")


if __name__ == "__main__":
    main()
