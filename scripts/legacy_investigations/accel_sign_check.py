#!/usr/bin/env python3
"""
accel_sign_check.py -- Step 3, sub-stage 1 of the IMU+vision fusion plan: the
sign/convention sanity check for the accel term, run BEFORE any lever-arm
complexity -- with lever arm assumed zero (accelerometer treated as
co-located with the body origin vision tracks), does gravity-corrected accel
roughly track vision-derived velocity change over short (single-frame-gap)
windows? This isolates "did we get the basic physics/sign conventions right"
from "did the lever-arm estimation converge" (sub-stage 2), so a bug in one
doesn't get misattributed to the other.

Frame: accel_body (src/imu_data.load_and_calibrate_controller_imu) lives in
the controller's own LED/body frame -- same frame world_pose(headset,t).
compose(T_ctrl_vision(t)) gives (UN-bridged, see gyro_preint_check.py's
module docstring for why no bridge is needed here, same reasoning applies to
accel).

g_world is UNKNOWN at this stage (that's what sub-stage 2 solves for jointly
with the lever arm) -- this script bootstraps a crude estimate via long-window
averaging (g_world ~= -mean(R_world_body(t) @ accel_body(t)) over the whole
recording, valid because a hand-held controller's net WORLD-frame acceleration
averages out over a long recording, so the mean of rotated accel is
dominated by gravity). This bootstrap is NOT the sub-stage-2 result -- it only
needs to be roughly right for this check, since a wrong ROTATION convention
(the thing actually being tested here) would corrupt the DYNAMIC part of the
signal regardless of the exact gravity constant used, while a small/moderate
error in the constant g_world mostly just shifts the correlation's intercept,
not its slope/sign.

Method: for each consecutive accepted-vision-frame gap [t_k, t_k+1], compute
  Delta_v_accel(k) = integrate_accel_segment(...)   (world frame, trapezoidal)
  Delta_v_vision(k) = v(t_k+1) - v(t_k)              (v = central-difference
                                                       vision velocity, world frame)
and cross-correlate the two per-axis, plus a time-series overlay -- same
cross-correlation spirit as check_headset_imu_axis.py / Step 1's gyro check,
now applied to accel/velocity instead of gyro/orientation.

Usage: python accel_sign_check.py [path/to/config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import integrate_accel_segment, load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import world_pose
from src.transformations import Transform

from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
_AXIS_NAMES = ("x", "y", "z")
_MAX_PAIR_DT_S = 0.15  # guard against TRACKING LOST-adjacent frame gaps contaminating
                        # the central-difference velocity estimate (Step 1 found the
                        # nominal gap is ~16-33ms; anything much larger than that is a
                        # real acquisition gap, not just this recording's normal cadence)


def world_vision_poses(ctrl_poses: dict, headset_mocap) -> dict:
    """{ts_ns: Transform} world-frame (mocap-world), UN-bridged vision pose per
    accepted frame -- world_pose(headset,t).compose(T_ctrl_vision(t)), same
    frame accel_body/gyro_body live in (controller LED/body frame). Frames
    with no headset mocap coverage at t are dropped."""
    out = {}
    for ts_ns, T_ctrl_vision in ctrl_poses.items():
        T_world_headsetImu = world_pose(headset_mocap, ts_ns)
        if T_world_headsetImu is None:
            continue
        out[ts_ns] = T_world_headsetImu.compose(T_ctrl_vision)
    return out


def central_diff_velocity(ts_sorted, pos_world: dict, max_pair_dt_s=_MAX_PAIR_DT_S) -> dict:
    """{ts_ns: v (3,) m/s} central-difference world-frame velocity at each
    INTERIOR frame (needs both neighbors present with a bounded gap) --
    v(t_k) = (p(t_k+1) - p(t_k-1)) / (t_k+1 - t_k-1)."""
    out = {}
    for i in range(1, len(ts_sorted) - 1):
        tm, tk, tp = ts_sorted[i - 1], ts_sorted[i], ts_sorted[i + 1]
        dt = (tp - tm) / 1e9
        if dt <= 0 or dt > 2 * max_pair_dt_s:
            continue
        if (tk - tm) / 1e9 > max_pair_dt_s or (tp - tk) / 1e9 > max_pair_dt_s:
            continue
        out[tk] = (pos_world[tp] - pos_world[tm]) / dt
    return out


def bootstrap_g_world(t_accel, accel_body, world_poses: dict) -> np.ndarray:
    """g_world ~= -mean(R_world_body(t) @ accel_body(t)) over every accepted
    vision frame -- see module docstring for why this crude long-window
    average is adequate for a sign-convention check, not a real gravity
    estimate (that's sub-stage 2's job)."""
    rows = []
    for ts_ns, T in world_poses.items():
        if ts_ns < t_accel[0] or ts_ns > t_accel[-1]:
            continue
        acc = np.array([np.interp(ts_ns, t_accel, accel_body[:, i]) for i in range(3)])
        rows.append(T.R @ acc)
    return -np.mean(np.stack(rows), axis=0)


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def trimmed_pearson(a, b, pct=99.0) -> float:
    """Pearson correlation after dropping gaps in the top (100-pct)% by
    |a| or |b| -- the overlay plot shows a handful of huge isolated spikes
    (same pattern Step 1 found: bad VISION frames, not bad gyro/accel) that
    dominate ordinary Pearson; this checks whether the BULK of the signal
    (everything but those outliers) actually agrees."""
    thresh_a = np.percentile(np.abs(a), pct)
    thresh_b = np.percentile(np.abs(b), pct)
    keep = (np.abs(a) <= thresh_a) & (np.abs(b) <= thresh_b)
    return float(np.corrcoef(a[keep], b[keep])[0, 1]), int(keep.sum())


def spearman(a, b) -> float:
    return float(spearmanr(a, b).correlation)


def plot_overlay(ctrl_name, t_mid_ns, dv_accel, dv_vision, out_path: Path):
    t_s = (t_mid_ns - t_mid_ns[0]) / 1e9
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_s, dv_accel[:, i], label="gravity-corrected accel (integrated)", color="darkorange", linewidth=0.8)
        ax.plot(t_s, dv_vision[:, i], label="vision-derived (central diff)", color="steelblue",
                 linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f"dv {_AXIS_NAMES[i]} (m/s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{ctrl_name}: accel-predicted vs vision-derived per-gap velocity change (world frame)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_scatter(ctrl_name, dv_accel, dv_vision, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, ax in enumerate(axes):
        ax.scatter(dv_vision[:, i], dv_accel[:, i], s=6, alpha=0.3, color="steelblue")
        lim = max(np.abs(dv_vision[:, i]).max(), np.abs(dv_accel[:, i]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"vision dv {_AXIS_NAMES[i]} (m/s)")
        ax.set_ylabel(f"accel dv {_AXIS_NAMES[i]} (m/s)")
        ax.set_title(f"r={pearson(dv_vision[:, i], dv_accel[:, i]):+.3f}")
    fig.suptitle(f"{ctrl_name}: per-axis agreement (dashed = perfect agreement)")
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
        t_accel, _gyro_body, accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)

        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        ts_sorted = sorted(world_poses.keys())
        pos_world = {ts: world_poses[ts].t for ts in ts_sorted}

        g_world = bootstrap_g_world(t_accel, accel_body, world_poses)
        print(f"[{ctrl_name}] bootstrap g_world = {g_world} -- |g|={np.linalg.norm(g_world):.3f} m/s^2 "
              f"(expect ~9.81 if rotation convention is sane)")

        v_vision = central_diff_velocity(ts_sorted, pos_world)
        v_keys = sorted(v_vision.keys())

        t_mid, dv_accel, dv_vision = [], [], []
        n_skipped = 0
        for tk, tk1 in zip(v_keys[:-1], v_keys[1:]):
            dv_v = v_vision[tk1] - v_vision[tk]
            dv_a = integrate_accel_segment(t_accel, accel_body, tk, tk1,
                                            world_poses[tk].R, world_poses[tk1].R, g_world)
            if dv_a is None:
                n_skipped += 1
                continue
            t_mid.append((tk + tk1) // 2)
            dv_accel.append(dv_a)
            dv_vision.append(dv_v)

        t_mid = np.array(t_mid, dtype=np.int64)
        dv_accel = np.array(dv_accel)
        dv_vision = np.array(dv_vision)
        n = len(t_mid)
        if n == 0:
            print(f"[{ctrl_name}] no gaps covered -- skipping")
            continue

        corr = [pearson(dv_vision[:, i], dv_accel[:, i]) for i in range(3)]
        corr_sp = [spearman(dv_vision[:, i], dv_accel[:, i]) for i in range(3)]
        corr_tr = [trimmed_pearson(dv_vision[:, i], dv_accel[:, i]) for i in range(3)]
        print(f"[{ctrl_name}] {n} gaps: per-axis correlation (accel-predicted dv vs vision-derived dv)")
        print(f"    Pearson         = [{corr[0]:+.3f}, {corr[1]:+.3f}, {corr[2]:+.3f}]")
        print(f"    Spearman (rank) = [{corr_sp[0]:+.3f}, {corr_sp[1]:+.3f}, {corr_sp[2]:+.3f}]")
        print(f"    Pearson, top-1% |dv| outliers dropped = "
              f"[{corr_tr[0][0]:+.3f} (n={corr_tr[0][1]}), {corr_tr[1][0]:+.3f} (n={corr_tr[1][1]}), "
              f"{corr_tr[2][0]:+.3f} (n={corr_tr[2][1]})]")

        plot_overlay(ctrl_name, t_mid, dv_accel, dv_vision, out_dir / f"accel_sign_{ctrl_name}_overlay.png")
        plot_scatter(ctrl_name, dv_accel, dv_vision, out_dir / f"accel_sign_{ctrl_name}_scatter.png")
        print(f"[{ctrl_name}] saved plots to {out_dir}/accel_sign_{ctrl_name}_{{overlay,scatter}}.png\n")


if __name__ == "__main__":
    main()
