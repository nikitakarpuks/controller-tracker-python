#!/usr/bin/env python3
"""
pose_fusion_heuristic_jump_check.py -- end-to-end validation of the jump-
detection additions to HeuristicPoseFusionFilter (src/pose_fusion_heuristic.py):
Case A's soft agreement ramp (warm-state pushback for a disagreement too small
to hit the pre-existing hard implausible_jump_pos_m/_rot_deg gate) and Case B's
3-frame cold-reacquisition confirmation (replacing "first candidate after a
long coast wins unconditionally").

Reuses this project's own existing offline-replay plumbing
(visualize_pose_fusion_validation.py's pattern: load real IMU + bootstrap
g_world from the low-motion frames of the FULL recording, align vision poses
into the world frame via mocap) rather than reinventing it -- unit coverage
for the pure state-machine/math pieces already lives in
tests/test_pose_fusion_heuristic.py; this script is the "does it actually
behave sanely replayed against this project's own real recording" check the
implementation plan called for.

Three checks:
  (a) SWAP PROBE -- same well-separated-timestamp swap simulation
      visualize_pose_fusion_validation.py uses for PoseFusionFilter, adapted
      to feed the RIGHT controller's pose into the LEFT controller's filter
      with siblings wired (set_siblings) -- this is exactly the sibling-
      collision check's target scenario, and previously had zero defense in
      HeuristicPoseFusionFilter (module docstring's own documented failure
      case: n_inliers=6/error=0.13px looking clean, pos_innov=998mm).
  (b) STABLE-TRACKING REGRESSION -- replays every real (non-gap) frame and
      reports the agreement-factor distribution: Case A should leave normal
      tracking essentially untouched (agreement ~1.0 the overwhelming
      majority of the time), only dropping for genuine disagreement.
  (c) REACQUISITION LATENCY -- for every real TRACKING LOST gap in the
      recording, counts how many post-gap frames the 3-frame confirmation
      window takes to CONFIRM (vs. the old behavior's implicit 1-frame
      "first candidate wins").

Usage: python pose_fusion_heuristic_jump_check.py [path/to/config.yml]
"""
import sys

import numpy as np

from accel_sign_check import _MAX_PAIR_DT_S, world_vision_poses
from accel_short_horizon_check import low_motion_bootstrap_g_world
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import accel_lever_arm_body
from src.imu_data import create_imu_calib_from_config, load_and_calibrate_controller_imu
from src.load_config import load_json_config, load_yaml_config
from src.pose_fusion_heuristic import HeuristicPoseFusionFilter
from src.transformations import Transform

from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
_BACKUP_POSE_CSV = "data/pose_log_2765frames.csv.bak_before_led_rerun"


class _FixedGWorld:
    def __init__(self, g):
        self.g_world = g


def _load_controller_data(ctrl_name, config, backup_poses, headset_mocap):
    from pathlib import Path
    mav0_root = Path(config["data"]["root"])
    imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
    ctrl_json_cfg = load_json_config(config["controllers"][ctrl_name]["config_path"])
    imu_calib = create_imu_calib_from_config(ctrl_json_cfg)
    lever_arm = accel_lever_arm_body(imu_calib)
    t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(
        mav0_root / imu_rel_path, ctrl_json_cfg, lag_ns=lag_ns)
    backup_wp = world_vision_poses(backup_poses[ctrl_name], headset_mocap)
    g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_gyro, accel_body, backup_wp)
    return dict(gyro_data=(t_gyro, gyro_body), accel_data=(t_gyro, accel_body),
                lever_arm=lever_arm, g_world=g_world)


def _make_filter(data, heuristic_cfg, shared_fusion_cfg, ctrl_name):
    cfg = {**shared_fusion_cfg, "fusion_heuristic": heuristic_cfg}
    return HeuristicPoseFusionFilter(data["gyro_data"], data["accel_data"], data["lever_arm"],
                                      _FixedGWorld(data["g_world"]), cfg, ctrl_name=ctrl_name)


def check_swap_probe(per_ctrl, heuristic_cfg, shared_fusion_cfg):
    print("\n=== (a) SWAP PROBE (sibling-collision check) ===")
    left_wp, right_wp = per_ctrl["left_controller"]["world_poses"], per_ctrl["right_controller"]["world_poses"]
    common_ts = sorted(set(left_wp.keys()) & set(right_wp.keys()))
    if len(common_ts) < 2:
        print("not enough overlapping timestamps -- skipped")
        return
    seps = np.array([np.linalg.norm(left_wp[ts].t - right_wp[ts].t) for ts in common_ts])
    i_best = int(np.argmax(seps[:-1]))
    ts0, ts1 = common_ts[i_best], common_ts[i_best + 1]

    left_filt = _make_filter(per_ctrl["left_controller"], heuristic_cfg, shared_fusion_cfg, "left_controller")
    right_filt = _make_filter(per_ctrl["right_controller"], heuristic_cfg, shared_fusion_cfg, "right_controller")
    left_filt.set_siblings([right_filt])
    right_filt.set_siblings([left_filt])

    # Give the sibling (right) some live, trustworthy state first.
    right_filt.try_update({"T_world_ctrl": Transform(right_wp[ts0].R, right_wp[ts0].t),
                            "error": 0.3, "confidence": 1.0}, ts0)
    # Now feed the LEFT filter (fresh, no prior state -- bootstrap branch)
    # the RIGHT controller's OWN pose at ts1 -- the exact identity-swap shape.
    swap_accepted = left_filt.try_update({"T_world_ctrl": Transform(right_wp[ts1].R, right_wp[ts1].t),
                                           "error": 0.3, "confidence": 1.0}, ts1)
    pos_sep = float(np.linalg.norm(right_wp[ts1].t - right_wp[ts0].t))
    print(f"separation at swap instant: {pos_sep * 1000:.0f}mm, gap={((ts1 - ts0) / 1e6):.1f}ms")
    print(f"left filter fed right's pose -> accepted={swap_accepted} "
          f"({'FAIL: swap NOT caught' if swap_accepted else 'PASS: swap caught by sibling check'})")


def check_stable_tracking(per_ctrl):
    print("\n=== (b) STABLE-TRACKING REGRESSION (Case A agreement distribution) ===")
    for ctrl_name, data in per_ctrl.items():
        world_poses = data["world_poses"]
        ts_sorted = sorted(world_poses.keys())
        filt = data["filter"]
        agreements = []
        outcomes = {}
        for i, ts in enumerate(ts_sorted):
            pose = world_poses[ts]
            filt.try_update({"T_world_ctrl": Transform(pose.R, pose.t), "error": 0.3,
                              "confidence": 1.0, "assignment": [(0, 0)] * 15}, ts)
            dbg = filt._last
            outcomes[dbg.get("outcome")] = outcomes.get(dbg.get("outcome"), 0) + 1
            if dbg.get("outcome") == "fused" and dbg.get("agreement") is not None:
                agreements.append(dbg["agreement"])
        print(f"[{ctrl_name}] {len(ts_sorted)} frames, outcomes={outcomes}")
        if agreements:
            agreements = np.array(agreements)
            print(f"  agreement: mean={agreements.mean():.3f} p10={np.percentile(agreements, 10):.3f} "
                  f"min={agreements.min():.3f} frac<1.0={(agreements < 0.999).mean() * 100:.1f}%")


def check_reacquisition_latency(per_ctrl):
    print("\n=== (c) REACQUISITION LATENCY (Case B 3-frame confirmation) ===")
    for ctrl_name, data in per_ctrl.items():
        world_poses = data["world_poses"]
        ts_sorted = sorted(world_poses.keys())
        filt = HeuristicPoseFusionFilter(data["gyro_data"], data["accel_data"], data["lever_arm"],
                                          _FixedGWorld(data["g_world"]),
                                          {**data["shared_fusion_cfg"], "fusion_heuristic": data["heuristic_cfg"]},
                                          ctrl_name=ctrl_name)
        latencies = []
        i = 0
        while i < len(ts_sorted) - 1:
            dt = (ts_sorted[i + 1] - ts_sorted[i]) / 1e9
            filt.try_update({"T_world_ctrl": Transform(world_poses[ts_sorted[i]].R, world_poses[ts_sorted[i]].t),
                              "error": 0.3, "confidence": 1.0, "assignment": [(0, 0)] * 15}, ts_sorted[i])
            if dt > _MAX_PAIR_DT_S:
                # Real gap -- count frames from the first post-gap candidate
                # until try_update finally returns True (CONFIRMED/fused) again.
                j = i + 1
                n_frames = 0
                while j < len(ts_sorted):
                    ts = ts_sorted[j]
                    ok = filt.try_update({"T_world_ctrl": Transform(world_poses[ts].R, world_poses[ts].t),
                                           "error": 0.3, "confidence": 1.0, "assignment": [(0, 0)] * 15}, ts)
                    n_frames += 1
                    if ok:
                        break
                    j += 1
                latencies.append((dt, n_frames))
                i = j
            i += 1
        print(f"[{ctrl_name}] {len(latencies)} real gaps (dt>{_MAX_PAIR_DT_S * 1000:.0f}ms)")
        for gap_dt, n_frames in latencies:
            print(f"  gap={gap_dt * 1000:.0f}ms -> reacquired after {n_frames} post-gap frame(s)")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)
    poses, _errors = load_pose_csv(config["debug"]["pose_csv"])
    from pathlib import Path
    backup_poses, _ = load_pose_csv(_BACKUP_POSE_CSV) if Path(_BACKUP_POSE_CSV).exists() else (poses, None)
    headset_mocap = load_device_mocap("headset")

    heuristic_cfg = config.get("fusion_heuristic", {})
    shared_fusion_cfg = config.get("fusion", {})

    per_ctrl = {}
    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses:
            continue
        data = _load_controller_data(ctrl_name, config, backup_poses, headset_mocap)
        data["world_poses"] = world_vision_poses(poses[ctrl_name], headset_mocap)
        data["shared_fusion_cfg"] = shared_fusion_cfg
        data["heuristic_cfg"] = heuristic_cfg
        data["filter"] = _make_filter(data, heuristic_cfg, shared_fusion_cfg, ctrl_name)
        per_ctrl[ctrl_name] = data
        print(f"[{ctrl_name}] loaded {len(data['world_poses'])} vision poses, g_world={data['g_world']}")

    if "left_controller" in per_ctrl and "right_controller" in per_ctrl:
        per_ctrl["left_controller"]["filter"].set_siblings([per_ctrl["right_controller"]["filter"]])
        per_ctrl["right_controller"]["filter"].set_siblings([per_ctrl["left_controller"]["filter"]])
        check_swap_probe(per_ctrl, heuristic_cfg, shared_fusion_cfg)

    check_stable_tracking(per_ctrl)
    check_reacquisition_latency(per_ctrl)


if __name__ == "__main__":
    main()
