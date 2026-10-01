#!/usr/bin/env python3
"""build_static_light_map.py -- Phase 3 offline tool for the mocap-based
static-light (ceiling-lamp) exclusion feature (see the approved plan under
.claude/plans/). Replays one camera's raw images alongside two CSVs main.py
can already produce (debug.vision_pose_csv, debug.led_detections_csv) to
determine, per frame: which controllers are "warm" this exact frame (a
same-frame vision solve above a confidence threshold, not a carried-over/
coasted one) and which pixels are already claimed by a tracked controller's
LEDs -- then runs a permissive full-image "survey" blob detection pass,
subtracts claimed blobs, casts each remaining blob's pixel into a room-frame
ray (via the headset's mocap pose), and accumulates votes into a
StaticLightVoxelAccumulator.

This makes zero changes to the production tracking pipeline: it only
consumes main.py's already-existing CSV outputs, it never drives
TrackingSystem itself.

USAGE
-----
1. Run main.py once against the config you want to survey, with
   debug.vision_pose_csv and debug.led_detections_csv set (and
   data.frame_range narrowed to the segment you want to survey) to produce
   the two CSVs this script reads.
2. python3 build_static_light_map.py --config config/config.yml --cam 3 \
       --vision-pose-csv <path> --led-csv <path> --out <path/to/map.json>
"""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.camera import Camera
from src.headset_pose_source import MocapHeadsetPoseSource
from src.load_config import load_json_config, load_yaml_config
from src.blob_detector import BlobDetector
from src.mocap_data import DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, \
                            DRIFT_CHECK_VARIANT
from src.static_light_geometry import camera_room_pose, pixel_to_room_ray, room_point_to_pixel
from src.static_light_map import StaticLightMap, StaticLightVoxelAccumulator, save_static_light_map
from src.static_light_survey import compute_survey_blob_set
from src.preprocess_data import get_data


def _calib_id(intrinsics_path: str) -> str:
    return hashlib.sha1(Path(intrinsics_path).read_bytes()).hexdigest()[:12]


def _load_headset_mocap(config: dict) -> DeviceMocap:
    """Mirrors main.py's headset-mocap loading (lines ~180-216) -- only the
    "headset" device, which is all this offline tool needs."""
    mocap_cfg = config.get("mocap", {})
    if not mocap_cfg.get("enabled", False):
        raise SystemExit("mocap.enabled is false in this config -- required for "
                          "static-light map building.")
    recording_root = Path(config["data"]["root"]).parent
    cam_cfg = config["cameras"]
    calib_path = cam_cfg.get("mocap_calib_path")
    offset_override_ns = cam_cfg.get("mocap_fine_offset_override_ns")
    device_dir = recording_root / "mocap_filtered" / "headset"
    data_path = device_dir / "data.csv"
    drift_path = device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json"
    if not calib_path or not data_path.exists() or (offset_override_ns is None and not drift_path.exists()):
        raise SystemExit(f"headset mocap data/calibration incomplete under {device_dir}")

    t_mocap, position, quat_xyzw = load_mocap_csv(data_path)
    fine_offset_ns = (float(offset_override_ns) if offset_override_ns is not None
                       else load_mocap_fine_offset_ns(drift_path))
    T_imu_marker = load_T_imu_marker(calib_path)
    max_gap_ns = float(mocap_cfg.get("max_interp_gap_ms", 30.0)) * 1e6
    return DeviceMocap(t_mocap, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                        max_interp_gap_ns=max_gap_ns)


def _load_warm_frames(vision_pose_csv: str, warm_trust_threshold: float, enabled_ctrls) -> set:
    """{frame_ts_ns, ...} where EVERY controller in `enabled_ctrls` has a
    same-frame row (vision_pose_csv writes one row per attempted solve,
    accepted or rejected -- presence of a row for this exact timestamp is
    the "fresh, not carried-over" signal) with confidence >= threshold."""
    per_frame: dict = defaultdict(dict)
    with open(vision_pose_csv) as f:
        for row in csv.DictReader(f):
            ts = int(row["timestamp_ns"])
            per_frame[ts][row["ctrl_name"]] = float(row["confidence"])

    warm = set()
    for ts, confidences in per_frame.items():
        if all(ctrl in confidences and confidences[ctrl] >= warm_trust_threshold
               for ctrl in enabled_ctrls):
            warm.add(ts)
    return warm


def _load_claimed_by_frame_cam(led_csv: str) -> dict:
    """{(frame_ts_ns, camera_id): [(pixel_x, pixel_y), ...]} -- every LED
    pixel actually matched that frame in that camera, across whichever
    controller(s) it belongs to (identity doesn't matter for subtraction)."""
    out: dict = defaultdict(list)
    with open(led_csv) as f:
        for row in csv.DictReader(f):
            key = (int(row["timestamp_ns"]), int(row["camera_id"]))
            out[key].append((float(row["pixel_x"]), float(row["pixel_y"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default="./config/config.yml")
    ap.add_argument("--cam", type=int, required=True, help="camera index to survey")
    ap.add_argument("--vision-pose-csv", type=str, required=True)
    ap.add_argument("--led-csv", type=str, required=True)
    ap.add_argument("--out", type=str, default=None,
                     help="defaults to config's static_light_exclusion.map_path")
    args = ap.parse_args()

    config = load_yaml_config(args.config)
    sl_cfg = config.get("static_light_exclusion", {})
    blob_cfg = config["blob_detection"]
    cam_idx = args.cam
    out_path = args.out or sl_cfg.get("map_path", "./data/static_light_maps/default.json")

    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    camera = Camera(calib_cfg, camera_idx=cam_idx,
                     extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))
    detector = BlobDetector(cam_idx, blob_cfg)

    headset_mocap = _load_headset_mocap(config)
    pose_source = MocapHeadsetPoseSource(headset_mocap)

    enabled_ctrls = [k for k in ("left_controller", "right_controller")
                     if config["controllers"].get(k, {}).get("enabled", False)]
    warm_trust_threshold = float(sl_cfg.get("warm_trust_threshold", 0.7))
    warm_frames = _load_warm_frames(args.vision_pose_csv, warm_trust_threshold, enabled_ctrls)
    claimed_by_frame_cam = _load_claimed_by_frame_cam(args.led_csv)

    accumulator = StaticLightVoxelAccumulator(
        voxel_size_m=float(sl_cfg.get("voxel_size_m", 0.10)),
        min_votes=int(sl_cfg.get("min_votes", 30)),
        min_positional_spread_m=float(sl_cfg.get("min_positional_spread_m", 0.15)),
        min_angular_spread_deg=float(sl_cfg.get("min_angular_spread_deg", 15.0)),
        radius_margin_m=float(sl_cfg.get("radius_margin_m", 0.05)),
        min_confirmed_median_radius_px=float(sl_cfg.get("min_confirmed_median_radius_px", 0.0)),
        min_confirmed_median_brightness=float(sl_cfg.get("min_confirmed_median_brightness", 0.0)),
        cluster_merge_distance_m=(float(sl_cfg["cluster_merge_distance_m"])
                                   if "cluster_merge_distance_m" in sl_cfg else None),
    )
    ray_max_range_m = float(sl_cfg.get("ray_max_range_m", 8.0))
    ray_step_m = float(sl_cfg.get("ray_step_m", 0.05))

    data_cfg = dict(config["data"])
    data_cfg["selected_cameras"] = [cam_idx]

    frame_idx_by_ts = {}
    T_room_cam_by_ts = {}
    n_frames_total = n_frames_warm = n_frames_mocap_gap = n_blobs_fed = 0
    for batch in get_data(data_cfg):
        img_path, cam_images = batch[0][0], batch[0][1]
        frame_ts_ns = int(img_path.stem)
        frame_idx_by_ts[frame_ts_ns] = n_frames_total
        n_frames_total += 1

        if frame_ts_ns not in warm_frames:
            continue
        n_frames_warm += 1

        T_room_headsetImu = pose_source.room_pose_at(frame_ts_ns)
        if T_room_headsetImu is None:
            n_frames_mocap_gap += 1
            continue

        T_room_cam = camera_room_pose(T_room_headsetImu, camera)
        T_room_cam_by_ts[frame_ts_ns] = T_room_cam
        claimed = claimed_by_frame_cam.get((frame_ts_ns, cam_idx), [])
        blobs = compute_survey_blob_set(detector, cam_images[cam_idx], blob_cfg, sl_cfg, claimed)
        if len(blobs) == 0:
            continue

        origin, dirs = pixel_to_room_ray(camera, T_room_cam, blobs.centroids)
        for d, r, b, px in zip(dirs, blobs.radii, blobs.brightnesses, blobs.centroids):
            accumulator.add_observation(origin, d, max_range_m=ray_max_range_m, step_m=ray_step_m,
                                         frame_ts_ns=frame_ts_ns,
                                         blob_radius_px=float(r), blob_brightness=float(b),
                                         blob_px=px)
        n_blobs_fed += len(blobs)

    print(f"[survey] cam{cam_idx}: {n_frames_total} frames read, {n_frames_warm} both-controllers-warm "
          f"(threshold={warm_trust_threshold}), {n_frames_mocap_gap} of those skipped for a mocap gap, "
          f"{n_blobs_fed} unclaimed survey-blob observations fed to the accumulator")

    result, debug_info = accumulator.finalize(
        camera_calib_id=_calib_id(config["cameras"]["intrinsics_path"]),
        created_from_recordings=[str(config["data"]["root"])],
        return_debug=True,
    )
    max_median_resid_px = float(sl_cfg.get("max_confirmed_median_reproj_resid_px", 0.0))
    keep_mask = []
    print(f"[result] {len(result.voxel_room_positions)} confirmed static-light location(s) "
          f"(pre reprojection-residual gate{'' if max_median_resid_px > 0 else ' -- gate disabled'})")
    for pos, radius, votes, dbg in zip(result.voxel_room_positions, result.voxel_radii_m,
                                        result.vote_counts, debug_info):
        idx_min = frame_idx_by_ts.get(dbg["ts_min"]) if dbg["ts_min"] is not None else None
        idx_max = frame_idx_by_ts.get(dbg["ts_max"]) if dbg["ts_max"] is not None else None
        med_r = f"{dbg['median_radius_px']:.2f}px" if dbg["median_radius_px"] is not None else "n/a"
        med_b = f"{dbg['median_brightness']:.1f}" if dbg["median_brightness"] is not None else "n/a"
        max_r = f"{dbg['max_radius_px']:.2f}px" if dbg["max_radius_px"] is not None else "n/a"
        max_b = f"{dbg['max_brightness']:.1f}" if dbg["max_brightness"] is not None else "n/a"
        p90_b = f"{dbg['p90_brightness']:.1f}" if dbg["p90_brightness"] is not None else "n/a"
        # Reprojection-residual check (see StaticLightVoxelAccumulator.add_observation's
        # own docstring): for each vote, reproject THIS cluster's fitted 3D centroid into
        # that vote's own frame and compare against the pixel the blob was actually
        # detected at. A real static light's votes should cluster tightly around 0
        # residual (just centroid/detection noise); a voxel that only accumulated by
        # coincidence should show a much larger, more scattered residual.
        residuals_px = []
        for ts, px in dbg["ts_px_pairs"]:
            T_room_cam_v = T_room_cam_by_ts.get(ts)
            if T_room_cam_v is None:
                continue
            pred_px = room_point_to_pixel(camera, T_room_cam_v, pos.reshape(1, 3))[0]
            residuals_px.append(float(np.linalg.norm(pred_px - px)))
        median_resid = float(np.median(residuals_px)) if residuals_px else None
        if residuals_px:
            resid_str = (f"median_reproj_resid={median_resid:.2f}px  "
                         f"p90_reproj_resid={np.percentile(residuals_px, 90):.2f}px  "
                         f"max_reproj_resid={np.max(residuals_px):.2f}px  "
                         f"n_residuals={len(residuals_px)}")
        else:
            resid_str = "reproj_resid=n/a"
        keep = not (max_median_resid_px > 0 and median_resid is not None
                    and median_resid > max_median_resid_px)
        keep_mask.append(keep)
        tag = "" if keep else "  [DROPPED -- median_reproj_resid exceeds max_confirmed_median_reproj_resid_px]"
        print(f"  room-frame pos={pos.round(3).tolist()}  radius={radius:.3f}m  votes={votes}  "
              f"n_voxels={dbg['n_voxels']}  contributing_frames={idx_min}-{idx_max}  "
              f"n_unique_frames={dbg['n_unique_frames']}  median_blob_radius={med_r}  "
              f"median_blob_brightness={med_b}  max_blob_radius={max_r}  max_blob_brightness={max_b}  "
              f"p90_blob_brightness={p90_b}  min_bbox_diag={dbg['min_bbox_diag_m']:.3f}m  "
              f"min_angle_deg={dbg['min_angle_deg']:.1f}  {resid_str}{tag}")

    keep_mask = np.array(keep_mask, dtype=bool)
    if max_median_resid_px > 0 and not keep_mask.all():
        result = StaticLightMap(
            voxel_size_m=result.voxel_size_m,
            voxel_room_positions=result.voxel_room_positions[keep_mask],
            voxel_radii_m=result.voxel_radii_m[keep_mask],
            vote_counts=result.vote_counts[keep_mask],
            camera_calib_id=result.camera_calib_id,
            created_from_recordings=result.created_from_recordings,
        )
        print(f"[reproj-residual gate] dropped {int((~keep_mask).sum())}, "
              f"{len(result.voxel_room_positions)} location(s) remain")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_static_light_map(result, out_path)
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
