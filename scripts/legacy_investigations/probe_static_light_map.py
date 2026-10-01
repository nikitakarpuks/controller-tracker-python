#!/usr/bin/env python3
"""probe_static_light_map.py -- visual sanity check for a StaticLightMap
built by build_static_light_map.py (Phase 4 of the mocap-based static-light/
ceiling-lamp exclusion feature). Same philosophy as the existing
probe_static_blobs.py: replay real frames and LOOK at the result before
trusting any number or wiring it into production matching.

Replays one camera's real frames, and for each one draws:
  - RED cross + vote count: every confirmed static-light entry in the map,
    projected into this frame via the headset's mocap pose. This is the
    thing you're actually here to check -- does it sit on the real lamp?
  - ORANGE circle (optional, --vision-pose-csv/--led-csv): the permissive
    survey pass's unclaimed blobs for this frame (same computation
    build_static_light_map.py fed into the accumulator).
  - YELLOW tilted-cross (optional, same flags): this frame's claimed
    controller-LED pixels, subtracted out of the survey set.

Output is a folder of per-frame PNGs (--out-dir/frames/) plus a legend.png
explaining the color code -- not a video, so you can open, zoom, and step
through frames individually. Frames with no mocap coverage at that timestamp
are still written (so the sequence stays complete) but get no red-cross
overlay and a "NO MOCAP" banner instead.

Raw frames from this pipeline are very dark near-IR captures (observed
max pixel value ~15/255 on at least one real recording) -- brightened by
--brighten before drawing, purely for human visibility; detection itself
always runs on the raw, unbrightened pixel values.

USAGE
-----
python3 probe_static_light_map.py --config config/config_static_light_probe.yml \
    --cam 3 --map data/static_light_maps/walk_medium_cam3_f1-100_probe.json \
    --vision-pose-csv data/static_light_probe/vision_pose_log_walk_medium_f1-100.csv \
    --led-csv data/static_light_probe/led_detections_walk_medium_f1-100.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.headset_pose_source import MocapHeadsetPoseSource
from src.load_config import load_json_config, load_yaml_config
from src.preprocess_data import get_data
from src.static_light_geometry import camera_room_pose, room_point_to_pixel
from src.static_light_map import load_static_light_map
from src.static_light_survey import compute_survey_blob_set

from build_static_light_map import _load_headset_mocap

_COLOR_LIGHT   = (0, 0, 255)     # BGR red   -- confirmed static-light projection
_COLOR_SURVEY  = (0, 140, 255)   # BGR orange -- unclaimed survey blob
_COLOR_CLAIMED = (0, 255, 255)   # BGR yellow -- this frame's claimed controller LED

_LEGEND_ENTRIES = [
    (_COLOR_LIGHT,   "cross", "Confirmed static-light projection (label = accumulated votes)"),
    (_COLOR_SURVEY,  "circle", "Unclaimed survey blob this frame (fed into the map builder)"),
    (_COLOR_CLAIMED, "tilted_cross", "Claimed controller-LED pixel this frame (subtracted from survey set)"),
    ((255, 255, 255), "text", "\"NO MOCAP THIS FRAME\" banner -- no headset pose at this timestamp,"
                              " light overlay skipped for that frame only"),
]


def _write_legend(out_dir: Path, width: int = 640) -> Path:
    row_h = 34
    canvas = np.full((row_h * len(_LEGEND_ENTRIES) + 16, width, 3), 30, dtype=np.uint8)
    for i, (color, marker, label) in enumerate(_LEGEND_ENTRIES):
        cy = 16 + i * row_h + row_h // 2
        cx = 30
        if marker == "cross":
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        elif marker == "circle":
            cv2.circle(canvas, (cx, cy), 6, color, 1)
        elif marker == "tilted_cross":
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 12, 1)
        else:
            cv2.putText(canvas, "\"...\"", (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (60, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    path = out_dir / "legend.png"
    cv2.imwrite(str(path), canvas)
    return path


def _load_optional_csvs(vision_pose_csv, led_csv, warm_trust_threshold, enabled_ctrls):
    if not vision_pose_csv or not led_csv:
        return None, None
    per_frame = defaultdict(dict)
    with open(vision_pose_csv) as f:
        for row in csv.DictReader(f):
            per_frame[int(row["timestamp_ns"])][row["ctrl_name"]] = float(row["confidence"])
    warm = {ts for ts, c in per_frame.items()
            if all(c.get(k, 0) >= warm_trust_threshold for k in enabled_ctrls)}

    claimed = defaultdict(list)
    with open(led_csv) as f:
        for row in csv.DictReader(f):
            claimed[(int(row["timestamp_ns"]), int(row["camera_id"]))].append(
                (float(row["pixel_x"]), float(row["pixel_y"])))
    return warm, claimed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default="./config/config.yml")
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--map", type=str, required=True, help="StaticLightMap JSON from build_static_light_map.py")
    ap.add_argument("--vision-pose-csv", type=str, default=None,
                     help="optional: also overlay unclaimed survey blobs (needs --led-csv too)")
    ap.add_argument("--led-csv", type=str, default=None)
    ap.add_argument("--brighten", type=float, default=8.0,
                     help="display-only linear gain applied before drawing (detection runs on raw pixels)")
    ap.add_argument("--out-dir", type=str, default="./visualization/static_light_probe")
    args = ap.parse_args()

    config = load_yaml_config(args.config)
    sl_cfg = config.get("static_light_exclusion", {})
    blob_cfg = config["blob_detection"]
    cam_idx = args.cam
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    camera = Camera(calib_cfg, camera_idx=cam_idx,
                     extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))
    pose_source = MocapHeadsetPoseSource(_load_headset_mocap(config))
    smap = load_static_light_map(args.map)
    print(f"[map] {len(smap.voxel_room_positions)} confirmed static-light location(s) loaded from {args.map}")

    detector = None
    warm_frames = claimed_by_frame_cam = None
    if args.vision_pose_csv and args.led_csv:
        enabled_ctrls = [k for k in ("left_controller", "right_controller")
                         if config["controllers"].get(k, {}).get("enabled", False)]
        warm_trust_threshold = float(sl_cfg.get("warm_trust_threshold", 0.7))
        warm_frames, claimed_by_frame_cam = _load_optional_csvs(
            args.vision_pose_csv, args.led_csv, warm_trust_threshold, enabled_ctrls)
        detector = BlobDetector(cam_idx, blob_cfg)
        print(f"[survey overlay] enabled -- {len(warm_frames)} both-controllers-warm frames")
    else:
        print("[survey overlay] disabled (pass --vision-pose-csv and --led-csv to enable)")

    data_cfg = dict(config["data"])
    data_cfg["selected_cameras"] = [cam_idx]

    frames_dir = out_dir / f"cam{cam_idx}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    legend_path = _write_legend(out_dir)
    csv_path = out_dir / f"cam{cam_idx}_static_light_overlay.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_ts_ns", "light_idx", "pixel_x", "pixel_y", "votes", "radius_m", "has_mocap"])

    n_frames = n_with_mocap = 0
    for batch in get_data(data_cfg):
        img_path, cam_images = batch[0][0], batch[0][1]
        frame_ts_ns = int(img_path.stem)
        image = cam_images[cam_idx]
        n_frames += 1

        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        canvas = cv2.convertScaleAbs(canvas, alpha=args.brighten)

        T_room_headsetImu = pose_source.room_pose_at(frame_ts_ns)
        has_mocap = T_room_headsetImu is not None
        if has_mocap:
            n_with_mocap += 1
            T_room_cam = camera_room_pose(T_room_headsetImu, camera)
            px_lights = (room_point_to_pixel(camera, T_room_cam, smap.voxel_room_positions)
                         if len(smap.voxel_room_positions) else np.zeros((0, 2)))
            for i, ((px, py), votes, radius_m) in enumerate(
                    zip(px_lights, smap.vote_counts, smap.voxel_radii_m)):
                cv2.drawMarker(canvas, (int(px), int(py)), _COLOR_LIGHT, cv2.MARKER_CROSS, 16, 2)
                cv2.putText(canvas, str(int(votes)), (int(px) + 8, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_LIGHT, 1, cv2.LINE_AA)
                csv_writer.writerow([frame_ts_ns, i, f"{px:.1f}", f"{py:.1f}", int(votes), f"{radius_m:.3f}", True])
        else:
            cv2.putText(canvas, "NO MOCAP THIS FRAME", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for i in range(len(smap.voxel_room_positions)):
                csv_writer.writerow([frame_ts_ns, i, "", "", int(smap.vote_counts[i]),
                                      f"{smap.voxel_radii_m[i]:.3f}", False])

        if detector is not None and frame_ts_ns in warm_frames:
            claimed = claimed_by_frame_cam.get((frame_ts_ns, cam_idx), [])
            for (px, py) in claimed:
                cv2.drawMarker(canvas, (int(px), int(py)), _COLOR_CLAIMED,
                                cv2.MARKER_TILTED_CROSS, 10, 1)
            blobs = compute_survey_blob_set(detector, image, blob_cfg, sl_cfg, claimed)
            for c in blobs.centroids:
                cv2.circle(canvas, (int(c[0]), int(c[1])), 4, _COLOR_SURVEY, 1)

        cv2.putText(canvas, f"ts={frame_ts_ns}", (8, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frame_path = frames_dir / f"frame_{n_frames - 1:04d}_ts{frame_ts_ns}.png"
        cv2.imwrite(str(frame_path), canvas)

    csv_file.close()
    print(f"[out] {frames_dir}/  ({n_frames} PNGs)")
    print(f"[out] {legend_path}")
    print(f"[out] {csv_path}")
    print(f"[summary] {n_frames} frames written, {n_with_mocap} had mocap coverage "
          f"({n_frames - n_with_mocap} skipped the light overlay for a mocap gap)")


if __name__ == "__main__":
    main()
