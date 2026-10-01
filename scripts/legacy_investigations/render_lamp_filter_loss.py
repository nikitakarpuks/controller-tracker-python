#!/usr/bin/env python3
"""render_lamp_filter_loss.py -- generic tool: point it at ANY folder of raw
PNG frames and it visualizes, per frame, exactly which candidate blobs the
lamp filter removes -- i.e. "how much real signal are we losing" -- with NO
ground truth and NO per-recording setup required.

Method: runs BlobDetector.detect() TWICE per frame, cold (predicted_leds=
None) both times:
  (A) lamp_blob_filter DISABLED entirely -- the raw, unfiltered baseline: every
      real candidate blob that would otherwise reach the tracking pool.
  (B) lamp_blob_filter as configured in config.yml (production behavior).
A candidate present in (A) with no close match in (B) (nearest-neighbor,
REMOVED_MATCH_PX default 3px) was REMOVED by the lamp filter -- flagged in
orange for manual review. Matched-in-both candidates are drawn in green.

Scope note: this compares the SAME-FRAME lamp_blob_filter mechanism
(detect_lamp_blobs' own row recognition) plus static_lamp_mask's persistent
region exclusion IF you also pass --camera-idx/--intrinsics/--mocap-root
(needs a real recording's mocap + camera calibration, so it's off by default
and this generic per-folder tool works with zero setup otherwise). Without
those flags, static_lamp_mask never activates (BlobDetector.detect()'s own
fail-open behavior -- no camera/pose_source/frame_ts_ns supplied), so what
you see is detect_lamp_blobs' own recognition-based removal only.

When static_lamp_mask IS active, the region contours are also drawn (same
convention as the labeled-sequence overlays): RED = an active region (hits
>= min_hits_to_exclude, i.e. actually excluding candidates this frame),
YELLOW (thin) = a held region not yet reinforced enough to exclude anything.

Usage:
  python3 render_lamp_filter_loss.py <input_dir> <output_dir>
  python3 render_lamp_filter_loss.py <input_dir> <output_dir> \\
      --camera-idx 3 --intrinsics ./data/cameras/calibration_basalt.json \\
      --mocap-root /path/to/recording  (sibling to that recording's mav0/)

Images saved as JPEG (quality 60) to keep total output size down for a full
folder, per request -- not committed.
"""
import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blob_detector import BlobDetector
from src.load_config import load_yaml_config, load_json_config

REMOVED_MATCH_PX = 3.0
JPEG_QUALITY = 60


def _match_removed(off_centroids, on_centroids, tol_px=REMOVED_MATCH_PX):
    """Centroids in off_centroids with no on_centroids match within tol_px."""
    if len(off_centroids) == 0:
        return np.empty((0, 2))
    if len(on_centroids) == 0:
        return np.asarray(off_centroids).reshape(-1, 2)
    off = np.asarray(off_centroids, dtype=np.float64).reshape(-1, 2)
    on = np.asarray(on_centroids, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(off[:, None, :] - on[None, :, :], axis=2)
    unmatched = d.min(axis=1) > tol_px
    return off[unmatched]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--config", default="./config/config.yml")
    ap.add_argument("--camera-idx", type=int, default=None,
                     help="Enables static_lamp_mask too (needs --intrinsics and --mocap-root).")
    ap.add_argument("--intrinsics", default=None)
    ap.add_argument("--mocap-root", type=Path, default=None,
                     help="Recording root -- parent of that recording's mav0/, "
                          "sibling to its mocap_filtered/ -- for static_lamp_mask's mocap lookup.")
    ap.add_argument("--limit", type=int, default=None, help="Process only N frames (smoke-testing).")
    ap.add_argument("--start-index", type=int, default=0, help="Skip the first N frames before --limit applies.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*.png"))
    if args.start_index:
        paths = paths[args.start_index:]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no PNG files found in {args.input_dir}")
        sys.exit(1)
    print(f"{len(paths)} frames found in {args.input_dir}")

    config = load_yaml_config(args.config)
    on_cfg = copy.deepcopy(config["blob_detection"])
    off_cfg = copy.deepcopy(config["blob_detection"])
    off_cfg["lamp_blob_filter"] = dict(off_cfg["lamp_blob_filter"])
    off_cfg["lamp_blob_filter"]["enabled"] = False

    camera = pose_source = None
    if args.camera_idx is not None:
        if not (args.intrinsics and args.mocap_root):
            print("--camera-idx given without --intrinsics/--mocap-root -- static_lamp_mask "
                  "needs all three, falling back to detect_lamp_blobs-only comparison.")
        else:
            from src.camera import Camera
            from src.headset_pose_source import MocapHeadsetPoseSource
            from src.mocap_data import DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker
            calib_cfg = load_json_config(args.intrinsics)
            camera = Camera(calib_cfg, camera_idx=args.camera_idx,
                             extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))
            device_dir = args.mocap_root / "mocap_filtered" / "headset"
            t_mocap, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
            calib_path = config["cameras"]["mocap_calib_path"]
            offset_override_ns = config["cameras"].get("mocap_fine_offset_override_ns")
            if offset_override_ns is not None:
                fine_offset_ns = float(offset_override_ns)
            else:
                from main import DRIFT_CHECK_VARIANT
                fine_offset_ns = load_mocap_fine_offset_ns(
                    device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
            T_imu_marker = load_T_imu_marker(calib_path)
            max_gap_ns = float(config.get("mocap", {}).get("max_interp_gap_ms", 30.0)) * 1e6
            headset_mocap = DeviceMocap(t_mocap, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                                         max_interp_gap_ns=max_gap_ns)
            pose_source = MocapHeadsetPoseSource(headset_mocap)
            print(f"static_lamp_mask ENABLED (cam{args.camera_idx}, mocap from {args.mocap_root})")

    cam_idx = args.camera_idx if args.camera_idx is not None else 0
    detector_on = BlobDetector(cam_idx, on_cfg)
    detector_off = BlobDetector(cam_idx, off_cfg)

    total_off, total_removed = 0, 0
    per_frame_removed = []
    for i, p in enumerate(paths):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        ts = int(p.stem) if p.stem.isdigit() else None
        kwargs = {}
        if camera is not None and ts is not None:
            kwargs = dict(camera=camera, pose_source=pose_source, frame_ts_ns=ts)

        result_off, _ = detector_off.detect(image, predicted_leds=None, visualize=False)
        result_on, _ = detector_on.detect(image, predicted_leds=None, visualize=False, **kwargs)

        removed = _match_removed(result_off.centroids, result_on.centroids)
        total_off += len(result_off.centroids)
        total_removed += len(removed)
        per_frame_removed.append(len(removed))

        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        canvas = cv2.convertScaleAbs(canvas, alpha=2.5, beta=0)  # brighten for visibility

        n_active_regions = n_held_regions = 0
        if camera is not None and ts is not None and detector_on._lamp_region_memory is not None:
            T_room_headsetImu = pose_source.room_pose_at(ts)
            if T_room_headsetImu is not None:
                from src.static_light_geometry import camera_room_pose
                T_room_cam = camera_room_pose(T_room_headsetImu, camera)
                mem = detector_on._lamp_region_memory
                active_contours = mem.reprojected_contours(camera, T_room_cam)
                all_contours = mem.all_reprojected_contours(camera, T_room_cam)
                n_active_regions = len(active_contours)
                n_held_regions = len(all_contours) - n_active_regions
                for cnt in all_contours:
                    if cnt is None:
                        continue
                    is_active = any(np.array_equal(cnt, ac) for ac in active_contours if ac is not None)
                    if is_active:
                        continue
                    pts = np.asarray(cnt, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(canvas, [pts], True, (0, 200, 255), 1, lineType=cv2.LINE_AA)
                for cnt in active_contours:
                    if cnt is None:
                        continue
                    pts = np.asarray(cnt, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)

        for cx, cy in result_on.centroids:
            cv2.drawMarker(canvas, (int(round(cx)), int(round(cy))), (0, 220, 0),
                            markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        for cx, cy in removed:
            cv2.drawMarker(canvas, (int(round(cx)), int(round(cy))), (0, 140, 255),
                            markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
        label = f"{p.stem}  kept={len(result_on.centroids)}  removed_by_filter={len(removed)}"
        if camera is not None:
            label += f"  regions={n_active_regions}active/{n_held_regions}held"
        cv2.putText(canvas, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = args.output_dir / f"{p.stem}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if i % 50 == 0:
            print(f"  {i}/{len(paths)}: {out_path.name}  removed={len(removed)}")

    print(f"\ndone -- wrote {len(paths)} images to {args.output_dir}")
    print(f"total raw candidates (filter off): {total_off}")
    print(f"total removed by lamp filter:      {total_removed} "
          f"({total_removed/total_off*100 if total_off else 0:.1f}%)")
    worst = sorted(range(len(paths)), key=lambda i: -per_frame_removed[i])[:10]
    print("worst 10 frames by removed count (check these first):")
    for i in worst:
        if per_frame_removed[i] == 0:
            break
        print(f"  {paths[i].stem}: removed={per_frame_removed[i]}")


if __name__ == "__main__":
    main()
