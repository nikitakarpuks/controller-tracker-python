#!/usr/bin/env python3
"""find_lamp_structures.py -- targeted single-frame structural lamp-fixture
detector (see src/lamp_structure_detector.py) + a lightweight mocap-based
staticness check, as an alternative to the multi-frame 3D voxel-vote
approach in build_static_light_map.py. Finds candidate fixtures from ONE
frame's blob geometry (parallel lines of small dim blobs), then confirms
each candidate is actually static in the room by reprojecting it into a
handful of OTHER frames and checking a blob still shows up near the
predicted pixel there.

USAGE
-----
python3 find_lamp_structures.py --config config/config_static_light_probe.yml \
    --cam 3 --frame-idx 3 --out-dir visualization/lamp_structures
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.headset_pose_source import MocapHeadsetPoseSource
from src.lamp_structure_detector import find_fixture_candidates, find_lines
from src.load_config import load_json_config, load_yaml_config
from src.preprocess_data import get_data
from src.static_light_geometry import camera_room_pose, pixel_to_room_ray, room_point_to_pixel
from src.static_light_survey import compute_survey_blob_set

from build_static_light_map import _load_headset_mocap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default="./config/config.yml")
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--frame-idx", type=int, default=0, help="0-based frame index to run detection on")
    ap.add_argument("--n-check-frames", type=int, default=10,
                     help="how many other frames (spread across the recording) to reproject into for staticness")
    ap.add_argument("--bearing-tol-deg", type=float, default=1.5,
                     help="a check-frame blob whose room-frame ray bearing is within this of the "
                          "candidate's own bearing counts as confirming staticness")
    ap.add_argument("--out-dir", type=str, default="./visualization/lamp_structures")
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
    detector = BlobDetector(cam_idx, blob_cfg)
    pose_source = MocapHeadsetPoseSource(_load_headset_mocap(config))

    data_cfg = dict(config["data"])
    data_cfg["selected_cameras"] = [cam_idx]

    frames = []  # (frame_idx, frame_ts_ns, image)
    for i, batch in enumerate(get_data(data_cfg)):
        img_path, cam_images = batch[0][0], batch[0][1]
        frames.append((i, int(img_path.stem), cam_images[cam_idx]))
    print(f"[data] {len(frames)} frames loaded for cam{cam_idx}")

    target_idx, target_ts, target_img = frames[args.frame_idx]

    blobs = compute_survey_blob_set(detector, target_img, blob_cfg, sl_cfg, [])
    print(f"[frame {target_idx}] {len(blobs)} unclaimed survey blobs")

    lines = find_lines(blobs)
    print(f"[structure] {len(lines)} line candidate(s) found")
    for ln in lines:
        print(f"  line: {len(ln.blob_indices)} blobs, direction={ln.direction.round(3).tolist()}, "
              f"span_px={np.linalg.norm(ln.points[-1] - ln.points[0]):.1f}")

    fixtures = find_fixture_candidates(blobs, lines)
    print(f"[structure] {len(fixtures)} fixture candidate(s)")

    T_room_headsetImu = pose_source.room_pose_at(target_ts)
    if T_room_headsetImu is None:
        print("[error] no mocap at this frame's timestamp, cannot verify staticness")
        return
    T_room_cam = camera_room_pose(T_room_headsetImu, camera)

    # Pick n_check_frames spread across the whole recording for the staticness check.
    check_frame_idxs = np.linspace(0, len(frames) - 1, args.n_check_frames).astype(int).tolist()
    check_frame_idxs = sorted(set(check_frame_idxs) - {target_idx})

    canvas = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    canvas = cv2.convertScaleAbs(canvas, alpha=14.0)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    for fi, fixture in enumerate(fixtures):
        color = colors[fi % len(colors)]
        for ln in fixture.lines:
            pts = ln.points.astype(int)
            for p in pts:
                cv2.circle(canvas, tuple(p), 3, color, 1)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(canvas, tuple(a), tuple(b), color, 1)
        cx, cy = fixture.centroid_px
        cv2.drawMarker(canvas, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)

        # No depth from a single ray, so staticness is checked via BEARING (room-frame
        # ray direction), not pixel/3D-point distance: a static room feature's bearing
        # from wherever the headset currently is should be reproducible -- re-detect
        # survey blobs in each check frame and see if any of them casts a ray within
        # a small angular tolerance of this candidate's own bearing in the target frame.
        origin, dirs = pixel_to_room_ray(camera, T_room_cam, fixture.centroid_px.reshape(1, 2))

        n_hits = 0
        n_checked = 0
        for cfi in check_frame_idxs:
            _, cts, cimg = frames[cfi]
            T_room_headsetImu_c = pose_source.room_pose_at(cts)
            if T_room_headsetImu_c is None:
                continue
            T_room_cam_c = camera_room_pose(T_room_headsetImu_c, camera)
            check_blobs = compute_survey_blob_set(detector, cimg, blob_cfg, sl_cfg, [])
            if len(check_blobs) == 0:
                n_checked += 1
                continue
            origin_c, dirs_c = pixel_to_room_ray(camera, T_room_cam_c, check_blobs.centroids)
            bearing_cos = dirs_c @ dirs[0]
            best_angle_deg = float(np.degrees(np.arccos(np.clip(bearing_cos.max(), -1.0, 1.0))))
            n_checked += 1
            if best_angle_deg <= args.bearing_tol_deg:
                n_hits += 1

        print(f"  fixture {fi}: centroid_px=({cx:.1f},{cy:.1f})  n_lines={len(fixture.lines)}  "
              f"n_blobs={len(fixture.blob_indices)}  staticness={n_hits}/{n_checked} frames confirmed")
        cv2.putText(canvas, f"{n_hits}/{n_checked}", (int(cx) + 10, int(cy) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    out_path = out_dir / f"cam{cam_idx}_frame{target_idx}_structures.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
