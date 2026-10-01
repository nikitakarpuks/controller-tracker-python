#!/usr/bin/env python3
"""verify_lamp_region_mask_e2e.py -- throwaway end-to-end check for
blob_detection.lamp_blob_filter.static_lamp_mask (see
src/lamp_region_memory.py, replacing the earlier point-based
src/lamp_anchor_memory.py). NOT committed as a permanent feature.

Drives BlobDetector.detect() directly (same call shape as main.py's
sequential branch) frame-by-frame in order (the region memory is stateful
per BlobDetector instance, same as main.py's real per-camera detector), with
no predicted_leds (has_prior=False, matching this recording's own real
cold-path bug -- no controller model needed). Reports, per frame: count of
real (post-lamp-filter) candidate centroids that fall inside the reported
lamp bbox [575,330,635,370] -- the leak signature -- and the region memory's
own held-region count.

Usage: python3 verify_lamp_region_mask_e2e.py --mask {true,false}
"""
import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.headset_pose_source import MocapHeadsetPoseSource
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker
from src.preprocess_data import _camera_dir

_LAMP_BBOX = (575, 330, 635, 370)  # x0, y0, x1, y1 -- the reported real lamp bbox


def _load_headset_mocap(config: dict) -> DeviceMocap:
    recording_root = Path(config["data"]["root"]).parent
    device_dir = recording_root / "mocap_filtered" / "headset"
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
    return DeviceMocap(t_mocap, position, quat_xyzw, fine_offset_ns, T_imu_marker, max_interp_gap_ns=max_gap_ns)


def _in_bbox(pt, bbox):
    x0, y0, x1, y1 = bbox
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=3)
    ap.add_argument("--mask", choices=["true", "false"], default="true")
    args = ap.parse_args()

    config = load_yaml_config("./config/config.yml")
    cam_idx = args.cam

    blob_cfg = copy.deepcopy(config["blob_detection"])
    blob_cfg["lamp_blob_filter"]["static_lamp_mask"]["enabled"] = (args.mask == "true")

    folder = _camera_dir(config["data"], cam_idx)
    paths = sorted(folder.glob("*.png"))
    fr = config["data"].get("frame_range") or {}
    paths = paths[fr.get("lower"):fr.get("upper")]
    top = 1 if config["data"].get("has_technical_row", True) else 0

    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    camera = Camera(calib_cfg, camera_idx=cam_idx,
                     extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))
    headset_mocap = _load_headset_mocap(config)
    pose_source = MocapHeadsetPoseSource(headset_mocap)

    detector = BlobDetector(cam_idx, blob_cfg)

    print(f"mask.enabled={blob_cfg['lamp_blob_filter']['static_lamp_mask']['enabled']} "
          f"ceiling_height_m={blob_cfg['lamp_blob_filter']['static_lamp_mask']['ceiling_height_m']}")
    print(f"{'frame':>5}  {'n_kept':>6}  {'n_in_lamp_bbox':>14}  {'n_regions':>9}  lamp_bbox_centroids")

    leak_frames = []
    total_leak = 0
    for i, p in enumerate(paths):
        ts = int(p.stem)
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[top:]
        result, _canvases = detector.detect(
            image, camera=camera, pose_source=pose_source, frame_ts_ns=ts)
        in_bbox = [tuple(np.round(c, 1)) for c in result.centroids if _in_bbox(c, _LAMP_BBOX)]
        n_regions = len(detector._lamp_region_memory._regions) if detector._lamp_region_memory else 0
        print(f"{i:>5}  {len(result.centroids):>6}  {len(in_bbox):>14}  {n_regions:>9}  {in_bbox}")
        total_leak += len(in_bbox)
        if in_bbox:
            leak_frames.append(i)

    print(f"\nTotal leaked centroid-detections: {total_leak}")
    print(f"Frames with a real candidate leaking inside the lamp bbox: {leak_frames}")


if __name__ == "__main__":
    main()
