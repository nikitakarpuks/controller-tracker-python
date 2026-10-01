#!/usr/bin/env python3
"""render_all_lamp_overlays.py -- throwaway: same live static_lamp_mask run as
eval_lamp_region_mask.py, but saves a GT-vs-our-mask overlay PNG for EVERY
frame in seq1 (not just a hand-picked sample), for full manual review.
Green = GT polygon, red = our ACTIVE exclusion mask (hits >= min_hits_to_exclude
-- what actually excludes something in production), yellow = a held-but-not-yet-
reinforced region (below min_hits_to_exclude, shown for context only, dashed).
Image brightened 3x for visibility (this recording is very dark). Text overlay:
image_id, containment, region hits/no_support_streak. NOT committed.

Usage: python3 render_all_lamp_overlays.py
"""
import copy
import json
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
from src.static_light_geometry import camera_room_pose

SEQ_DIR = Path("/home/nikitakarpuks/Documents/lamp_sequences/seq1")
GT_PATH = SEQ_DIR / "instances_default.json"
OUT_DIR = SEQ_DIR / "review" / "all"
CAM_IDX = 3


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


def _rasterize_polys(anns, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for ann in anns:
        seg = ann["segmentation"]
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def _rasterize_contours(contours, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for cnt in contours or []:
        if cnt is None:
            continue
        pts = np.asarray(cnt, dtype=np.float64).reshape(-1, 2)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.load(open(GT_PATH))
    images = {im["id"]: im for im in d["images"]}
    anns_by_image = {}
    for ann in d["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    ordered_image_ids = sorted(images.keys())
    ts_by_image = {i: int(Path(im["file_name"]).stem) for i, im in images.items()}

    config = load_yaml_config("./config/config.yml")
    blob_cfg = copy.deepcopy(config["blob_detection"])
    blob_cfg["lamp_blob_filter"]["static_lamp_mask"]["enabled"] = True

    folder = _camera_dir(config["data"], CAM_IDX)
    all_paths = sorted(folder.glob("*.png"))
    top = 1 if config["data"].get("has_technical_row", True) else 0
    path_by_ts = {int(p.stem): p for p in all_paths}

    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    camera = Camera(calib_cfg, camera_idx=CAM_IDX,
                     extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))
    headset_mocap = _load_headset_mocap(config)
    pose_source = MocapHeadsetPoseSource(headset_mocap)

    detector = BlobDetector(CAM_IDX, blob_cfg)

    for n, im_id in enumerate(ordered_image_ids):
        ts = ts_by_image[im_id]
        p = path_by_ts.get(ts)
        if p is None:
            continue
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        image = image[top:] if top else image
        detector.detect(image, predicted_leds=None, camera=camera, pose_source=pose_source,
                         frame_ts_ns=ts, visualize=False)

        T_room_headsetImu = pose_source.room_pose_at(ts)
        active_contours, all_contours, regions = [], [], []
        if T_room_headsetImu is not None and detector._lamp_region_memory is not None:
            T_room_cam = camera_room_pose(T_room_headsetImu, camera)
            mem = detector._lamp_region_memory
            active_contours = mem.reprojected_contours(camera, T_room_cam)
            all_contours = mem.all_reprojected_contours(camera, T_room_cam)
            regions = mem._regions

        shape = (images[im_id]["height"], images[im_id]["width"])
        gt_anns = anns_by_image.get(im_id, [])
        gt_mask = _rasterize_polys(gt_anns, shape)
        our_mask = _rasterize_contours(active_contours, shape)
        gt_area = int(gt_mask.sum())
        inter = int((gt_mask & our_mask).sum())
        containment = (inter / gt_area) if gt_area > 0 else None

        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        canvas = cv2.convertScaleAbs(canvas, alpha=3.0, beta=0)

        # yellow: held but not yet reinforced enough to actually exclude
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

        for a in gt_anns:
            pts = np.array(a["segmentation"][0], dtype=np.float64).reshape(-1, 2).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)

        hits_list = [r.hits for r in regions]
        streak_list = [r.no_support_streak for r in regions]
        cont_str = f"{containment:.2f}" if containment is not None else "n/a"
        label1 = f"id={im_id} ts={ts}"
        label2 = f"containment={cont_str} gt_inst={len(gt_anns)} regions={len(regions)}"
        label3 = f"hits={hits_list} no_support={streak_list}"
        for i_line, text in enumerate((label1, label2, label3)):
            cv2.putText(canvas, text, (6, 16 + 14 * i_line), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (6, 16 + 14 * i_line), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = OUT_DIR / f"{im_id:04d}_{ts}.png"
        cv2.imwrite(str(out_path), canvas)
        if n % 50 == 0:
            print(f"  {n}/{len(ordered_image_ids)}: wrote {out_path.name}")

    print(f"done -- wrote {len(ordered_image_ids)} overlays to {OUT_DIR}")


if __name__ == "__main__":
    main()
