#!/usr/bin/env python3
"""eval_lamp_region_mask.py -- throwaway eval harness: run the REAL cold-path
BlobDetector.detect() (static_lamp_mask enabled, mocap-driven) over seq1's 350
cam3 frames and score the resulting live exclusion mask against the
human-labeled ground truth (CVAT COCO export) for that same sequence. NOT
committed.

Metrics are deliberately forgiving, per the user's own framing: some GT
polygons are weak/uncertain, some noisy image regions could plausibly look
lamp-like even where GT drew nothing -- so this reports *distributions* and
flags concrete frames for joint review rather than a single pass/fail number.

Usage: python3 eval_lamp_region_mask.py
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

SEQ_DIR = Path("/home/nikitakarpuks/Documents/lamp_sequences/seq1")
GT_PATH = SEQ_DIR / "instances_default.json"
CAM_IDX = 3  # user-confirmed: disk cam7 == internal cam3


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


def _load_gt(gt_path: Path):
    d = json.load(open(gt_path))
    images = {im["id"]: im for im in d["images"]}
    anns_by_image = {}
    for ann in d["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    # file_name is "out/<ts>.png" -- ts is the raw source frame_ts_ns (confirmed:
    # identical filenames exist in the recording's mav0/cam7/data/).
    ts_by_image = {}
    for im_id, im in images.items():
        ts = int(Path(im["file_name"]).stem)
        ts_by_image[im_id] = ts
    return images, anns_by_image, ts_by_image


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
    images, anns_by_image, ts_by_image = _load_gt(GT_PATH)
    print(f"loaded GT: {len(images)} images, {sum(len(v) for v in anns_by_image.values())} annotations")

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

    # Order by image id (== CVAT's own frame ordering == chronological, matches
    # the source recording's own sorted order) so region hits accumulate
    # exactly as they would in a real cold-tracking run over this stretch.
    ordered_image_ids = sorted(images.keys())

    per_frame = []
    missing_source = 0
    for im_id in ordered_image_ids:
        ts = ts_by_image[im_id]
        p = path_by_ts.get(ts)
        if p is None:
            missing_source += 1
            continue
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if image is None:
            missing_source += 1
            continue
        image = image[top:] if top else image
        result, _ = detector.detect(
            image, predicted_leds=None, camera=camera, pose_source=pose_source,
            frame_ts_ns=ts, visualize=False)

        T_room_headsetImu = pose_source.room_pose_at(ts)
        our_contours = []
        if T_room_headsetImu is not None and detector._lamp_region_memory is not None:
            from src.static_light_geometry import camera_room_pose
            T_room_cam = camera_room_pose(T_room_headsetImu, camera)
            our_contours = detector._lamp_region_memory.reprojected_contours(camera, T_room_cam)

        shape = (images[im_id]["height"], images[im_id]["width"])
        gt_anns = anns_by_image.get(im_id, [])
        gt_mask = _rasterize_polys(gt_anns, shape)
        our_mask = _rasterize_contours(our_contours, shape)

        gt_area = int(gt_mask.sum())
        our_area = int(our_mask.sum())
        inter = int((gt_mask & our_mask).sum())

        per_frame.append({
            "image_id": im_id,
            "file_name": images[im_id]["file_name"],
            "ts": ts,
            "gt_area": gt_area,
            "our_area": our_area,
            "inter": inter,
            "n_gt_instances": len(gt_anns),
            "n_regions": len([c for c in our_contours if c is not None]),
            "containment": (inter / gt_area) if gt_area > 0 else None,
        })

    if missing_source:
        print(f"WARNING: {missing_source} GT images had no matching source frame on disk")

    _report(per_frame)
    out_path = SEQ_DIR / "eval_lamp_region_mask_per_frame.json"
    with open(out_path, "w") as f:
        json.dump(per_frame, f, indent=2)
    print(f"\nwrote per-frame detail to {out_path}")


def _report(per_frame):
    with_gt = [f for f in per_frame if f["gt_area"] > 0]
    without_gt = [f for f in per_frame if f["gt_area"] == 0]

    print(f"\n{len(per_frame)} frames scored ({len(with_gt)} with GT lamps, {len(without_gt)} GT-empty)")

    containments = [f["containment"] for f in with_gt]
    if containments:
        arr = np.array(containments)
        print(f"\ncontainment (fraction of GT area covered by our mask), n={len(arr)}:")
        print(f"  mean={arr.mean():.3f} median={np.median(arr):.3f} "
              f"p10={np.percentile(arr,10):.3f} min={arr.min():.3f}")
        for thresh in (0.8, 0.5, 0.2, 0.0):
            frac = (arr <= thresh).mean()
            print(f"  frames with containment <= {thresh}: {(arr<=thresh).sum()}/{len(arr)} ({frac*100:.1f}%)")

    # tightness only over frames where we actually overlapped at all (informational,
    # not primary -- oversize is safety-fine, per the project's own philosophy)
    tight = [f["our_area"] / f["gt_area"] for f in with_gt if f["inter"] > 0 and f["gt_area"] > 0]
    if tight:
        arr = np.array(tight)
        print(f"\ntightness (our_area/gt_area) where mask overlapped at all, n={len(arr)}:")
        print(f"  mean={arr.mean():.2f} median={np.median(arr):.2f}")

    fp_frames = [f for f in without_gt if f["our_area"] > 0]
    print(f"\nGT-empty frames where we still show a region: {len(fp_frames)}/{len(without_gt)}"
          f" (not necessarily wrong -- may be real lamp GT missed, flagged for review)")

    hard_misses = [f for f in with_gt if f["inter"] == 0]
    print(f"\nHARD MISSES (GT has lamp(s), our mask has zero overlap): {len(hard_misses)}/{len(with_gt)}")
    worst = sorted(with_gt, key=lambda f: (f["containment"] if f["containment"] is not None else 1.0))[:15]
    print("\nworst 15 frames by containment (for joint review):")
    for f in worst:
        print(f"  image_id={f['image_id']:4d} ts={f['ts']} file={f['file_name']} "
              f"containment={f['containment']:.3f} gt_inst={f['n_gt_instances']} "
              f"regions={f['n_regions']} gt_area={f['gt_area']} our_area={f['our_area']}")


if __name__ == "__main__":
    main()
