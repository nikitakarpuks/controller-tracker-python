#!/usr/bin/env python3
"""audit_led_mask_safety.py -- throwaway safety audit: run the REAL cold-path
BlobDetector.detect() (static_lamp_mask enabled) over seq1's 350 cam3 frames,
same as eval_lamp_region_mask.py, but this time reproject each REAL,
mocap-tracked controller's REAL LED positions (both controllers, not
synthetic random poses) into the same frame and check whether any land inside
the region mask that was ACTUALLY ACTIVE (hits >= min_hits_to_exclude, i.e.
would really exclude something) at that exact moment. This is the direct
"does this feature ever mask a real controller LED" check -- the concrete
form of "avoid pseudo detections" for this subsystem: a masked real LED is
the actual safety failure mode, not just a wrong-shaped box. NOT committed.

Usage: python3 audit_led_mask_safety.py
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
from src.controller import create_leds_from_config
from src.headset_pose_source import MocapHeadsetPoseSource
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import (DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns,
                             load_T_imu_marker, load_mocap_bridge, relative_pose)
from src.preprocess_data import _camera_dir

SEQ_DIR = Path("/home/nikitakarpuks/Documents/lamp_sequences/seq1")
GT_PATH = SEQ_DIR / "instances_default.json"
CAM_IDX = 3


def _load_device_mocap(recording_root: Path, device_dir_name: str, calib_path: str,
                        offset_override_ns, config: dict) -> DeviceMocap:
    device_dir = recording_root / "mocap_filtered" / device_dir_name
    t_mocap, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    if offset_override_ns is not None:
        fine_offset_ns = float(offset_override_ns)
    else:
        from main import DRIFT_CHECK_VARIANT
        fine_offset_ns = load_mocap_fine_offset_ns(
            device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(calib_path)
    max_gap_ns = float(config.get("mocap", {}).get("max_interp_gap_ms", 30.0)) * 1e6
    return DeviceMocap(t_mocap, position, quat_xyzw, fine_offset_ns, T_imu_marker, max_interp_gap_ns=max_gap_ns)


def main():
    d = json.load(open(GT_PATH))
    images = {im["id"]: im for im in d["images"]}
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

    recording_root = Path(config["data"]["root"]).parent
    headset_mocap = _load_device_mocap(
        recording_root, "headset", config["cameras"]["mocap_calib_path"],
        config["cameras"].get("mocap_fine_offset_override_ns"), config)
    pose_source = MocapHeadsetPoseSource(headset_mocap)

    facing_cos = np.cos(np.radians(float(config["matching"].get("led_facing_angle_deg", 75.0))))

    ctrl_data = {}
    for ctrl_key, mocap_dir in (("right_controller", "ctrlright"), ("left_controller", "ctrlleft")):
        ctrl_cfg = config["controllers"][ctrl_key]
        if not ctrl_cfg.get("enabled", False):
            continue
        ctrl_mocap = _load_device_mocap(
            recording_root, mocap_dir, ctrl_cfg["mocap_calib_path"],
            ctrl_cfg.get("mocap_fine_offset_override_ns"), config)
        bridge = load_mocap_bridge(ctrl_cfg["mocap_bridge_path"])
        with open(ctrl_cfg["config_path"]) as f:
            led_json = json.load(f)
        leds = create_leds_from_config(led_json)
        positions = np.array([l.position for l in leds], dtype=np.float64)
        normals = np.array([l.normal for l in leds], dtype=np.float64)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        ctrl_data[ctrl_key] = dict(mocap=ctrl_mocap, bridge=bridge, positions=positions, normals=normals)

    print(f"auditing {len(ordered_image_ids)} frames, cam{CAM_IDX}, controllers={list(ctrl_data)}")

    detector = BlobDetector(CAM_IDX, blob_cfg)

    incidents = []
    n_led_checks = 0
    n_frames_with_any_ctrl = 0
    for im_id in ordered_image_ids:
        ts = ts_by_image[im_id]
        p = path_by_ts.get(ts)
        if p is None:
            continue
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        image = image[top:] if top else image
        detector.detect(image, predicted_leds=None, camera=camera, pose_source=pose_source,
                         frame_ts_ns=ts, visualize=False)

        T_room_headsetImu = pose_source.room_pose_at(ts)
        active_contours = []
        if T_room_headsetImu is not None and detector._lamp_region_memory is not None:
            from src.static_light_geometry import camera_room_pose
            T_room_cam = camera_room_pose(T_room_headsetImu, camera)
            # The REAL exclusion set (hits >= min_hits_to_exclude gated) --
            # this is what actually removes candidates in production, unlike
            # all_reprojected_contours() (includes brand-new/unreinforced ones).
            active_contours = detector._lamp_region_memory.reprojected_contours(camera, T_room_cam)
        if not active_contours:
            continue

        frame_had_ctrl = False
        for ctrl_key, cd in ctrl_data.items():
            mocap_rel = relative_pose(headset_mocap, cd["mocap"], ts)
            if mocap_rel is None:
                continue
            T_world_ctrl = mocap_rel.compose(cd["bridge"].inverse())
            positions_world = T_world_ctrl.apply(cd["positions"])
            normals_world = T_world_ctrl.R @ cd["normals"].T
            normals_world = normals_world.T
            pts_cam = camera.T_cam_imu.apply(positions_world)
            in_front = pts_cam[:, 2] > 0.05
            if in_front.sum() == 0:
                continue
            normals_cam = camera.T_cam_imu.R @ normals_world.T
            normals_cam = normals_cam.T
            view_dir = -pts_cam / np.linalg.norm(pts_cam, axis=1, keepdims=True)
            facing = np.einsum("ij,ij->i", normals_cam, view_dir) > facing_cos
            visible = in_front & facing
            if visible.sum() == 0:
                continue
            frame_had_ctrl = True
            idx_visible = np.nonzero(visible)[0]
            try:
                rvec = np.zeros(3, dtype=np.float32)
                tvec = np.zeros(3, dtype=np.float32)
                px, _ = camera.project_points(pts_cam[visible].astype(np.float32), rvec, tvec)
            except Exception:
                continue
            px = np.asarray(px).reshape(-1, 2)
            h, w = image.shape[:2]
            for k in range(len(px)):
                x, y = float(px[k, 0]), float(px[k, 1])
                if not (0 <= x < w and 0 <= y < h) or not np.isfinite(x) or not np.isfinite(y):
                    continue
                n_led_checks += 1
                for region_cnt in active_contours:
                    dist = cv2.pointPolygonTest(region_cnt, (x, y), True)
                    if dist >= 0:
                        incidents.append({
                            "image_id": im_id, "file_name": images[im_id]["file_name"],
                            "ts": ts, "ctrl": ctrl_key, "led_idx": int(idx_visible[k]),
                            "px": [x, y], "dist_inside_px": float(dist),
                        })
        if frame_had_ctrl:
            n_frames_with_any_ctrl += 1

    print(f"\nframes with >=1 real, visible, mocap-tracked LED and an ACTIVE mask: {n_frames_with_any_ctrl}")
    print(f"total individual (frame, visible LED) checks against an active mask: {n_led_checks}")
    print(f"INCIDENTS (a real LED landed inside an active exclusion mask): {len(incidents)}")
    for inc in incidents[:30]:
        print(f"  image_id={inc['image_id']:4d} ts={inc['ts']} ctrl={inc['ctrl']} led={inc['led_idx']} "
              f"px={inc['px']} dist_inside_px={inc['dist_inside_px']:.1f}")
    out_path = SEQ_DIR / "audit_led_mask_safety_incidents.json"
    with open(out_path, "w") as f:
        json.dump(incidents, f, indent=2)
    print(f"\nwrote {len(incidents)} incident records to {out_path}")


if __name__ == "__main__":
    main()
