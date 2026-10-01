#!/usr/bin/env python3
"""verify_lamp_mask_parallel_pool.py -- throwaway check that
static_lamp_mask actually works through the REAL multiprocess worker pool
(src/parallel_search.py), not just the sequential BlobDetector.detect() path
verify_lamp_region_mask_e2e.py already covers. NOT committed.

Drives the exact same real recording/region as that script, but dispatches
every frame through create_pool()/run_blob_detect() with camera/pose_source
registered via register_blob_detector_spec(), round-tripping memory_in/
region_memory_in exactly as main.py's own _run_blob_detect_batch_multi does.
Compares against the sequential BlobDetector.detect() path frame-by-frame --
they should match exactly (same underlying logic, just pickled across a
process boundary).

Usage: python3 verify_lamp_mask_parallel_pool.py
"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_headset_mocap(config):
    from src.mocap_data import DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker
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


def main():
    from src.blob_detector import BlobDetector
    from src.camera import Camera
    from src.headset_pose_source import MocapHeadsetPoseSource
    from src.load_config import load_yaml_config, load_json_config
    from src.parallel_search import create_pool, register_blob_detector_spec, run_blob_detect, warmup_pool
    from src.preprocess_data import _camera_dir

    config = load_yaml_config("./config/config.yml")
    cam_idx = 3
    blob_cfg = copy.deepcopy(config["blob_detection"])
    blob_cfg["lamp_blob_filter"]["static_lamp_mask"]["enabled"] = True

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

    # -- Sequential reference --
    seq_detector = BlobDetector(cam_idx, blob_cfg)
    seq_results = []
    for p in paths:
        ts = int(p.stem)
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[top:]
        result, _ = seq_detector.detect(image, camera=camera, pose_source=pose_source, frame_ts_ns=ts)
        seq_results.append(result.centroids.copy())

    # -- Real multiprocess pool path --
    register_blob_detector_spec(cam_idx, blob_cfg, camera.width, camera.height,
                                 camera=camera, pose_source=pose_source)
    pool = create_pool(max_workers=2)
    warmup_pool(pool, 2, timeout=60.0)

    memory_in = None
    region_memory_in = None
    pool_results = []
    for p in paths:
        ts = int(p.stem)
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[top:]
        fut = pool.submit(run_blob_detect, cam_idx, "test", image,
                           None, 0.0, 1.0, 0.0, False, None, p.name,
                           memory_in, ts, region_memory_in)
        result, canvases, memory_in, region_memory_in, diag = fut.result()
        pool_results.append(result.centroids.copy())
    pool.shutdown(wait=True)

    n_mismatch = 0
    for i, (a, b) in enumerate(zip(seq_results, pool_results)):
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-4):
            n_mismatch += 1
            print(f"frame {i}: MISMATCH  sequential={a.tolist()}  pool={b.tolist()}")
    print(f"\n{len(seq_results)} frames compared, {n_mismatch} mismatches")
    print(f"final region_memory_in: {len(region_memory_in._regions) if region_memory_in else 0} regions")


if __name__ == "__main__":
    main()
