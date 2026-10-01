#!/usr/bin/env python3
"""probe_spacing_max_px_safety.py -- throwaway re-validation of
blob_detection.lamp_blob_filter.spacing_max_px's safety margin, prompted by a
real recording (cam3) whose upper lamp row has a genuine ~30.5px physical gap
(confirmed by direct pixel inspection: pure background noise in between, no
hidden dim element) -- slightly past the current spacing_max_px=25.0, so that
row's leftmost KEPT element never joins a >= min_points chain and leaks into
the real candidate pool every frame 1-13 of the user's own reported sequence.

Reproduces this project's own documented safety methodology (see
config.yml's spacing_max_px/min_points comments and
tests/test_lamp_blob_filter.py's ControllerRingSafetyTests): project the
REAL right-controller LED ring (32 LEDs, this project's own camera model)
through many random poses, check whether detect_lamp_blobs' own line-finder
EVER removes a real, visible LED at production's current min_points=5 --
across a range of candidate spacing_max_px values -- to get a concrete,
reproducible false-positive rate before touching this safety-critical
parameter. NOT committed.

Usage: python3 probe_spacing_max_px_safety.py [--trials 30000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blob_detector import BlobResult
from src.camera import Camera
from src.controller import create_leds_from_config
from src.lamp_blob_filter import detect_lamp_blobs
from src.load_config import load_yaml_config, load_json_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cam", type=int, default=3)
    args = ap.parse_args()

    config = load_yaml_config("./config/config.yml")
    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    cam = Camera(calib_cfg, camera_idx=args.cam,
                 extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))

    with open(config["controllers"]["right_controller"]["config_path"]) as f:
        d = json.load(f)
    leds = create_leds_from_config(d)
    positions = np.array([l.position for l in leds], dtype=np.float64)
    normals = np.array([l.normal for l in leds], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    facing_cos = np.cos(np.radians(float(config["matching"].get("led_facing_angle_deg", 70.0))))

    base_cfg = dict(config["blob_detection"]["lamp_blob_filter"])
    min_points = int(base_cfg.get("min_points", 5))
    area_min = float(base_cfg.get("area_min", 0.4))
    area_max = float(base_cfg.get("area_max", 50.0))
    max_brightness = float(base_cfg.get("max_brightness", 210.0))
    max_line_residual_px = float(base_cfg.get("max_line_residual_px", 2.0))

    spacing_candidates = [25.0, 28.0, 30.0, 32.0, 35.0]

    rng = np.random.default_rng(args.seed)
    n = args.trials

    # Uniform-random SO(3) via random quaternions, uniform-random distance
    # 0.15-1.5m (this project's own real observed controller-camera range),
    # matching the "30,000 uniformly-random orientations/distances" wording
    # in config.yml's own min_points history comment.
    from scipy.spatial.transform import Rotation as Rot
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    rots = Rot.from_quat(quats)
    dists = rng.uniform(0.15, 1.5, size=n)

    checked = 0
    fp_counts = {s: 0 for s in spacing_candidates}
    worst_examples = {s: None for s in spacing_candidates}

    for i in range(n):
        R = rots[i].as_matrix()
        t_cam_ctrl = np.array([0.0, 0.0, dists[i]])
        pts_cam = (R @ positions.T).T + t_cam_ctrl
        normals_cam = (R @ normals.T).T
        in_front = pts_cam[:, 2] > 0.05
        if in_front.sum() < 4:
            continue
        view_dir = -pts_cam / np.linalg.norm(pts_cam, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals_cam, view_dir) > facing_cos
        visible = in_front & facing
        if visible.sum() < 4:
            continue
        pts3d_vis = pts_cam[visible]
        try:
            rvec = np.zeros(3, dtype=np.float32)
            tvec = np.zeros(3, dtype=np.float32)
            px, _ = cam.project_points(pts3d_vis.astype(np.float32), rvec, tvec)
        except Exception:
            continue
        px = np.asarray(px).reshape(-1, 2)
        if not np.isfinite(px).all():
            continue
        in_img = (px[:, 0] > -50) & (px[:, 0] < 700) & (px[:, 1] > -50) & (px[:, 1] < 550)
        if in_img.sum() < 4:
            continue
        px = px[in_img]
        brightness = np.full(len(px), 20.0, dtype=np.float32)
        radii = np.full(len(px), np.sqrt(1.0 / np.pi), dtype=np.float32)
        contours = [np.array([[p[0] - 0.5, p[1] - 0.5], [p[0] + 0.5, p[1] - 0.5],
                               [p[0] + 0.5, p[1] + 0.5], [p[0] - 0.5, p[1] + 0.5]], dtype=np.float32)
                    for p in px]
        blobs = BlobResult(centroids=px, radii=radii, brightnesses=brightness, contours=contours)
        checked += 1
        for s in spacing_candidates:
            cfg = {"max_brightness": max_brightness, "max_line_residual_px": max_line_residual_px,
                   "spacing_min_px": float(base_cfg.get("spacing_min_px", 4.0)), "spacing_max_px": s,
                   "min_points": min_points, "area_min": area_min, "area_max": area_max,
                   "max_lines": int(base_cfg.get("max_lines", 20))}
            result = detect_lamp_blobs(blobs, cfg)
            if not result.keep_mask.all():
                fp_counts[s] += 1
                if worst_examples[s] is None:
                    n_removed = int((~result.keep_mask).sum())
                    worst_examples[s] = (i, n_removed, len(px))

    print(f"checked {checked} valid poses (of {n} sampled)")
    print(f"{'spacing_max_px':>15} {'false_positive_rate':>20} {'count':>8}  first_example(pose_idx,n_removed,n_visible)")
    for s in spacing_candidates:
        rate = fp_counts[s] / checked * 100 if checked else 0.0
        print(f"{s:>15.1f} {rate:>19.3f}% {fp_counts[s]:>8d}  {worst_examples[s]}")


if __name__ == "__main__":
    main()
