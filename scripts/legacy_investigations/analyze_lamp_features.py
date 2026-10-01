#!/usr/bin/env python3
"""analyze_lamp_features.py -- throwaway feature analysis: extract every raw
candidate blob (post threshold+area+circularity screen, pre any lamp-specific
filtering) across all 350 seq1 frames, label each as lamp(1)/not(0) via the
hand-labeled GT polygons, engineer a feature set, and fit classical ML models
(logistic regression + random forest, no NNs) to find out what actually
discriminates a real lamp-row element from background noise / real controller
LEDs on THIS real, dark, noisy recording -- prompted by the user's own
observation that the current heuristic (brightness/area/spacing/residual
thresholds) captures too much noise. NOT committed.

Usage: python3 analyze_lamp_features.py
"""
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blob_detector import _detect_blobs
from src.load_config import load_yaml_config
from src.preprocess_data import _camera_dir

SEQ_DIR = Path("/home/nikitakarpuks/Documents/lamp_sequences/seq1")
GT_PATH = SEQ_DIR / "instances_default.json"
CAM_IDX = 3


def _rasterize_polys(anns, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for ann in anns:
        seg = ann["segmentation"]
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def _shape_features(contour):
    cnt = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0
    if len(cnt) >= 3:
        (rw, rh) = cv2.minAreaRect(cnt)[1]
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
        elongation = long_side / short_side
    else:
        elongation = 1.0
    return area, circularity, elongation


def main():
    d = json.load(open(GT_PATH))
    images = {im["id"]: im for im in d["images"]}
    anns_by_image = {}
    for ann in d["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    ordered_image_ids = sorted(images.keys())
    ts_by_image = {i: int(Path(im["file_name"]).stem) for i, im in images.items()}

    config = load_yaml_config("./config/config.yml")
    blob_cfg = copy.deepcopy(config["blob_detection"])
    blob_cfg["lamp_blob_filter"]["enabled"] = False  # raw candidates only, no lamp-specific logic at all

    folder = _camera_dir(config["data"], CAM_IDX)
    all_paths = sorted(folder.glob("*.png"))
    top = 1 if config["data"].get("has_technical_row", True) else 0
    path_by_ts = {int(p.stem): p for p in all_paths}

    pixel_threshold = int(blob_cfg["min_threshold"])
    req_factor = float(blob_cfg.get("required_threshold_factor", 1.5))
    required_threshold = min(int(pixel_threshold * req_factor), 255)

    rows = []  # list of dicts, one per blob
    n_frames_used = 0
    for im_id in ordered_image_ids:
        ts = ts_by_image[im_id]
        p = path_by_ts.get(ts)
        if p is None:
            continue
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        image = image[top:] if top else image
        h, w = image.shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0

        raw = _detect_blobs(image, pixel_threshold, required_threshold, blob_cfg, visualize=False)
        centroids, contours, radii, brightnesses = raw[0], raw[1], raw[2], raw[3]
        if len(centroids) == 0:
            n_frames_used += 1
            continue
        centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)

        gt_mask = _rasterize_polys(anns_by_image.get(im_id, []), (h, w))

        # neighbor/row-context features: for each blob, look at same-frame
        # neighbors within a generous radius (matches spacing_max_px's own
        # ballpark) to get a density + local-collinearity signal, mimicking
        # (loosely) what detect_lamp_blobs' own line-finder looks for.
        n = len(centroids)
        dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        nn_dist = dists.min(axis=1) if n > 1 else np.full(n, -1.0)
        neighbor_mask = dists < 45.0
        neighbor_count = neighbor_mask.sum(axis=1)

        for i in range(n):
            area, circularity, elongation = _shape_features(contours[i])
            cx, cy = centroids[i]
            rpix = float(np.hypot(cx - cx0, cy - cy0))

            # local collinearity residual: fit a line through this blob + its
            # neighbors (if enough), measure this blob's own perpendicular
            # distance to that fit -- low residual = "sits neatly in a row"
            residual = -1.0
            nbr_idx = np.nonzero(neighbor_mask[i])[0]
            if len(nbr_idx) >= 2:
                pts = centroids[np.append(nbr_idx, i)]
                pts_c = pts - pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts_c)
                direction = vt[0]
                normal = np.array([-direction[1], direction[0]])
                residual = float(abs((centroids[i] - pts.mean(axis=0)) @ normal))

            label = int(gt_mask[int(round(cy)), int(round(cx))]) if (0 <= int(round(cy)) < h and 0 <= int(round(cx)) < w) else 0
            rows.append(dict(
                image_id=im_id, cx=cx, cy=cy, area=float(area), max_pix=float(brightnesses[i]),
                radius=float(radii[i]), circularity=float(circularity), elongation=float(elongation),
                rpix=rpix, nn_dist=float(nn_dist[i]), neighbor_count=int(neighbor_count[i]),
                row_residual=residual, label=label,
            ))
        n_frames_used += 1

    print(f"{n_frames_used} frames processed, {len(rows)} raw candidate blobs extracted")
    n_pos = sum(r["label"] for r in rows)
    print(f"label distribution: {n_pos} lamp / {len(rows) - n_pos} not-lamp "
          f"({n_pos/len(rows)*100:.1f}% positive)")

    out_path = SEQ_DIR / "blob_features.json"
    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
