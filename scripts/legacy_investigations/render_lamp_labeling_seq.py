#!/usr/bin/env python3
"""render_lamp_labeling_seq.py -- throwaway script: run cold/brute blob
detection (no proximity/warm tracking, no static-lamp-mask filtering) over
every frame in a standalone image sequence, and save the pass1 debug canvas
(legend strip cropped off) for manual lamp labeling in CVAT. NOT committed.

- Proximity/warm tracking disabled: every frame calls detect() with
  predicted_leds=None (has_prior=False), so every frame runs the full cold
  two-pass detection independently -- no cross-frame prediction state needed
  ("no-history detection" is fine, per the user's own framing).
- lamp_blob_filter.enabled forced False: this disables BOTH the row-finder
  that would normally remove recognized lamp candidates from the kept/LED
  output AND static_lamp_mask (nested under it, moot once the outer filter
  is off) -- so every real blob (lamp or controller LED) shows up in its
  normal, unbiased category color (white "kept", cyan "split", etc.),
  giving a clean view of literal blob-detection output with none of our own
  lamp-vs-not guesses baked in, for the user to label independently.
- Legend strip removed: BlobDetector's own visualize block appends a legend
  strip below the drawn canvas; cropped back to the source image's own
  height here rather than touching src/blob_detector.py for a one-off
  labeling need.

Usage: python3 render_lamp_labeling_seq.py <input_dir> <output_dir>
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blob_detector import BlobDetector
from src.load_config import load_yaml_config


def main():
    if len(sys.argv) != 3:
        print("usage: python3 render_lamp_labeling_seq.py <input_dir> <output_dir>")
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config("./config/config.yml")
    blob_cfg = config["blob_detection"]
    blob_cfg = dict(blob_cfg)  # shallow copy of the top level is enough --
    blob_cfg["lamp_blob_filter"] = dict(blob_cfg["lamp_blob_filter"])
    blob_cfg["lamp_blob_filter"]["enabled"] = False   # disables row-finder AND static_lamp_mask

    paths = sorted(in_dir.glob("*.png"))
    if not paths:
        print(f"no PNG files found in {in_dir}")
        sys.exit(1)
    print(f"{len(paths)} frames found in {in_dir}")

    detector = BlobDetector(camera_idx=0, cfg=blob_cfg)

    for i, p in enumerate(paths):
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        result, canvases = detector.detect(
            image, predicted_leds=None, visualize=True)
        canvas = canvases.get("pass1")
        if canvas is None:
            print(f"  frame {i} ({p.name}): no pass1 canvas produced, skipping")
            continue
        cropped = canvas[:image.shape[0], :image.shape[1]]
        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), cropped)
        if i % 25 == 0:
            print(f"  frame {i}/{len(paths)}: {len(result.centroids)} kept centroids -> {out_path.name}")

    print(f"done -- wrote {len(paths)} images to {out_dir}")


if __name__ == "__main__":
    main()
