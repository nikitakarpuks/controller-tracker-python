#!/usr/bin/env python3
"""probe_lamp_anchor_height.py -- throwaway calibration for
blob_detection.lamp_blob_filter.static_anchor.ceiling_height_m (see
src/lamp_anchor_memory.py). NOT committed as a permanent feature.

Real reprojection-error calibration, NOT self-triangulation and NOT a
self-consistency-spread trick -- both were tried directly against this
recording's own data and found unreliable/degenerate over the short (~150mm)
headset-translation baseline available in this window (see the plan / project
memory for why). Instead: use the real per-frame lamp recognition
(_detect_blobs's own lamp_removed_centroids_out, no anchors involved) to get
independently-detected lamp centroids at an EARLY and a LATER frame, then for a
range of candidate heights, ray-cast the early frame's centroids through that
assumed ceiling plane and reproject into the later frame's own real pose --
the height minimizing real pixel error against the later frame's own
independent detection is the calibrated value.

Usage: python3 probe_lamp_anchor_height.py --cam 3 [--lower 5350] [--upper 5364]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.blob_detector import _detect_blobs
from src.camera import Camera
from src.headset_pose_source import MocapHeadsetPoseSource
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker
from src.preprocess_data import _camera_dir
from src.static_light_geometry import camera_room_pose, pixel_to_room_ray, room_point_to_pixel


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


def _recognized_lamp_centroids(image, cfg) -> np.ndarray:
    """Real, independent per-frame lamp recognition, no anchors involved --
    result[9] is _detect_blobs's own lamp_removed_centroids_out.

    IMPORTANT (found empirically, cam3/this recording): a real fixture here is
    TWO nearly-parallel rows close enough in y that a naive y-gap split
    doesn't reliably separate them (their y-ranges can be only ~1-2px apart at
    the nearest ends even though the rows themselves are ~10px apart overall).
    Both rows get recognized as SEPARATE lines by detect_lamp_blobs, but
    lamp_removed_centroids_out flattens them together -- naively nearest-
    matching against the flat union silently pairs points across the wrong
    row, producing a spurious ~20-25px "error" that looks like a height-
    calibration problem but isn't (confirmed: restricting to one row by hand
    dropped the error to ~2-6px). _detect_blobs's own construction (`for cand
    in lamp_result.rejected_candidates: for gi in cand.blob_indices: ...`)
    keeps each candidate's own points CONTIGUOUS in the flat list, in their
    own along-line order -- so split on a large jump between CONSECUTIVE
    elements in the ORIGINAL (not y-sorted) order instead, which reliably
    finds the true per-line boundary, and keep the largest resulting group."""
    result = _detect_blobs(image, int(cfg["min_threshold"]),
                            min(int(cfg["min_threshold"] * cfg.get("required_threshold_factor", 1.5)), 255),
                            cfg)
    pts = np.array(result[9], dtype=np.float64).reshape(-1, 2)
    if len(pts) < 8:
        # Too few points for the multi-row split below to be safe -- it can
        # spuriously fragment a single real (already-degraded, low-count) row
        # by chance (found empirically: applying it to a real 5-point row
        # split it 3+2). A frame with this few points is exactly the
        # "degraded, near min_points" case this feature targets anyway, where
        # a real fixture practically can't still show two full separate rows.
        return pts
    # Neither a y-gap split nor a large-consecutive-step split reliably
    # separates two close, nearly-parallel rows here (both tried and found
    # unreliable on this exact data -- a row's END can sit geometrically
    # closer to the OTHER row's START than to its own members). Instead: fit
    # one line (PCA/total-least-squares) through ALL points, project onto the
    # direction PERPENDICULAR to it -- two parallel rows separate cleanly
    # along that axis regardless of their own internal x/y ordering -- and
    # split by a 1D gap in that projection (or the single largest gap if none
    # exceeds a fixed threshold, since we don't know row count in general).
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    perp = vt[1]  # 2nd principal axis = perpendicular to the dominant line direction
    proj = centered @ perp
    order = np.argsort(proj)
    gaps = np.diff(proj[order])
    split_i = int(np.argmax(gaps)) + 1  # single largest perpendicular gap = the row boundary
    groups = [pts[order[:split_i]], pts[order[split_i:]]]
    return max(groups, key=len)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=3)
    ap.add_argument("--lower", type=int, default=None, help="override frame_range.lower")
    ap.add_argument("--upper", type=int, default=None, help="override frame_range.upper")
    args = ap.parse_args()

    config = load_yaml_config("./config/config.yml")
    cam_idx = args.cam
    data_cfg = dict(config["data"])
    if args.lower is not None or args.upper is not None:
        data_cfg["frame_range"] = {"lower": args.lower or 0, "upper": args.upper}

    folder = _camera_dir(data_cfg, cam_idx)
    paths = sorted(folder.glob("*.png"))
    fr = data_cfg.get("frame_range") or {}
    paths = paths[fr.get("lower"):fr.get("upper")]
    top = 1 if data_cfg.get("has_technical_row", True) else 0

    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    camera = Camera(calib_cfg, camera_idx=cam_idx,
                     extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))
    headset_mocap = _load_headset_mocap(config)
    pose_source = MocapHeadsetPoseSource(headset_mocap)

    blob_cfg = config["blob_detection"]

    frames = []  # (frame_idx, ts, image, centroids_px, T_room_cam)
    for i, p in enumerate(paths):
        ts = int(p.stem)
        T_room_headsetImu = pose_source.room_pose_at(ts)
        if T_room_headsetImu is None:
            print(f"frame {i}: no mocap coverage, skipping")
            continue
        import cv2
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[top:]
        centroids = _recognized_lamp_centroids(image, blob_cfg)
        T_room_cam = camera_room_pose(T_room_headsetImu, camera)
        frames.append((i, ts, centroids, T_room_cam))
        print(f"frame {i}: {len(centroids)} lamp centroids recognized, cam_pos={T_room_cam.t}")

    # Restrict to the LONGEST CONTIGUOUS initial run of frames with a real
    # recognition -- this is the actual use case (one continuous degrading
    # streak, e.g. frames 1-13 in the reported bug) and, importantly, is the
    # only way to be confident "early" and "later" are the SAME physical row:
    # a later frame reached only after a gap (0 recognized in between) could
    # easily be looking at a DIFFERENT part of a multi-row fixture, which
    # produces a spurious large "error" that looks like a height problem but
    # is actually a row/frame-selection mismatch (found empirically on this
    # exact recording -- see _recognized_lamp_centroids' own docstring for the
    # within-frame version of the same issue).
    contiguous = []
    for f in frames:
        if len(f[2]) == 0:
            break
        contiguous.append(f)
    if len(contiguous) < 2:
        print("Fewer than 2 contiguous frames with real recognized lamp centroids -- cannot calibrate.")
        return

    early = contiguous[0]
    later = contiguous[-1]
    pose_delta_m = np.linalg.norm(later[3].t - early[3].t)
    print(f"\nReference frame {early[0]} -> target frame {later[0]} "
          f"(camera-position delta {pose_delta_m * 1000:.1f}mm)")

    early_centroids, early_T = early[2], early[3]
    later_centroids, later_T = later[2], later[3]

    origin, dirs = pixel_to_room_ray(camera, early_T, early_centroids)

    print("\nheight_m  mean_err_px  max_err_px")
    results = []
    for h in np.arange(1.5, 8.05, 0.1):
        valid = dirs[:, 1] > 1e-6
        t = np.full(len(dirs), np.nan)
        t[valid] = (h - origin[1]) / dirs[valid, 1]
        valid &= t > 0
        if not np.any(valid):
            continue
        pts_room = origin + t[:, None] * dirs
        pts_room = pts_room[valid]
        px_pred = room_point_to_pixel(camera, later_T, pts_room)
        # For each of the LATER (sparser, degraded) frame's own real points,
        # nearest predicted point -- NOT the other direction. The reference
        # frame's row usually has MORE points than the later, degraded frame
        # (that's the whole scenario this feature targets); measuring "for
        # each reference point, nearest later point" wrongly penalizes
        # reference elements that are simply no longer visible at all, which
        # is expected and irrelevant to reprojection accuracy, not an error.
        d = np.linalg.norm(px_pred[:, None, :] - later_centroids[None, :, :], axis=2)
        err = d.min(axis=0)
        results.append((h, err.mean(), err.max()))
        print(f"  {h:.2f}    {err.mean():8.2f}    {err.max():8.2f}")

    if results:
        best = min(results, key=lambda r: r[1])
        print(f"\nBest height (min mean reprojection error): {best[0]:.2f}m "
              f"(mean={best[1]:.2f}px, max={best[2]:.2f}px)")
        print(f"Line-finder tolerances: spacing_min_px={blob_cfg['lamp_blob_filter']['spacing_min_px']} "
              f"spacing_max_px={blob_cfg['lamp_blob_filter']['spacing_max_px']} "
              f"max_line_residual_px={blob_cfg['lamp_blob_filter']['max_line_residual_px']}")


if __name__ == "__main__":
    main()
