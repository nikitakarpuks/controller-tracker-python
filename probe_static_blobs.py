#!/usr/bin/env python3
"""
probe_static_blobs.py — Throwaway validation of the Phase 1 "static blob"
idea from guidance/temp.txt's static-ambient-light-masking design: does a
VIO-derotation residual actually separate static blobs (e.g. a lamp) from
moving ones, before any tracker/survey-grid/matching-integration code gets
written?

No persistent multi-frame tracker, no survey grid, no wiring into
main.py/pose_search.py — that's Phase 2/3, only worth building if this
number looks good. This script does the ONE measurement Phase 1 is betting
on: for every consecutive frame pair (t0, t1), derotate each t0 blob by the
headset's VIO rotation and see how far the prediction lands from the
nearest actually-detected blob in t1. A static blob (lamp) should predict
almost exactly; a moving blob (controller) won't.

DEROTATION MATH
----------------
For blob pixel p0 in frame t0, its predicted pixel in frame t1 (assuming
it's static in world -- i.e. ignoring translation/parallax, which is the
same simplification the design doc's Phase 1 makes: over one frame, a
static object's bearing change is rotation-dominated):

    d0      = undistort_points(p0)                        # cam-frame ray at t0 (real KB4 inverse, not pinhole)
    R_rel   = R_cam_imu @ R_world_imu(t1).T @ R_world_imu(t0) @ R_imu_cam
    d1_pred = R_rel @ d0
    p1_pred = project_points(d1_pred)                      # redistort via the real KB4 forward model

R_world_imu(t) comes from VIO (imu_data.interpolate_vio). R_imu_cam MUST be
a true camera-to-physical-IMU rotation, not camera-to-cam0: the active
config's kb4_calib_data_tuned.json was tuned under the "T_cam0_camN"
convention and its cam0 T_imu_cam entry is identity (see
src/imu_data.py's load_T_imu_cam docstring) -- composing that with VIO's
IMU-frame rotation would silently produce garbage. The one calibration file
in this repo with a real T_imu_cam is data/cameras/backup/
mateosss.reverbg2v1.kleineinzeigen.bslt.json (KNOWN CAVEAT: fit under a
pinhole-radtan8 intrinsic model, not kb4 -- its extrinsic ROTATION is used
here as the best available approximation of the physical rig geometry;
its intrinsics/translation are not used for anything). If residuals look
suspiciously large even for genuinely static blobs, this mismatch is the
first thing to re-derive properly.

USAGE
-----
python probe_static_blobs.py [--cam 0] [--max-frames N] [--gate-px 12]
                              [--static-px 2.0] [--fps 30]
                              [--lamp-roi x0,y0,x1,y1] [--out-dir DIR]

Reads data.root/cameras/blob_detection straight from config.yml, same as
main.py. Outputs, under --out-dir (default ./visualization/static_blob_probe):
  cam{N}_residuals.csv  — one row per (frame pair, blob): predicted vs.
                          matched position, residual px, matched flag,
                          head_rot_deg (how much the headset actually
                          rotated between t0/t1 -- residuals from
                          near-zero-rotation pairs are uninformative,
                          filter on this before trusting a "static" verdict)
  cam{N}_overlay.mp4    — frames with blobs colored by residual (green =
                          static-looking, orange = moved, red = unmatched)
                          plus a blue '+' at the derotation-predicted point,
                          for eyeballing before trusting any number
  cam{N}_hist.png       — residual histogram (only if matplotlib available)

Prints a Step-0 sanity check up front: the headset's own rotation-rate
distribution across the whole VIO trace. If it barely rotates in this clip,
derotation predicts ~identity everywhere and static vs. moving blobs will
look identical -- that's a data problem, not a signal problem, and no
threshold tuning will fix it.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.imu_data import load_T_imu_cam, load_vio_csv, interpolate_vio, load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data

# See DEROTATION MATH above for why this specific file, not
# config["cameras"]["intrinsics_path"], is the source of R_imu_cam.
_MATEOSSS_EXTRINSICS_PATH = "./data/cameras/backup/mateosss.reverbg2v1.kleineinzeigen.bslt.json"

# Same file/lag mapping main.py uses for gyro-based pose prediction (Stage 1
# cross-correlation, this recording only -- see src/imu_data.py's module
# docstring). Controller IMU is a ground-truth cross-check, independent of
# vision entirely: it says whether THAT controller was moving, regardless of
# whether/where it's visible. It does NOT identify which detected blob is
# which controller's LED -- there's no per-blob identity here, only a
# per-frame-pair "was any controller moving" covariate.
_CTRL_IMU_FILES = {"left_controller":  ("imu1/data.csv", -5_000_000),
                    "right_controller": ("imu2/data.csv", -7_000_000)}

_COLOR_STATIC    = (0, 200, 0)     # BGR green
_COLOR_MOVED     = (0, 140, 255)   # BGR orange
_COLOR_UNMATCHED = (0, 0, 255)     # BGR red
_COLOR_PRED      = (255, 120, 0)   # BGR blue


def _list_frame_timestamps(data_cfg: dict, cam_idx: int) -> list:
    """Mirrors preprocess_data's "per_camera_folders" layout (single-camera
    slice of it) without decoding any images -- just filenames, which this
    recording already encodes as nanosecond timestamps. Only handles that
    one layout; config.yml's active recording uses it."""
    folder = (Path(data_cfg["root"])
              / data_cfg.get("camera_folder_pattern", "cam{idx}").format(idx=cam_idx)
              / data_cfg.get("images_subdir", ""))
    paths = sorted(folder.glob("*.png"))
    fr = data_cfg.get("frame_range") or {}
    paths = paths[fr.get("lower"):fr.get("upper")]
    return [int(p.stem) for p in paths]


def _report_rotation_sanity(t_vio, quat_xyzw_vio):
    """Step 0: is there even enough headset rotation in this clip to test
    derotation against? Computed from VIO's own sample-to-sample deltas,
    independent of any camera/frame_range slicing."""
    rots = Rotation.from_quat(quat_xyzw_vio)
    deltas_deg = np.array([
        (rots[i].inv() * rots[i + 1]).magnitude() * 180.0 / np.pi
        for i in range(len(rots) - 1)
    ])
    dt_s = np.diff(t_vio.astype(np.float64)) / 1e9
    rate_deg_s = deltas_deg / dt_s
    print(f"[sanity] VIO trace: {len(t_vio)} samples, "
          f"per-sample rotation: median={np.median(deltas_deg):.3f} deg, "
          f"p90={np.percentile(deltas_deg, 90):.3f} deg  |  "
          f"implied rate: median={np.median(rate_deg_s):.1f} deg/s, "
          f"p90={np.percentile(rate_deg_s, 90):.1f} deg/s")
    if np.percentile(rate_deg_s, 90) < 2.0:
        print("[sanity] WARNING: headset barely rotates in this clip -- "
              "derotation will predict ~identity everywhere, and static vs. "
              "moving blobs may be indistinguishable regardless of how good "
              "the math is. Consider a segment with more head rotation.")


def _load_controller_gyro_norms(config: dict, data_root: Path) -> dict:
    """{ctrl_key: (t_ns int64[N], gyro_norm_deg_s float[N])} for whichever
    controller IMU files exist -- same mix+bias calibration + axis transform
    + clock lag main.py applies (load_and_calibrate_controller_imu), just
    reduced to a scalar angular-rate magnitude per sample."""
    out = {}
    for ctrl_key, (rel_path, lag_ns) in _CTRL_IMU_FILES.items():
        ctrl_cfg_path = config.get("controllers", {}).get(ctrl_key, {}).get("config_path")
        imu_path = data_root / rel_path
        if not ctrl_cfg_path or not imu_path.exists():
            continue
        t_ns, gyro_body, _accel = load_and_calibrate_controller_imu(
            imu_path, load_json_config(ctrl_cfg_path), lag_ns=lag_ns)
        out[ctrl_key] = (t_ns, np.linalg.norm(gyro_body, axis=1) * 180.0 / np.pi)
    return out


def _gyro_window_max(ctrl_gyro: dict, t0_ns: int, t1_ns: int) -> dict:
    """Max per-controller gyro magnitude (deg/s) with a sample timestamp in
    [t0_ns, t1_ns] -- max, not mean, since a camera frame gap (~11-30ms) can
    hold only a handful of ~200Hz gyro samples and a brief motion spike is
    exactly what we don't want averaged away. NaN if no controller sample
    fell in the window (gap in the IMU stream) or that controller has no
    loaded IMU at all."""
    out = {}
    for ctrl_key, (t_g, norm_g) in ctrl_gyro.items():
        mask = (t_g >= t0_ns) & (t_g <= t1_ns)
        out[ctrl_key] = float(norm_g[mask].max()) if mask.any() else float("nan")
    return out


def _unit_rays(cam: Camera, px: np.ndarray) -> np.ndarray:
    """(N,2) distorted pixels -> (N,3) camera-frame ray directions (x/z, y/z, 1),
    via the real per-camera-model inverse (KB4 bisection for fisheye)."""
    norm = cam.undistort_points(px)
    return np.hstack([norm, np.ones((norm.shape[0], 1), dtype=np.float64)])


def _redistort(cam: Camera, rays: np.ndarray) -> np.ndarray:
    """(N,3) camera-frame directions -> (N,2) pixels via the real forward
    model (cv2.fisheye.projectPoints for kb4). Invariant to positive scale,
    so rays need not be unit-normalized."""
    pts, _ = cam.project_points(rays, rvec=np.zeros(3), tvec=np.zeros(3))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0, help="camera index to probe")
    ap.add_argument("--max-frames", type=int, default=None,
                     help="override config.yml's frame_range upper bound")
    ap.add_argument("--gate-px", type=float, default=12.0,
                     help="max pixel distance for a predicted->actual blob match")
    ap.add_argument("--static-px", type=float, default=2.0,
                     help="matched residual below this is colored 'static' in the overlay")
    ap.add_argument("--fps", type=float, default=30.0, help="overlay video playback fps")
    ap.add_argument("--lamp-roi", type=str, default=None,
                     help="x0,y0,x1,y1 pixel box (in --cam's frame) known to contain "
                          "the static light -- if given, prints residual percentiles "
                          "inside vs. outside this box for a quick separation check")
    ap.add_argument("--out-dir", type=str, default="./visualization/static_blob_probe")
    args = ap.parse_args()

    config = load_yaml_config("./config/config.yml")
    cam_idx = args.cam
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Camera model (intrinsics only -- see module docstring for why
    # T_imu_cam is NOT taken from this file) ────────────────────────────────
    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    cam = Camera(calib_cfg, camera_idx=cam_idx,
                 extrinsics_convention=config["cameras"].get("extrinsics_convention", "T_imu_cam"))

    mateosss_cfg = load_json_config(_MATEOSSS_EXTRINSICS_PATH)
    T_imu_cam = load_T_imu_cam(mateosss_cfg, camera_idx=cam_idx)
    R_imu_cam = T_imu_cam.R
    R_cam_imu = T_imu_cam.inverse().R

    blob_detector = BlobDetector(cam_idx, config["blob_detection"])

    # ── VIO ──────────────────────────────────────────────────────────────
    data_root = Path(config["data"]["root"])
    t_vio, pos_vio, quat_vio = load_vio_csv(data_root / "vio" / "data.csv")
    _report_rotation_sanity(t_vio, quat_vio)

    # ── Controller IMU (independent ground-truth cross-check, see
    # _load_controller_gyro_norms) ──────────────────────────────────────────
    ctrl_gyro = _load_controller_gyro_norms(config, data_root)
    for ctrl_key, (t_g, norm_g) in ctrl_gyro.items():
        print(f"[ctrl-imu] {ctrl_key}: {len(t_g)} samples, "
              f"gyro |w| median={np.median(norm_g):.1f} deg/s, "
              f"p10={np.percentile(norm_g, 10):.1f} deg/s "
              f"(low p10 relative to median = long stretches genuinely at rest)")

    # ── Frame list + per-frame R_world_imu, batched once (cheap: one Slerp
    # build + one vectorised query, vs. rebuilding it every frame) ─────────
    data_cfg = dict(config["data"])
    data_cfg["selected_cameras"] = [cam_idx]
    if args.max_frames is not None:
        data_cfg["frame_range"] = {"lower": 0, "upper": args.max_frames}

    frame_ts = np.array(_list_frame_timestamps(data_cfg, cam_idx), dtype=np.int64)
    in_range = (frame_ts >= t_vio[0]) & (frame_ts <= t_vio[-1])
    n_dropped = int((~in_range).sum())
    if n_dropped:
        print(f"[warn] {n_dropped}/{len(frame_ts)} frames fall outside VIO's covered "
              f"time range and will be skipped (interpolate_vio does not extrapolate)")
    frame_ts = frame_ts[in_range]
    R_world_imu_all, _ = interpolate_vio(frame_ts, t_vio, pos_vio, quat_vio)

    print(f"[run] cam{cam_idx}: {len(frame_ts)} frames in range, "
          f"gate={args.gate_px}px  static_thresh={args.static_px}px")

    lamp_roi = None
    if args.lamp_roi:
        x0, y0, x1, y1 = (float(v) for v in args.lamp_roi.split(","))
        lamp_roi = (x0, y0, x1, y1)

    csv_path = out_dir / f"cam{cam_idx}_residuals.csv"
    video_path = out_dir / f"cam{cam_idx}_overlay.mp4"
    csv_file = open(csv_path, "w", newline="")
    csv_file.write("frame_idx,t0_ns,t1_ns,blob_idx,x0,y0,x1_pred,y1_pred,residual_px,matched,"
                    "head_rot_deg,left_gyro_deg_s,right_gyro_deg_s\n")

    writer = None
    all_residuals = []          # (residual_px or nan, in_roi bool or None)
    frame_level = []            # (frame_idx, frame_max_residual_px or nan, left_gyro_deg_s, right_gyro_deg_s)
    prev = None                 # (image, blobs, R_world_imu, ts_ns)

    for frame_idx, batch in enumerate(get_data(data_cfg)):
        img_path, cam_images = batch[0][0], batch[0][1]
        ts_ns = int(img_path.stem)
        if ts_ns not in frame_ts:
            continue
        image = cam_images[cam_idx]
        blobs, _canvases = blob_detector.detect(image)  # cold, full-frame, no predicted_leds
        idx = int(np.searchsorted(frame_ts, ts_ns))
        R_world_imu_t = R_world_imu_all[idx]

        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      args.fps, (w, h))

        if prev is not None:
            prev_image, prev_blobs, prev_R, prev_ts = prev
            R_rel = R_cam_imu @ R_world_imu_t.T @ prev_R @ R_imu_cam
            head_rot_deg = Rotation.from_matrix(R_rel).magnitude() * 180.0 / np.pi
            gyro_window = _gyro_window_max(ctrl_gyro, prev_ts, ts_ns)
            left_gyro  = gyro_window.get("left_controller", float("nan"))
            right_gyro = gyro_window.get("right_controller", float("nan"))
            frame_max_residual = float("nan")

            canvas = cv2.cvtColor(prev_image, cv2.COLOR_GRAY2BGR)

            if len(prev_blobs) > 0:
                d0 = _unit_rays(cam, prev_blobs.centroids)
                d1_pred = (R_rel @ d0.T).T
                p1_pred = _redistort(cam, d1_pred)

                if len(blobs) > 0:
                    dists = np.linalg.norm(p1_pred[:, None, :] - blobs.centroids[None, :, :], axis=2)
                    nn_dist = dists.min(axis=1)
                else:
                    nn_dist = np.full(len(p1_pred), np.inf)
                matched = nn_dist <= args.gate_px

                for i in range(len(prev_blobs)):
                    x0, y0 = prev_blobs.centroids[i]
                    xp, yp = p1_pred[i]
                    r = float(nn_dist[i]) if matched[i] else float("nan")
                    in_roi = None
                    if lamp_roi is not None:
                        rx0, ry0, rx1, ry1 = lamp_roi
                        in_roi = (rx0 <= x0 <= rx1) and (ry0 <= y0 <= ry1)
                    all_residuals.append((r, in_roi))
                    if matched[i] and not np.isnan(r):
                        frame_max_residual = r if np.isnan(frame_max_residual) else max(frame_max_residual, r)

                    csv_file.write(f"{frame_idx},{prev_ts},{ts_ns},{i},"
                                    f"{x0:.2f},{y0:.2f},{xp:.2f},{yp:.2f},"
                                    f"{r if matched[i] else ''},{bool(matched[i])},{head_rot_deg:.4f},"
                                    f"{left_gyro:.2f},{right_gyro:.2f}\n")

                    color = (_COLOR_UNMATCHED if not matched[i]
                             else _COLOR_STATIC if r <= args.static_px
                             else _COLOR_MOVED)
                    radius = max(3, int(prev_blobs.radii[i]))
                    cv2.circle(canvas, (int(x0), int(y0)), radius, color, 2)
                    cv2.drawMarker(canvas, (int(xp), int(yp)), _COLOR_PRED,
                                    markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

            frame_level.append((frame_idx, frame_max_residual, left_gyro, right_gyro))

            cv2.putText(canvas, f"frame {frame_idx}  head_rot={head_rot_deg:.2f}deg  "
                                 f"L={left_gyro:.0f} R={right_gyro:.0f} deg/s",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)

        prev = (image, blobs, R_world_imu_t, ts_ns)

    if writer is not None:
        writer.release()
    csv_file.close()
    print(f"[out] {csv_path}")
    print(f"[out] {video_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    residuals = np.array([r for r, _ in all_residuals], dtype=np.float64)
    matched_mask = ~np.isnan(residuals)
    n_total = len(residuals)
    n_matched = int(matched_mask.sum())
    print(f"[summary] {n_total} blob observations, {n_matched} matched "
          f"({100.0 * n_matched / max(n_total, 1):.1f}%), "
          f"{n_total - n_matched} unmatched (residual > gate or no blob in t1)")
    if n_matched > 0:
        mr = residuals[matched_mask]
        print(f"[summary] matched residual (px): "
              f"p50={np.percentile(mr, 50):.2f}  p90={np.percentile(mr, 90):.2f}  "
              f"p99={np.percentile(mr, 99):.2f}  max={mr.max():.2f}")

    if lamp_roi is not None:
        in_roi_res  = np.array([r for r, roi in all_residuals if roi and not np.isnan(r)])
        out_roi_res = np.array([r for r, roi in all_residuals if roi is False and not np.isnan(r)])
        if len(in_roi_res) and len(out_roi_res):
            print(f"[roi] in-ROI  (n={len(in_roi_res)}):  "
                  f"p50={np.percentile(in_roi_res, 50):.2f}  p95={np.percentile(in_roi_res, 95):.2f}")
            print(f"[roi] out-ROI (n={len(out_roi_res)}): "
                  f"p05={np.percentile(out_roi_res, 5):.2f}  p50={np.percentile(out_roi_res, 50):.2f}")
            print("[roi] separation is real if in-ROI's p95 sits clearly below out-ROI's p05 -- "
                  "if they overlap, the residual signal alone isn't discriminative on this clip.")
        else:
            print("[roi] not enough matched samples inside/outside the given ROI to compare")

    # ── Controller-IMU cross-check: does this camera's vision residual
    # actually track REAL controller motion? Independent of the vision
    # pipeline entirely -- a frame-pair with high gyro but low residual, or
    # vice versa, is worth a look regardless of what the rest of this script
    # concludes. No per-blob controller identity, so this is a per-frame-pair
    # (max residual across all blobs that frame) vs. (max gyro in that
    # window) comparison, not a per-blob one. ────────────────────────────────
    fl_frame, fl_res, fl_left, fl_right = (np.array(x) for x in zip(*frame_level)) if frame_level else (
        np.array([]),) * 4
    for ctrl_key, fl_gyro in (("left_controller", fl_left), ("right_controller", fl_right)):
        if fl_gyro.size == 0 or np.all(np.isnan(fl_gyro)):
            continue
        valid = ~(np.isnan(fl_res) | np.isnan(fl_gyro))
        if valid.sum() < 3:
            continue
        corr = float(np.corrcoef(fl_res[valid], fl_gyro[valid])[0, 1])
        print(f"[ctrl-imu] {ctrl_key}: corr(frame-max residual, gyro |w|) = {corr:.2f} "
              f"over {int(valid.sum())} frame-pairs -- near 0 means this camera's blobs don't "
              f"track that controller's real motion (consistent with it being out of view or "
              f"only briefly passing through); strongly positive means they do.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if n_matched > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(mr, bins=60)
            ax.set_xlabel("residual (px)")
            ax.set_ylabel("count")
            ax.set_title(f"cam{cam_idx} derotation residual, matched blobs (n={n_matched})")
            fig.tight_layout()
            hist_path = out_dir / f"cam{cam_idx}_hist.png"
            fig.savefig(hist_path, dpi=120)
            print(f"[out] {hist_path}")

        if fl_res.size > 0 and not np.all(np.isnan(fl_res)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, gyro, label in zip(axes, (fl_left, fl_right), ("left_controller", "right_controller")):
                valid = ~(np.isnan(fl_res) | np.isnan(gyro))
                ax.scatter(gyro[valid], fl_res[valid], s=8, alpha=0.4)
                ax.set_xlabel(f"{label} gyro |w| in window (deg/s)")
                ax.set_ylabel("frame-max residual (px)")
                ax.set_title(label)
            fig.suptitle(f"cam{cam_idx}: vision residual vs. controller IMU motion")
            fig.tight_layout()
            scatter_path = out_dir / f"cam{cam_idx}_gyro_vs_residual.png"
            fig.savefig(scatter_path, dpi=120)
            print(f"[out] {scatter_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
