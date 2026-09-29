#!/usr/bin/env python3
"""Coverage decomposition: for every ground-truth (GT) frame WITHOUT a reported pose, was tracking
physically impossible or an algorithmic failure? Uses mocap alone plus the pipeline's own visibility model.

Method
  1. GT = mocap pose of the controller relative to the headset IMU (T_headsetImu_ctrlImu). The LED-reference-
     frame pose in the pipeline's world frame (= headset IMU frame, Camera.T_world_cam = T_imu_cam) is
     T_world_ledRef = T_gt @ bridge^-1  (evaluate_mocap.py's residual is (T_est @ bridge)^-1 @ T_gt, so this is
     the exact inverse of its composition; the bridge is composed, never skipped).
  2. For each of the 4 cameras the 32 LEDs are projected with the calibrated kb4 model and scored with the
     repository's own src/_visibility.py::_visible_mask (facing cone led_facing_angle_deg, frustum/handle
     occlusion with visibility_occlusion_margin_m, in-frame + rpmax check) and, when the other controller's GT
     exists at the same timestamp, src/_visibility.py::_cross_occluded_mask -- the same calls proximity_search makes.
  3. "Expected-visible" LEDs of a camera = score >= proximity_vis_score_threshold (0.95): the pipeline's own
     expected-visible set. A lenient upper bound (score > 0, i.e. facing angle < 90 deg and not fully occluded)
     is computed as a sensitivity check.
  4. Every GT frame is classified by max-over-cameras expected-visible count N:
        tracked                      a fused pose was reported (per_frame_errors_*.csv)
        lost, N < min_inliers (5)    physically impossible: no camera can see enough LEDs for a solve
        lost, 5 <= N < 8             visible enough for a solve, but none was reported
        lost, N >= 8                 clearly visible (>= vision_weight_strong_inliers), none was reported
  5. Sanity checks: (a) tracked frames should nearly all have N >= 4-5 and the logged inlier_count should not
     exceed the summed lenient-visible count; (b) image check on sampled frames: detect blobs with the pipeline's
     BlobDetector (cold path) and measure how many expected-visible LEDs have a blob within 3/6 px.

Code is imported from a read-only snapshot of committed HEAD (--code); data is read from the live repo, never written.
Usage:
  git -C <repo> archive HEAD | tar -x -C <snapshot>
  python3 TUM-THESIS/scripts/coverage_decomposition.py --code <snapshot> [--skip-compute] [--skip-image-check]
Outputs (all under TUM-THESIS/): figures/coverage_decomposition.pdf, figures/tables/coverage_decomposition*.tex,
  figures/snippets/coverage_decomposition.tex, figures/coverage_decomposition.json, figures/data/coverage/*.npz
"""
import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

THESIS = Path(__file__).resolve().parents[1]
REPO = THESIS.parent
FIG = THESIS / "figures"
RECS = ["static_dark", "static_easy", "static_medium", "static_hard",
        "walk_dark", "walk_easy", "walk_medium", "walk_hard"]
CTRLS = ["left_controller", "right_controller"]
CODE = REPO  # overwritten by --code before any repo import
BATCH_SRC = REPO / "visualization" / "evaluate_2026-09-27_full"
PERFRAME = FIG / "data" / "batch_0927"
NPZ_DIR = FIG / "data" / "coverage"


# ------------------------------------------------------------------------------------ setup helpers
def _imports():
    sys.path.insert(0, str(CODE))
    global Camera, _visible_mask, _cross_occluded_mask, _compute_geometry, mirror_primitives
    global load_mocap_bridge, load_yaml_config, load_json_config, Transform, BlobDetector
    from src.camera import Camera
    from src._visibility import _visible_mask, _cross_occluded_mask
    from src.geometry import _compute_geometry
    from src.controller import mirror_primitives
    from src.mocap_data import load_mocap_bridge
    from src.load_config import load_yaml_config, load_json_config
    from src.transformations import Transform
    from src.blob_detector import BlobDetector
    spec = importlib.util.spec_from_file_location("evaluate_mocap", REPO / "evaluate_mocap.py")
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    return em


def _repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def build_rec(rec):
    """Everything needed for one recording, built from that recording's own config snapshot."""
    cfg = load_yaml_config(str(BATCH_SRC / rec / "config.yml"))
    calib = load_json_config(str(_repo_path(cfg["cameras"]["intrinsics_path"])))
    cams = [Camera(calib, i) for i in cfg["data"]["selected_cameras"]]
    models, geoms, bridges = {}, {}, {}
    right_prim = cfg["controllers"]["right_controller"].get("handle_primitives")
    for ctrl in CTRLS:
        ccfg = cfg["controllers"][ctrl]
        js = load_json_config(str(_repo_path(ccfg["config_path"])))
        leds = js["CalibrationInformation"]["ControllerLeds"]
        pos = np.array([l["Position"] for l in leds], dtype=np.float32)
        nrm = np.array([l["Normal"] for l in leds], dtype=np.float32)
        geo = dict(cfg.get("geometry", {}))
        if "handle_primitives" in ccfg:
            geo["handle_primitives"] = ccfg["handle_primitives"]
        elif ctrl == "left_controller" and right_prim is not None:
            geo["handle_primitives"] = mirror_primitives(right_prim)
        models[ctrl] = (pos, nrm)
        geoms[ctrl] = _compute_geometry(pos, nrm, geo)
        bridges[ctrl] = load_mocap_bridge(str(_repo_path(ccfg["mocap_bridge_path"])))
    return cfg, cams, models, geoms, bridges


def load_tracked(rec, ctrl):
    """{timestamp_ns: True} for GT frames with a fused pose, from the per-frame errors CSV."""
    import csv
    out = {}
    with open(PERFRAME / rec / f"per_frame_errors_{ctrl}.csv") as f:
        r = csv.reader(f)
        head = next(r)
        ix = head.index("fused_x")
        for row in r:
            out[int(row[0])] = row[ix] != ""
    return out


# ------------------------------------------------------------------------------------ per-pair compute
def process_pair(args):
    rec, ctrl = args
    em = _imports()
    cfg, cams, models, geoms, bridges = build_rec(rec)
    mt = cfg["matching"]
    face = float(mt["led_facing_angle_deg"])
    margin = float(mt.get("visibility_occlusion_margin_m", 0.0))
    thr = float(mt.get("proximity_vis_score_threshold", 0.95))
    occ_r = float(mt.get("cross_occlusion_bounding_radius_m", 0.18))
    occ_m = float(mt.get("cross_occlusion_gate_margin_px", 20.0))
    use_cross = bool(mt.get("cross_controller_occlusion", True))
    other = "right_controller" if ctrl == "left_controller" else "left_controller"

    gt = {c: em.load_mocap_gt_csv(BATCH_SRC / rec / "mocap_gt" / f"{c}_mocap_gt.csv") for c in CTRLS}
    tracked = load_tracked(rec, ctrl)
    ts_all = sorted(gt[ctrl])
    pos, nrm = models[ctrl]
    pos_o = models[other][0]
    T_cw = [c.T_world_cam.inverse() for c in cams]  # world -> camera

    n = len(ts_all)
    ncam = len(cams)
    Nstrict = np.zeros((n, ncam), np.int8)
    Nlen = np.zeros((n, ncam), np.int8)
    Ninfr = np.zeros((n, ncam), np.int8)
    centre_in = np.zeros((n, ncam), bool)
    is_tr = np.zeros(n, bool)
    for i, ts in enumerate(ts_all):
        is_tr[i] = tracked.get(ts, False)
        T_w = gt[ctrl][ts].compose(bridges[ctrl].inverse())            # LED-ref frame in world (headset IMU) frame
        T_wo = gt[other][ts].compose(bridges[other].inverse()) if ts in gt[other] else None
        for ci, cam in enumerate(cams):
            Tc = T_cw[ci].compose(T_w)
            R, t = Tc.R.astype(np.float64), Tc.t.astype(np.float64)
            sc = _visible_mask(R, t, pos, nrm, geoms[ctrl], cam_K=cam.camera_matrix, cam_dc=cam.dist_coeffs,
                               cam_w=cam.width, cam_h=cam.height, cam_rpmax=cam.rpmax,
                               cam_is_fisheye=cam.is_fisheye, facing_threshold_deg=face, occlusion_margin_m=margin)
            if use_cross and T_wo is not None:
                To = T_cw[ci].compose(T_wo)
                blocked = _cross_occluded_mask(R, t, pos, To.R.astype(np.float64), To.t.astype(np.float64),
                                               geoms[other], occ_r, occ_r, float(max(cam.fx, cam.fy)), occ_m)
                sc = np.where(blocked, 0.0, sc)
            Nstrict[i, ci] = int((sc >= thr).sum())
            Nlen[i, ci] = int((sc > 0.0).sum())
            # geometric in-frame count, ignoring facing/occlusion
            uv, _ = cam.project_points(pos, R, t)
            led_c = (R @ pos.T).T + t
            r_px = np.hypot(uv[:, 0] - cam.cx, uv[:, 1] - cam.cy)
            infr = (led_c[:, 2] > 1e-6) & (r_px <= cam.rpmax_px) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) \
                   & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
            Ninfr[i, ci] = int(infr.sum())
            uc, _ = cam.project_points(np.zeros((1, 3), np.float32), R, t)
            rc = float(np.hypot(uc[0, 0] - cam.cx, uc[0, 1] - cam.cy))
            centre_in[i, ci] = bool(t[2] > 1e-6 and rc <= cam.rpmax_px and 0 <= uc[0, 0] < cam.width
                                    and 0 <= uc[0, 1] < cam.height)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_DIR / f"{rec}__{ctrl}.npz", ts=np.array(ts_all, np.int64), tracked=is_tr,
                        Nstrict=Nstrict, Nlen=Nlen, Ninfr=Ninfr, centre_in=centre_in)
    return rec, ctrl, n


# ------------------------------------------------------------------------------------ aggregation
def classify(d, key, min_in=5, strong=8):
    N = d[key].max(axis=1)
    tr = d["tracked"]
    lost = ~tr
    return {
        "n_gt": int(len(tr)),
        "tracked": int(tr.sum()),
        "lost": int(lost.sum()),
        "lost_lt5": int((lost & (N < min_in)).sum()),
        "lost_5to7": int((lost & (N >= min_in) & (N < strong)).sum()),
        "lost_ge8": int((lost & (N >= strong)).sum()),
    }


def aggregate():
    out = {"per_recording": {}, "pooled": {}}
    pooled = {k: 0 for k in ["n_gt", "tracked", "lost", "lost_lt5", "lost_5to7", "lost_ge8"]}
    pooled_len = dict(pooled)
    hist_tr, hist_lost = np.zeros(33, int), np.zeros(33, int)
    hist_tr_l, hist_lost_l = np.zeros(33, int), np.zeros(33, int)
    lost_total = lost_no_led_infr = lost_centre_out = lost_all_out_strict0 = 0
    trk_lt4 = trk_lt5 = trk_total = 0
    for rec in RECS:
        out["per_recording"][rec] = {}
        for ctrl in CTRLS:
            p = NPZ_DIR / f"{rec}__{ctrl}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            s, l = classify(d, "Nstrict"), classify(d, "Nlen")
            out["per_recording"][rec][ctrl] = {"primary": s, "lenient": l}
            for k in pooled:
                pooled[k] += s[k]
                pooled_len[k] += l[k]
            Ns, Nl = d["Nstrict"].max(1), d["Nlen"].max(1)
            tr = d["tracked"]
            np.add.at(hist_tr, Ns[tr], 1)
            np.add.at(hist_lost, Ns[~tr], 1)
            np.add.at(hist_tr_l, Nl[tr], 1)
            np.add.at(hist_lost_l, Nl[~tr], 1)
            lost = ~tr
            lost_total += int(lost.sum())
            lost_no_led_infr += int((lost & (d["Ninfr"].max(1) == 0)).sum())
            lost_centre_out += int((lost & (~d["centre_in"].any(1))).sum())
            trk_total += int(tr.sum())
            trk_lt4 += int((tr & (Ns < 4)).sum())
            trk_lt5 += int((tr & (Ns < 5)).sum())
    out["pooled"] = {"primary": pooled, "lenient": pooled_len,
                     "hist_maxN_primary": {"tracked": hist_tr.tolist(), "lost": hist_lost.tolist()},
                     "hist_maxN_lenient": {"tracked": hist_tr_l.tolist(), "lost": hist_lost_l.tolist()},
                     "lost_total": lost_total,
                     "lost_no_led_projects_in_frame_of_any_camera": lost_no_led_infr,
                     "lost_controller_centre_outside_all_cameras": lost_centre_out,
                     "tracked_total": trk_total, "tracked_with_maxN_lt4": trk_lt4, "tracked_with_maxN_lt5": trk_lt5}
    return out


# ------------------------------------------------------------------------------------ sanity checks
def sanity_inliers(agg):
    """Logged inlier_count of tracked frames vs the summed lenient-visible count and the max primary count."""
    import csv
    res = {}
    tot = ok = 0
    over_by = []
    for rec in RECS:
        for ctrl in CTRLS:
            p = NPZ_DIR / f"{rec}__{ctrl}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            inl = {}
            with open(BATCH_SRC / rec / "pose.csv") as f:
                for row in csv.DictReader(f):
                    if row["ctrl_name"] == ctrl:
                        inl[int(row["timestamp_ns"])] = int(row["inlier_count"])
            ts = d["ts"]
            for i in np.where(d["tracked"])[0]:
                v = inl.get(int(ts[i]))
                if v is None:
                    continue
                tot += 1
                lim = int(d["Nlen"][i].sum())
                if v <= lim:
                    ok += 1
                else:
                    over_by.append(v - lim)
    res = {"n": tot, "inlier_count_le_sum_lenient_visible": ok,
           "fraction": ok / max(tot, 1), "median_excess_when_violated": float(np.median(over_by)) if over_by else 0.0}
    return res


def sanity_images(em, recs=("static_medium", "walk_medium"), n_per_class=30, seed=0):
    """Blob-vs-projection check on sampled frames. Two questions:
      (i) validate the visibility model + detector: on TRACKED frames, project the tracker's OWN estimated pose
          (fused pose.csv) and count expected-visible LEDs with a cold-path blob within 3/6 px;
      (ii) validate the mocap-derived projection: pixel offset between GT-projected and estimated-pose-projected LEDs,
          and blob hit-rates of the GT projection at loose tolerances (mocap alignment error is several px);
      then for LOST frames (GT projection only): do blobs exist where the visible LEDs should be?"""
    import cv2
    rng = np.random.default_rng(seed)
    out = {}
    for rec in recs:
        cfg, cams, models, geoms, bridges = build_rec(rec)
        root = Path(cfg["data"]["root"])
        top = 1 if cfg["data"].get("has_technical_row", True) else 0
        off = int(cfg["data"].get("controller_cam_start_index", 0))
        gt = {c: em.load_mocap_gt_csv(BATCH_SRC / rec / "mocap_gt" / f"{c}_mocap_gt.csv") for c in CTRLS}
        est = em.load_pose_csv(BATCH_SRC / rec / "pose.csv")
        dets = [BlobDetector(i, cfg["blob_detection"]) for i in range(len(cams))]
        T_cw = [c.T_world_cam.inverse() for c in cams]
        mt = cfg["matching"]
        face, marg = float(mt["led_facing_angle_deg"]), float(mt.get("visibility_occlusion_margin_m", 0.0))
        thr = float(mt.get("proximity_vis_score_threshold", 0.95))

        def vis_uv(cam, ci, T_w, pos, nrm, geom):
            Tc = T_cw[ci].compose(T_w)
            R, t = Tc.R.astype(np.float64), Tc.t.astype(np.float64)
            sc = _visible_mask(R, t, pos, nrm, geom, cam_K=cam.camera_matrix, cam_dc=cam.dist_coeffs, cam_w=cam.width,
                               cam_h=cam.height, cam_rpmax=cam.rpmax, cam_is_fisheye=cam.is_fisheye,
                               facing_threshold_deg=face, occlusion_margin_m=marg)
            vis = np.where(sc >= thr)[0]
            uv, _ = cam.project_points(pos, R, t)
            return vis, uv

        for ctrl in CTRLS:
            d = np.load(NPZ_DIR / f"{rec}__{ctrl}.npz")
            Ns = d["Nstrict"].max(1)
            classes = {"tracked": np.where(d["tracked"] & (Ns >= 5))[0],
                       "lost_lt5": np.where(~d["tracked"] & (Ns < 5))[0],
                       "lost_ge8": np.where(~d["tracked"] & (Ns >= 8))[0]}
            pos, nrm = models[ctrl]
            for cname, idx in classes.items():
                if len(idx) == 0:
                    continue
                pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
                acc = {"gt_exp": 0, "gt6": 0, "gt10": 0, "gt15": 0, "est_exp": 0, "est3": 0, "est6": 0}
                offsets, fr_gt10, fr_est3, n_est_frames = [], 0, 0, 0
                for i in pick:
                    ts = int(d["ts"][i])
                    T_gt = gt[ctrl][ts].compose(bridges[ctrl].inverse())
                    T_es = est.get(ctrl, {}).get(ts) if cname == "tracked" else None
                    best_gt = best_es = 0
                    for ci, cam in enumerate(cams):
                        img_p = root / f"cam{ci + off}" / cfg["data"].get("images_subdir", "data") / f"{ts}.png"
                        if not img_p.exists():
                            continue
                        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)[top:].copy()
                        dets[ci]._memory = {}
                        r = dets[ci].detect(img, ctrl_label=ctrl)
                        res = r[0] if isinstance(r, tuple) else r
                        bl = np.asarray(res.centroids, np.float64).reshape(-1, 2)

                        def nearest(uv):
                            if len(bl) == 0:
                                return np.full(len(uv), 1e9)
                            return np.linalg.norm(uv[:, None, :] - bl[None, :, :], axis=2).min(1)

                        vis, uv = vis_uv(cam, ci, T_gt, pos, nrm, geoms[ctrl])
                        if len(vis):
                            dd = nearest(uv[vis])
                            acc["gt_exp"] += len(vis)
                            acc["gt6"] += int((dd <= 6).sum()); acc["gt10"] += int((dd <= 10).sum())
                            acc["gt15"] += int((dd <= 15).sum())
                            best_gt = max(best_gt, int((dd <= 10).sum()))
                        if T_es is not None:
                            vis_e, uv_e = vis_uv(cam, ci, T_es, pos, nrm, geoms[ctrl])
                            if len(vis_e):
                                de = nearest(uv_e[vis_e])
                                acc["est_exp"] += len(vis_e)
                                acc["est3"] += int((de <= 3).sum()); acc["est6"] += int((de <= 6).sum())
                                best_es = max(best_es, int((de <= 3).sum()))
                            if len(vis):
                                offsets.extend(np.linalg.norm(uv[vis] - uv_e[vis], axis=1).tolist())
                    fr_gt10 += int(best_gt >= 5)
                    if T_es is not None:
                        n_est_frames += 1
                        fr_est3 += int(best_es >= 5)
                rec_out = {"frames_sampled": int(len(pick)),
                           "gt_expected_visible_leds": acc["gt_exp"],
                           "gt_frac_blob_within_6px": acc["gt6"] / max(acc["gt_exp"], 1),
                           "gt_frac_blob_within_10px": acc["gt10"] / max(acc["gt_exp"], 1),
                           "gt_frac_blob_within_15px": acc["gt15"] / max(acc["gt_exp"], 1),
                           "gt_frac_frames_ge5_led_blob_matches_10px_in_one_camera": fr_gt10 / len(pick)}
                if cname == "tracked" and n_est_frames:
                    rec_out.update({"est_expected_visible_leds": acc["est_exp"],
                                    "est_frac_blob_within_3px": acc["est3"] / max(acc["est_exp"], 1),
                                    "est_frac_blob_within_6px": acc["est6"] / max(acc["est_exp"], 1),
                                    "est_frac_frames_ge5_led_blob_matches_3px_in_one_camera": fr_est3 / n_est_frames,
                                    "median_px_offset_gt_vs_estimated_projection": float(np.median(offsets)) if offsets else None,
                                    "p90_px_offset_gt_vs_estimated_projection": float(np.percentile(offsets, 90)) if offsets else None})
                out.setdefault(rec, {}).setdefault(ctrl, {})[cname] = rec_out
                print(f"  image check {rec}/{ctrl}/{cname}: {rec_out}", flush=True)
    return out


# ------------------------------------------------------------------------------------ figure / tables
def report(agg, sanity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertinus Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4, "savefig.bbox": "tight"})
    BLUE, VERM, GREEN, GRAY, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#B0B0B0", "#E69F00", "#56B4E9"
    cols = [BLUE, GRAY, ORANGE, VERM]
    labs = ["tracked", "lost, fewer than 5 visible LEDs (physical limit)",
            "lost, 5--7 visible LEDs", "lost, $\\geq 8$ visible LEDs"]
    labs = [l.replace("--", "\u2013") for l in labs]
    per = agg["per_recording"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.6, 1.4]})
    xs, xt, xl = [], [], []
    x = 0.0
    for rec in RECS:
        for ctrl in CTRLS:
            if ctrl not in per.get(rec, {}):
                continue
            s = per[rec][ctrl]["primary"]
            n = s["n_gt"]
            parts = [s["tracked"], s["lost_lt5"], s["lost_5to7"], s["lost_ge8"]]
            bot = 0.0
            for p, c in zip(parts, cols):
                ax.bar(x, 100.0 * p / n, bottom=bot, width=0.85, color=c, edgecolor="white", linewidth=0.3)
                bot += 100.0 * p / n
            x += 1
        xt.append(x - 1.5)
        xl.append(rec.replace("_", "\n"))
        x += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of ground-truth frames [%]")
    ax.set_title("(a) left/right bar per recording", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labs)], loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0)

    ht = np.array(agg["pooled"]["hist_maxN_primary"]["tracked"], float)
    hl = np.array(agg["pooled"]["hist_maxN_primary"]["lost"], float)
    k = np.arange(len(ht))
    ax2.step(k, np.cumsum(ht) / ht.sum(), where="post", color=BLUE, label="tracked frames")
    ax2.step(k, np.cumsum(hl) / hl.sum(), where="post", color=VERM, label="lost frames")
    ax2.axvline(5, color="k", ls=":", lw=0.8)
    ax2.set_xlim(0, 17)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("visible LEDs $N$ (best camera)")
    ax2.set_ylabel("cumulative fraction")
    ax2.set_title("(b) pooled, all recordings", loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)
    fig.savefig(FIG / "coverage_decomposition.pdf")
    plt.close(fig)

    # ---- tables
    def pct(a, b):
        return f"{100.0 * a / b:.1f}" if b else "--"

    rows = []
    for rec in RECS:
        first = True
        for ctrl in CTRLS:
            if ctrl not in per.get(rec, {}):
                continue
            s = per[rec][ctrl]["primary"]
            n, lost = s["n_gt"], s["lost"]
            name = "\\texttt{" + rec.replace("_", "\\_") + "}" if first else ""
            rows.append(f"{name} & {ctrl.split('_')[0]} & {n} & {pct(s['tracked'], n)} & {pct(lost, n)} & "
                        f"{pct(s['lost_lt5'], n)} & {pct(s['lost_5to7'], n)} & {pct(s['lost_ge8'], n)} & "
                        f"{pct(s['lost_lt5'], lost)} \\\\")
            first = False
        rows.append("\\addlinespace")
    P = agg["pooled"]["primary"]
    body = "\n".join(rows[:-1])
    tab = ("\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
           "Recording & Ctrl. & GT frames & Tracked & Lost & \\multicolumn{3}{c}{of GT frames: lost with $N$ visible} & "
           "Lost with $N<5$ \\\\\n\\cmidrule(lr){6-8}\n"
           " & & & [\\%] & [\\%] & $N<5$ [\\%] & $5\\le N<8$ [\\%] & $N\\ge8$ [\\%] & of lost [\\%] \\\\\n\\midrule\n"
           f"{body}\n\\midrule\n"
           f"pooled & both & {P['n_gt']} & {pct(P['tracked'], P['n_gt'])} & {pct(P['lost'], P['n_gt'])} & "
           f"{pct(P['lost_lt5'], P['n_gt'])} & {pct(P['lost_5to7'], P['n_gt'])} & {pct(P['lost_ge8'], P['n_gt'])} & "
           f"{pct(P['lost_lt5'], P['lost'])} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    (FIG / "tables").mkdir(exist_ok=True)
    (FIG / "tables" / "coverage_decomposition.tex").write_text(tab)

    L = agg["pooled"]["lenient"]
    tab2 = ("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nDefinition of ``visible'' & Lost [\\%] & $N<5$ & $5\\le N<8$ & $N\\ge8$ \\\\\n"
            " & & \\multicolumn{3}{c}{share of lost frames [\\%]} \\\\\n\\cmidrule(lr){3-5}\n\\midrule\n"
            f"Pipeline expected-visible (score $\\geq0.95$) & {pct(P['lost'], P['n_gt'])} & {pct(P['lost_lt5'], P['lost'])} & "
            f"{pct(P['lost_5to7'], P['lost'])} & {pct(P['lost_ge8'], P['lost'])} \\\\\n"
            f"Lenient upper bound (facing $<90^\\circ$, not fully occluded) & {pct(L['lost'], L['n_gt'])} & "
            f"{pct(L['lost_lt5'], L['lost'])} & {pct(L['lost_5to7'], L['lost'])} & {pct(L['lost_ge8'], L['lost'])} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n")
    (FIG / "tables" / "coverage_decomposition_pooled.tex").write_text(tab2)

    cap = (f"Decomposition of the ground-truth frames of each recording (2026-09-27 final run). (a) Share of frames that are "
           f"tracked, and of lost frames split by the maximum over the four cameras of the number of LEDs expected to be "
           f"visible from the mocap pose (pipeline visibility model, including self- and cross-occlusion). Pooled, "
           f"{pct(P['lost_lt5'], P['lost'])}\\,\\% of the lost frames have fewer than five expected-visible LEDs in every camera "
           f"(a physical limit), {pct(P['lost_5to7'], P['lost'])}\\,\\% have five to seven and {pct(P['lost_ge8'], P['lost'])}\\,\\% "
           f"at least eight. (b) Cumulative distribution of that maximum for tracked and lost frames; the dotted line marks "
           f"the minimum inlier count.")
    snip = ("\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=\\textwidth]{figures/coverage_decomposition.pdf}\n"
            f"  \\caption{{{cap}}}%\n  \\label{{fig:coverage-decomposition}}\n\\end{{figure}}\n")
    (FIG / "snippets").mkdir(exist_ok=True)
    (FIG / "snippets" / "coverage_decomposition.tex").write_text(snip)
    agg["sanity"] = sanity
    json.dump(agg, open(FIG / "coverage_decomposition.json", "w"), indent=1)


def main():
    global CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=str(REPO), help="dir containing src/ (use a `git archive HEAD` snapshot)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-compute", action="store_true")
    ap.add_argument("--skip-image-check", action="store_true")
    a = ap.parse_args()
    CODE = Path(a.code)
    em = _imports()
    os.chdir(REPO)
    if not a.skip_compute:
        tasks = [(r, c) for r in RECS for c in CTRLS]
        with ProcessPoolExecutor(a.workers) as ex:
            for rec, ctrl, n in ex.map(process_pair, tasks):
                print(f"computed {rec}/{ctrl}: {n} GT frames", flush=True)
    agg = aggregate()
    sanity = {"inlier_consistency": sanity_inliers(agg)}
    print("inlier sanity:", sanity["inlier_consistency"], flush=True)
    if not a.skip_image_check:
        sanity["image_check"] = sanity_images(em)
    report(agg, sanity)
    P = agg["pooled"]["primary"]
    print("POOLED primary:", P)
    print("POOLED lenient:", agg["pooled"]["lenient"])


if __name__ == "__main__":
    main()
