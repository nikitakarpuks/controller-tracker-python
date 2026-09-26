#!/usr/bin/env python3
"""Detectability metric: how many frames could a controller pose theoretically be detected, and how many were.

Stage 1 (per recording, ~1-3 min): for every mocap-valid (frame, controller) compute how many LEDs are theoretically
visible per camera from the MOCAP pose (real `_visible_mask` + cross-controller occlusion from the other controller's
MOCAP pose) at three emission-cone angles (60 strict / 75 nominal / 90 loose) -> <rec>/<rec>_detectability.csv.
Stage 2: join with the tracker outputs (<rec>_pose.csv, <rec>_vision_pose.csv), classify the displayed pose, score it
against mocap (bridge composed) and print the per-recording outcome grid + metrics.

Usage: python3 detectability.py [--dir visualization/evaluate_2026-09-26] [--rec static_hard ...] [--skip-stage1]

Definitions (frozen; see the design discussion):
  frame set   camera timestamps with valid mocap for that controller (mocap_gt csv); tracker outputs left-joined,
              a missing row = NONE. Frame weight = dt to next frame, capped at 50 ms (stalls do not dominate).
  tier        by the BEST SINGLE CAMERA visible-LED count at the nominal 75 deg cone:
              D2 >= 6 (detectable), D1 3-5 (marginal), D0 <= 2 (undetectable). 60/90 deg shown as variants.
  state       none (no row / NaN pose) | coast (finite pose, reproj nan) | constrained (primary inliers <= 3)
              | strong (pooled inliers >= 8) | weak (accepted, pooled < 8). pooled = vision_pose n_inliers.
  accuracy    good <= 20 mm & <= 5 deg, wrong > 60 mm or > 15 deg, else usable. Constrained poses: position only.
  detection   vision-accepted (strong/weak) pose that is 'good'; constrained counts separately (position only);
              coast is never a detection.
"""
import argparse, json, sys, os
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

RECS = ["static_dark", "static_easy", "static_medium", "static_hard", "walk_dark", "walk_easy", "walk_medium", "walk_hard"]
CTRLS = ("left_controller", "right_controller")
DT_CAP = 0.05
GOOD_MM, GOOD_DEG, WRONG_MM, WRONG_DEG = 20.0, 5.0, 60.0, 15.0
TIER_D2, TIER_D1 = 6, 3
CONES = (60.0, 75.0, 90.0)


def stage1(rec: str, base: Path):
    from src.load_config import load_yaml_config, load_json_config
    from src.camera import Camera
    from src.controller import ControllerModel, create_leds_from_config, mirror_primitives
    from src.geometry import _compute_geometry
    from src._visibility import _visible_mask, _cross_occluded_mask
    from src.mocap_data import load_mocap_bridge
    import evaluate_mocap as em

    d = base / rec
    cfg = load_yaml_config(str(d / f"_config_{rec}.yml"))
    calib = load_json_config(cfg["cameras"]["intrinsics_path"])
    conv = cfg["cameras"].get("extrinsics_convention", "T_imu_cam")
    cams = {i: Camera(calib, camera_idx=i, extrinsics_convention=conv) for i in range(4)}
    right = cfg["controllers"]["right_controller"]
    models, geoms, bridges = {}, {}, {}
    for k in CTRLS:
        cc = cfg["controllers"][k]
        m = ControllerModel(create_leds_from_config(load_json_config(cc["config_path"])), k)
        geo = dict(cfg.get("geometry", {}))
        if "handle_primitives" in cc:
            geo["handle_primitives"] = cc["handle_primitives"]
        elif k == "left_controller":
            geo["handle_primitives"] = mirror_primitives(right["handle_primitives"])
        models[k] = m
        geoms[k] = _compute_geometry(m.positions.astype("float32"), m.normals.astype("float32"), geo)
        bridges[k] = load_mocap_bridge(cc["mocap_bridge_path"])
    gt = {k: em.load_mocap_gt_csv(d / f"{rec}_mocap_gt" / f"{k}_mocap_gt.csv") for k in CTRLS}

    def counts(k, ko, Tw, Two, cone):
        out = []
        for cid, cam in cams.items():
            Tci = cam.T_world_cam.inverse().compose(Tw)
            R, t = Tci.R.astype(np.float32), Tci.t.astype(np.float32)
            s = _visible_mask(R, t, models[k].positions, models[k].normals, geoms[k], cam_K=cam.camera_matrix,
                              cam_dc=cam.dist_coeffs, cam_w=cam.width, cam_h=cam.height, cam_rpmax=cam.rpmax,
                              cam_is_fisheye=cam.is_fisheye, facing_threshold_deg=cone, image_margin_px=0.0)
            if Two is not None:
                To = cam.T_world_cam.inverse().compose(Two)
                fo = float(max(cam.camera_matrix[0, 0], cam.camera_matrix[1, 1]))
                occ = _cross_occluded_mask(R, t, models[k].positions, To.R.astype(np.float32),
                                           To.t.astype(np.float32), geoms[ko], 0.18, 0.18, fo, 20.0)
                s = s.copy(); s[occ] = 0
            out.append(int((s >= 1.0).sum()))
        return out

    rows = []
    for k in CTRLS:
        ko = CTRLS[1] if k == CTRLS[0] else CTRLS[0]
        for ts in sorted(gt[k]):
            Tw = gt[k][ts].compose(bridges[k].inverse())
            Tog = gt[ko].get(ts)
            Two = Tog.compose(bridges[ko].inverse()) if Tog is not None else None
            r = dict(ts=ts, ctrl=k)
            for cone in CONES:
                c = counts(k, ko, Tw, Two, cone)
                r[f"best{int(cone)}"] = max(c); r[f"tot{int(cone)}"] = sum(c)
                if cone == 75.0:
                    r["c75"] = ";".join(map(str, c))
            r["best75_noocc"] = max(counts(k, ko, Tw, None, 75.0)) if Two is not None else r["best75"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(d / f"{rec}_detectability.csv", index=False)
    return len(df)


def _load(rec, base):
    import evaluate_mocap as em
    from src.load_config import load_yaml_config
    from src.mocap_data import load_mocap_bridge
    d = base / rec
    cfg = load_yaml_config(str(d / f"_config_{rec}.yml"))
    det = pd.read_csv(d / f"{rec}_detectability.csv")
    pdf = pd.read_csv(d / f"{rec}_pose.csv"); vdf = pd.read_csv(d / f"{rec}_vision_pose.csv")
    pose = em.load_pose_csv(d / f"{rec}_pose.csv")
    prow = {(int(r.timestamp_ns), r.ctrl_name): (float(r.reproj_err_px) if str(r.reproj_err_px) != "nan" else np.nan,
                                                 int(r.inlier_count)) for r in pdf.itertuples()}
    vin = {(int(r.timestamp_ns), r.ctrl_name): int(r.n_inliers) for r in vdf.itertuples()}
    out = []
    for k in CTRLS:
        gt = em.load_mocap_gt_csv(d / f"{rec}_mocap_gt" / f"{k}_mocap_gt.csv")
        br = load_mocap_bridge(cfg["controllers"][k]["mocap_bridge_path"])
        s = det[det.ctrl == k].sort_values("ts").reset_index(drop=True)
        ts = s.ts.values
        w = np.minimum(np.append(np.diff(ts) / 1e9, 0.022), DT_CAP)
        for i, r in enumerate(s.itertuples()):
            t = int(r.ts); P = pose.get(k, {}).get(t)
            pr = prow.get((t, k))
            if P is None or pr is None:
                state, pe, re_ = "none", np.nan, np.nan
            else:
                res = P.compose(br).inverse().compose(gt[t])
                pe, re_ = float(np.linalg.norm(res.t) * 1e3), float(em.rotation_angle_deg(res.R))
                reproj, prim = pr
                if np.isnan(reproj) or prim == 0:
                    state = "coast"
                elif prim <= 3:
                    state = "constrained"
                else:
                    state = "strong" if vin.get((t, k), prim) >= 8 else "weak"
            out.append(dict(rec=rec, ctrl=k, ts=t, w=w[i], best60=r.best60, best75=r.best75, best90=r.best90,
                            tot75=r.tot75, best75_noocc=r.best75_noocc, state=state, pos_mm=pe, rot_deg=re_))
    d = pd.DataFrame(out)
    tier = lambda b: np.where(b >= TIER_D2, "D2", np.where(b >= TIER_D1, "D1", "D0"))
    d["tier"] = tier(d.best75); d["tier60"] = tier(d.best60); d["tier90"] = tier(d.best90)
    posonly = d.state == "constrained"
    good = (d.pos_mm <= GOOD_MM) & ((d.rot_deg <= GOOD_DEG) | posonly)
    wrong = (d.pos_mm > WRONG_MM) | ((d.rot_deg > WRONG_DEG) & ~posonly)
    d["acc"] = np.where(d.state == "none", "-", np.where(good, "good", np.where(wrong, "wrong", "usable")))
    d["det"] = d.state.isin(["strong", "weak"]) & (d.acc == "good")
    return d


def _pct(a, b):
    return 100.0 * a / b if b > 0 else float("nan")


def summarize(d: pd.DataFrame, rng=np.random.default_rng(0)):
    W = d.w.sum(); out = {}
    out["frames"] = {c: int((d.ctrl == c).sum()) for c in CTRLS}
    out["hours"] = round(W / 3600, 4)
    out["tier_share_pct"] = {t: round(_pct(d.w[d.tier == t].sum(), W), 1) for t in ("D0", "D1", "D2")}
    out["D2_share_pct_60_75_90"] = [round(_pct(d.w[d[c] == "D2"].sum(), W), 1) for c in ("tier60", "tier", "tier90")]
    out["uncertain_share_pct(D2 differs 60 vs 90)"] = round(_pct(d.w[(d.tier60 == "D2") != (d.tier90 == "D2")].sum(), W), 1)
    out["cross_occlusion_changed_tier_pct"] = round(_pct(d.w[(d.best75 >= TIER_D2) != (d.best75_noocc >= TIER_D2)].sum(), W), 2)
    grid = {}
    for t in ("D2", "D1", "D0"):
        s = d[d.tier == t]; g = {}
        for st in ("strong", "weak", "constrained", "coast", "none"):
            q = s[s.state == st]
            g[st] = dict(pct_of_tier=round(_pct(q.w.sum(), s.w.sum()), 1),
                         good=round(_pct(q.w[q.acc == "good"].sum(), q.w.sum()), 1) if st != "none" and len(q) else None,
                         wrong=round(_pct(q.w[q.acc == "wrong"].sum(), q.w.sum()), 1) if st != "none" and len(q) else None,
                         n=len(q))
        grid[t] = g
    out["grid_time_weighted"] = grid
    D2 = d[d.tier == "D2"]; D1 = d[d.tier == "D1"]
    out["recall_D2_pct"] = round(_pct(D2.w[D2.det].sum(), D2.w.sum()), 1)
    out["recall_D2_any_accuracy_pct"] = round(_pct(D2.w[D2.state.isin(["strong", "weak"])].sum(), D2.w.sum()), 1)
    out["recall_D2_incl_constrained_pos_pct"] = round(_pct(D2.w[D2.det | ((D2.state == "constrained") & (D2.acc == "good"))].sum(), D2.w.sum()), 1)
    out["recall_D1_pct"] = round(_pct(D1.w[D1.det].sum(), D1.w.sum()), 1)
    for thr in (5, 7):
        s = d[d.best75 >= thr]
        out[f"sens_recall_best>={thr}_pct"] = round(_pct(s.w[s.det].sum(), s.w.sum()), 1)
    shown = d[d.state != "none"]
    wr = shown[shown.acc == "wrong"]
    out["displayed_wrong_pct_by_state"] = {st: round(_pct(len(wr[wr.state == st]), int((shown.state == st).sum())), 2)
                                           for st in ("strong", "weak", "constrained", "coast")}
    out["wrong_poses_per_hour_of_D2_time"] = round(len(wr[wr.tier == "D2"]) / max(D2.w.sum() / 3600, 1e-9), 1)
    out["displayed_while_D0_pct"] = round(_pct(d.w[(d.tier == "D0") & (d.state != "none")].sum(), d.w[d.tier == "D0"].sum()), 2)
    out["availability_good_pct"] = dict(
        vision_only=round(_pct(d.w[d.det].sum(), W), 1),
        vision_plus_constrained=round(_pct(d.w[d.det | ((d.state == "constrained") & (d.acc == "good"))].sum(), W), 1),
        with_coast=round(_pct(d.w[(d.det | (d.state.isin(["coast", "constrained"]) & (d.acc == "good")))].sum(), W), 1))
    out["accurate_vision_labelled_D0_or_D1_pct"] = round(_pct(d.w[d.det & (d.tier != "D2")].sum(), d.w[d.det].sum()), 1)
    # D2 miss episodes (contiguous, per controller): D2 frame without a detection; min length 3 frames
    ep = []; short = 0
    for c in CTRLS:
        s = d[d.ctrl == c].sort_values("ts"); miss = ((s.tier == "D2") & ~s.det).values
        tsv = s.ts.values; i = 0
        while i < len(s):
            if miss[i]:
                j = i
                while j + 1 < len(s) and miss[j + 1]:
                    j += 1
                n = j - i + 1
                if n >= 3:
                    ep.append((n, (tsv[j] - tsv[i]) / 1e9 + 0.022))
                else:
                    short += n
                i = j + 1
            else:
                i += 1
    out["D2_miss_episodes(>=3 frames)"] = dict(count=len(ep), total_s=round(sum(e[1] for e in ep), 1),
                                              median_s=round(float(np.median([e[1] for e in ep])), 2) if ep else 0,
                                              max_s=round(max([e[1] for e in ep]), 2) if ep else 0,
                                              frames_in_shorter_runs=short)
    # block bootstrap (1 s blocks, both controllers jointly) on D2 recall and good availability
    blk = ((d.ts - d.ts.min()) // 1_000_000_000).astype(int)
    groups = {b: g for b, g in d.groupby(blk)}
    keys = np.array(list(groups)); stats = []
    agg = {b: (g.w[(g.tier == "D2") & g.det].sum(), g.w[g.tier == "D2"].sum(), g.w[g.det].sum(), g.w.sum()) for b, g in groups.items()}
    for _ in range(300):
        pick = rng.choice(keys, len(keys)); a = np.array([agg[b] for b in pick]).sum(axis=0)
        stats.append((100 * a[0] / max(a[1], 1e-9), 100 * a[2] / a[3]))
    st = np.array(stats)
    out["recall_D2_95CI"] = [round(float(np.percentile(st[:, 0], 2.5)), 1), round(float(np.percentile(st[:, 0], 97.5)), 1)]
    out["vision_avail_95CI"] = [round(float(np.percentile(st[:, 1], 2.5)), 1), round(float(np.percentile(st[:, 1], 97.5)), 1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="visualization/evaluate_2026-09-26")
    ap.add_argument("--rec", nargs="*", default=RECS)
    ap.add_argument("--skip-stage1", action="store_true")
    a = ap.parse_args(); base = REPO / a.dir
    if not a.skip_stage1:
        from multiprocessing import Pool
        with Pool(min(4, len(a.rec))) as p:
            for rec, n in zip(a.rec, p.starmap(stage1, [(r, base) for r in a.rec])):
                print(f"stage1 {rec}: {n} rows", flush=True)
    allrows = []
    for rec in a.rec:
        d = _load(rec, base); allrows.append(d)
        s = summarize(d)
        (base / rec / f"{rec}_detectability_summary.json").write_text(json.dumps(s, indent=2))
        d.to_csv(base / rec / f"{rec}_detectability_frames.csv", index=False)
        print(f"\n=== {rec}  ({s['hours']} h, frames {s['frames']}) ===")
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
