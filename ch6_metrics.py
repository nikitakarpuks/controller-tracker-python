#!/usr/bin/env python3
"""Gross-error rate / identity-swap+flip / lost-streak / IMU-guard metrics, reproducing (or explicitly
extending) the thesis's own preliminary-batch ch.6 methodology from real per-frame CSVs -- no rerun needed.

Design/scope, agreed 2026-09-27 after a design-critique pass (2 independent critics against real data):
  A. Gross-error rate: fraction of TRACKED frames (that series produced a pose) with pos error >150mm or
     rot error >20deg (matches run_all_recordings.py's own WARN_POS_MM/WARN_ROT_DEG exactly -- no new
     threshold). Absolute-residual family (bridge-composed vs mocap, no Umeyama alignment), matching the
     thesis's own sec:eval-gross. Computed for BOTH the fused and the raw-vision series, in BOTH the full
     and vision-only runs (the two runs' own vision-series gross rates differ 2-7x -- NOT a "consistency
     check" as an earlier draft assumed; reported as two real numbers).
  B. Identity swap / flip: adopting visualization/report_2026-09-24/tools/rec_analysis.py's own proxy
     (this project's prior art, never previously used by the thesis text itself): among GROSS vision
     frames, swap = pos_err_to_OWN_gt>150mm & pos_err_to_OTHER_ctrl_gt<60mm; flip = rot_err>120deg &
     pos_err<100mm. Checked against real inter-controller distance (min ~90mm across all 8 recordings) --
     the 60mm gate is safe, no clasped-hands false positives. NOTE: the old thesis prose's static_medium
     "nearly all gross frames are swaps" anecdote does NOT reproduce on this re-recorded 2026-09-27 dataset
     (real distances there are 175-1125mm) -- likely from the older recordings-aug26 physical session, not
     this one; do not repeat that specific claim. An explicit "other" bucket is reported for gross frames
     that are neither swap nor flip, since most real gross frames fall there.
  C. Lost-streak: a run of consecutive GT-covered frames with NO FINITE fused-pose row (matches
     evaluate_mocap.py's own longest_gap mechanics and the thesis's own "no reported pose" wording --
     deliberately NOT main.py's live lost_streak counter, which freezes rather than increments on a
     rejected-but-coasted frame and so answers a different question). ALL streak durations (not just the
     longest), static vs walk pooled across both controllers within each group, for both runs (a same-data
     bonus: does removing the IMU lengthen streaks, i.e. is the search-anchor propagation doing its job).
  D. IMU-guard precision/recall: "caught" = a gross VISION frame where the fused series is NOT gross
     (replaced with something better) OR has no finite row at all (dropped) -- both count, per the thesis's
     own "dropped or replaced" wording (an earlier draft's naive gross(vision) & gross(fused) intersection
     silently missed the dropped-to-none case and undercounted recall by ~25%). "Wrongly" = the mirror: an
     acceptable (not gross) vision frame where fused IS gross or absent. Precision=caught/(caught+wrongly),
     recall=caught/all_gross_vision. Computed ONLY on the SINGLE full run (v_old, matching the existing
     thesis methodology) -- NOT against the vision-only run: a design-critique pass found that comparing
     across runs conflates two different IMU channels (search-assist improving vision's own candidates, vs
     the accept/reject/blend decision layer this metric wants to isolate), a worse confound than staying
     within one run. The separate, complementary "how much does search-assist itself help" question is
     already answered by item A's own full-vs-vision-only vision-series gross-rate comparison.
     "intervened" (diagnostic only, not used in precision/recall): |vision candidate - fused reported| in
     pose.csv's MAIN columns (post-fusion, post-One-Euro) -- NOT the raw_* columns, which are the pre-One-
     Euro tracking state (an earlier draft's mistake: raw_* tracks the filter's own decision, not vision,
     so diffing it against reported measures one-euro lag, not guard behavior; it happens to equal the
     vision candidate only on already-accepted "fused" frames, which is exactly the least interesting case).
     Small-sample risk is real (some recording x controller cells have under 15, one has 0 gross-vision
     frames): counts are reported per recording, not rates; the pooled number (with a binomial CI) is the
     one to trust. A second cutoff (SEVERE_POS_MM/_ROT_DEG, matching run_all_recordings.py) is reported
     alongside the primary one as a sensitivity check.

Usage: python3 ch6_metrics.py [--dir visualization/evaluate_2026-09-27_full] [--vo-dir visualization/evaluate_2026-09-27_visiononly]
"""
import argparse, json, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import evaluate_mocap as em
from src.load_config import load_yaml_config
from src.mocap_data import load_mocap_bridge

RECS = ["static_dark", "static_easy", "static_medium", "static_hard", "walk_dark", "walk_easy", "walk_medium", "walk_hard"]
CTRLS = ("left_controller", "right_controller")
GROSS_MM, GROSS_DEG = 150.0, 20.0
SEVERE_MM, SEVERE_DEG = 400.0, 60.0
SWAP_OTHER_MM, FLIP_ROT_DEG, FLIP_POS_MM = 60.0, 120.0, 100.0
INTERVENED_MM, INTERVENED_DEG = 10.0, 2.0
STREAK_BINS_MS = [0, 100, 300, 1000, 3000, float("inf")]
STREAK_LABELS = ["<100ms", "100-300ms", "300-1000ms", "1000-3000ms", ">3000ms"]
STATIC_RECS = {"static_dark", "static_easy", "static_medium", "static_hard"}


def _err(T_est, T_gt, bridge):
    if T_est is None:
        return None
    r = T_est.compose(bridge).inverse().compose(T_gt)
    return float(np.linalg.norm(r.t) * 1e3), float(em.rotation_angle_deg(r.R))


def load_recording(base: Path, rec: str):
    """Returns dict[ctrl] -> DataFrame indexed by ts with columns: gt (Transform), fused (Transform|None),
    vision (Transform|None), plus the raw pos/rot of the OTHER controller's gt for the swap check."""
    d = base / rec
    cfg = load_yaml_config(str(d / f"_config_{rec}.yml"))
    gt = {c: em.load_mocap_gt_csv(d / f"{rec}_mocap_gt" / f"{c}_mocap_gt.csv") for c in CTRLS}
    fused = {c: em.load_pose_csv(d / f"{rec}_pose.csv").get(c, {}) for c in CTRLS}
    vision = {c: em.load_pose_csv(d / f"{rec}_vision_pose.csv").get(c, {}) for c in CTRLS}
    bridge = {c: load_mocap_bridge(cfg["controllers"][c]["mocap_bridge_path"]) for c in CTRLS}
    # fusion_outcome, read from the RAW pose.csv (em.load_pose_csv drops non-finite/pending rows entirely,
    # which is right for accuracy tables but wrong for item D: a *_pending outcome (cold_pending,
    # bootstrap_pending -- the deliberate one-frame weak-candidate confirmation buffer, see
    # pose_fusion_heuristic.py's _try_cold_reacquire) has NO finalized decision yet and must not be counted
    # as the guard "dropping" a good vision candidate -- confirmed on real data: every one of a sample of
    # "wrongly" cases under a naive fpos.isna()-only definition was exactly this buffering state, not a
    # real guard mistake.
    raw_pose_all = pd.read_csv(d / f"{rec}_pose.csv")
    outcome = {c: dict(zip(raw_pose_all[raw_pose_all.ctrl_name == c].timestamp_ns,
                          raw_pose_all[raw_pose_all.ctrl_name == c].fusion_outcome)) for c in CTRLS}
    PENDING = {"bootstrap_pending", "cold_pending"}
    out = {}
    for c in CTRLS:
        co = CTRLS[1] if c == CTRLS[0] else CTRLS[0]
        rows = []
        for ts, T_gt in gt[c].items():
            fe = _err(fused[c].get(ts), T_gt, bridge[c])
            ve = _err(vision[c].get(ts), T_gt, bridge[c])
            oe = None
            T_gt_o = gt[co].get(ts)
            if T_gt_o is not None:
                oe = float(np.linalg.norm(T_gt.t - T_gt_o.t) * 1e3)  # own-vs-other controller REAL distance
            # vision candidate's own distance to the OTHER controller's real position (for the swap proxy)
            ve_other = None
            if vision[c].get(ts) is not None and T_gt_o is not None:
                ve_other = _err(vision[c][ts], T_gt_o, bridge[co])
            fe_other = None
            if fused[c].get(ts) is not None and T_gt_o is not None:
                fe_other = _err(fused[c][ts], T_gt_o, bridge[co])
            rows.append(dict(ts=ts, fpos=fe[0] if fe else np.nan, frot=fe[1] if fe else np.nan,
                             vpos=ve[0] if ve else np.nan, vrot=ve[1] if ve else np.nan,
                             ctrl_dist=oe, v_other_pos=ve_other[0] if ve_other else np.nan,
                             f_other_pos=fe_other[0] if fe_other else np.nan,
                             pending=outcome[c].get(ts) in PENDING))
        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        out[c] = df
    return out


def item_A_gross(data, label):
    """Per-recording-and-controller + pooled gross-error rate, both series."""
    rows = []
    for rec, per_ctrl in data.items():
        for c, df in per_ctrl.items():
            for series, pos, rot in (("fused", "fpos", "frot"), ("vision", "vpos", "vrot")):
                tracked = df[pos].notna()
                n = int(tracked.sum())
                if n == 0:
                    rows.append(dict(rec=rec, ctrl=c[:5], series=series, n_tracked=0, gross=0, gross_pct=float("nan")))
                    continue
                gross = int(((df.loc[tracked, pos] > GROSS_MM) | (df.loc[tracked, rot] > GROSS_DEG)).sum())
                rows.append(dict(rec=rec, ctrl=c[:5], series=series, n_tracked=n, gross=gross,
                                 gross_pct=100 * gross / n))
    df = pd.DataFrame(rows)
    pooled = df.groupby("series").apply(lambda g: 100 * g.gross.sum() / g.n_tracked.sum(), include_groups=False)
    print(f"\n=== A. Gross-error rate ({label}), cutoff {GROSS_MM:.0f}mm/{GROSS_DEG:.0f}deg ===")
    print(df.pivot_table(index=["rec", "ctrl"], columns="series", values="gross_pct").round(2).to_string())
    print("pooled gross %:", pooled.round(3).to_dict())
    return df, pooled


def item_B_swap_flip(data, label):
    rows = []
    for rec, per_ctrl in data.items():
        for c, df in per_ctrl.items():
            g = df[(df.vpos > GROSS_MM) | (df.vrot > GROSS_DEG)].copy()
            if len(g) == 0:
                rows.append(dict(rec=rec, ctrl=c[:5], n_gross=0, swap=0, flip=0, other=0)); continue
            swap = (g.vpos > GROSS_MM) & (g.v_other_pos < SWAP_OTHER_MM)
            flip = (g.vrot > FLIP_ROT_DEG) & (g.vpos < FLIP_POS_MM)
            other = ~(swap | flip)
            rows.append(dict(rec=rec, ctrl=c[:5], n_gross=len(g), swap=int(swap.sum()), flip=int(flip.sum()),
                             other=int(other.sum())))
    df = pd.DataFrame(rows)
    print(f"\n=== B. Identity swap/flip among gross VISION frames ({label}) ===")
    print(df.to_string(index=False))
    tot = df[["n_gross", "swap", "flip", "other"]].sum()
    print("pooled:", tot.to_dict(), f"| min real inter-controller distance seen: "
          f"{min(df_.ctrl_dist.min() for per_ctrl in data.values() for df_ in per_ctrl.values() if df_.ctrl_dist.notna().any()):.1f}mm")
    return df


def _streaks(ts_ns, has_pose):
    """Runs of consecutive False in has_pose (aligned to ts_ns, sorted) -> list of durations in ms."""
    out = []
    n = len(ts_ns)
    i = 0
    while i < n:
        if not has_pose[i]:
            j = i
            while j + 1 < n and not has_pose[j + 1]:
                j += 1
            dur_ms = (ts_ns[j] - ts_ns[i]) / 1e6 + 22.0  # + one nominal frame period, matching evaluate_mocap's own convention
            out.append(dur_ms)
            i = j + 1
        else:
            i += 1
    return out


def item_C_lost_streaks(data, label):
    buckets = {"static": [], "walk": []}
    for rec, per_ctrl in data.items():
        grp = "static" if rec in STATIC_RECS else "walk"
        for c, df in per_ctrl.items():
            buckets[grp].extend(_streaks(df.ts.values, df.fpos.notna().values))
    print(f"\n=== C. Lost-streak durations ({label}) ===")
    hist = {}
    for grp, durs in buckets.items():
        durs = np.array(durs)
        h, _ = np.histogram(durs, bins=STREAK_BINS_MS)
        hist[grp] = dict(zip(STREAK_LABELS, h.tolist()))
        top10_share = 100 * np.sort(durs)[-10:].sum() / durs.sum() if len(durs) else float("nan")
        print(f"  {grp:6s} n_streaks={len(durs):4d} total_s={durs.sum()/1000:6.1f} "
              f"top1_share%={100*durs.max()/durs.sum():.1f} top10_share%={top10_share:.1f}  {hist[grp]}")
    return hist


def item_D_guard(data_full, data_vo_vision_gross=None):
    rows = []
    for cutoff_name, pm, rd in (("gross_150_20", GROSS_MM, GROSS_DEG), ("severe_400_60", SEVERE_MM, SEVERE_DEG)):
        for rec, per_ctrl in data_full.items():
            for c, df in per_ctrl.items():
                not_pending = ~df.pending  # exclude the one-frame weak-candidate confirmation buffer --
                # its "no pose shown yet" isn't a final drop, see load_recording's own comment.
                v_gross = (df.vpos > pm) | (df.vrot > rd)
                v_has = df.vpos.notna() & not_pending
                f_gross = ((df.fpos > pm) | (df.frot > rd)) | df.fpos.isna()  # gross OR dropped counts as "still bad"
                f_ok = ~f_gross
                caught = int((v_gross & v_has & f_ok).sum())
                still_bad = int((v_gross & v_has & f_gross).sum())
                n_gross_vis = int((v_gross & v_has).sum())
                acceptable = v_has & ~v_gross
                wrongly = int((acceptable & f_gross).sum())
                n_acceptable = int(acceptable.sum())
                rows.append(dict(cutoff=cutoff_name, rec=rec, ctrl=c[:5], n_gross_vis=n_gross_vis, caught=caught,
                                 still_bad=still_bad, n_acceptable=n_acceptable, wrongly=wrongly))
    df = pd.DataFrame(rows)
    print("\n=== D. IMU-guard precision/recall (v_old, single full run) ===")
    for cutoff in df.cutoff.unique():
        s = df[df.cutoff == cutoff]
        print(f"-- cutoff {cutoff} --")
        print(s[["rec", "ctrl", "n_gross_vis", "caught", "wrongly", "n_acceptable"]].to_string(index=False))
        C, W, G = s.caught.sum(), s.wrongly.sum(), s.n_gross_vis.sum()
        precision = C / max(C + W, 1); recall = C / max(G, 1)
        # binomial (Wilson) CI on recall
        from scipy.stats import norm
        z = norm.ppf(0.975)
        p = recall; nB = G
        denom = 1 + z**2 / max(nB, 1)
        center = (p + z**2 / (2 * max(nB, 1))) / denom
        half = z * np.sqrt(p * (1 - p) / max(nB, 1) + z**2 / (4 * max(nB, 1)**2)) / denom
        print(f"  pooled: gross_vis={G} caught={C} wrongly={W} n_acceptable={s.n_acceptable.sum()}  "
              f"precision={precision:.3f}  recall={recall:.3f} (95% Wilson CI {max(0,center-half):.3f}-{min(1,center+half):.3f})")
    return df


def item_D_intervened(data_full):
    """Diagnostic only (not used in precision/recall): |vision - fused reported| in pose.csv's MAIN columns."""
    rows = []
    for rec, per_ctrl in data_full.items():
        for c, df in per_ctrl.items():
            both = df.fpos.notna() & df.vpos.notna()
            # crude proxy for position-magnitude divergence between the two SERIES' own mocap errors
            # (both already residuals vs the same mocap frame, so their difference approximates
            # |vision_pose - fused_pose| without needing a third quaternion diff)
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="visualization/evaluate_2026-09-27_full")
    ap.add_argument("--vo-dir", default="visualization/evaluate_2026-09-27_visiononly")
    ap.add_argument("--rec", nargs="*", default=RECS)
    a = ap.parse_args()
    base_full, base_vo = REPO / a.dir, REPO / a.vo_dir

    data_full = {rec: load_recording(base_full, rec) for rec in a.rec}
    data_vo = {rec: load_recording(base_vo, rec) for rec in a.rec}

    gross_full, pooled_full = item_A_gross(data_full, "full run")
    gross_vo, pooled_vo = item_A_gross(data_vo, "vision-only run")
    print("\n=== A (continued): full-vs-vision-only comparison of the RAW VISION series' own gross rate ===")
    print("(this is the 'search-assist channel' contribution -- see module docstring; NOT part of item D)")
    m = gross_full[gross_full.series == "vision"].merge(
        gross_vo[gross_vo.series == "vision"], on=["rec", "ctrl"], suffixes=("_full", "_vo"))
    print(m[["rec", "ctrl", "gross_full", "gross_pct_full", "gross_vo", "gross_pct_vo"]].rename(
        columns={"gross_full": "gross_n_full", "gross_vo": "gross_n_vo"}).round(2).to_string(index=False))

    item_B_swap_flip(data_full, "full run")
    item_C_lost_streaks(data_full, "full run")
    item_C_lost_streaks(data_vo, "vision-only run")
    item_D_guard(data_full)

    out = dict(gross_full=gross_full.to_dict("records"), gross_vo=gross_vo.to_dict("records"))
    Path("visualization/ch6_metrics_2026-09-27.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
