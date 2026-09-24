"""Step 13: do the fitted constants TRANSFER across recordings (power-cycle / session dependence)?
 Gyro: K (+b) fit on SOURCE recording (whole recording, vision steps, gyro regressor) applied to TARGET recording (all anchors);
 compared with the target's own K fit (first 60 s) and zero.  Accel: b / (b,K_a) fit on SOURCE (mocap windows) applied to TARGET.
 Source = static_dark, target = walk_medium (and vice versa)."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected, gap_err_gyro
from step6_gyro_estimators import block_bootstrap_ci
from step10b_accel_K import fit_bK, predict

SRC, TGT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("static_dark", "walk_medium")
GAPS_G = [0.05, 0.25, 0.5, 1.0, 2.0]; GAPS_A = [0.25, 0.5, 1.0]

def gyro_fit(run, c, t_hi_s=None):
    prep = prepare(run, c, ego="mocap"); steps, _ = make_steps(run, c, prep)
    t0 = prep["ts"][prep["ok"]][0]
    tr = [s for s in steps if t_hi_s is None or s["t"] < t0 + int(t_hi_s * 1e9)]
    beta, _, _ = fit_body(tr, key="omega_g")
    return beta[:3], beta[3:].reshape(3, 3), prep

def accel_fit(run, c, prep, gyro_c, t_hi_s=None, W=5):
    trk_m = track_mocap(run, c); AW = AccelWindows(run, c, trk_m, gyro=gyro_c)
    t0 = prep["ts"][prep["ok"]][0]; t_end = prep["ts"][-1]
    hi = t_end if t_hi_s is None else t0 + int(t_hi_s * 1e9)
    b, _, _ = fit_bK(AW, t0, hi, W, True, False); bb, KK, _ = fit_bK(AW, t0, hi, W, True, True)
    return b, bb, KK

runs = {SRC: Run(SRC), TGT: Run(TGT)}
for c in CTRLS:
    print(f"\n================ {c}:  fit on {SRC}  ->  test on {TGT}")
    b_s, K_s, prep_s = gyro_fit(runs[SRC], c)
    b_t60, K_t60, prep_t = gyro_fit(runs[TGT], c, 60)
    b_tall, K_tall, _ = gyro_fit(runs[TGT], c)
    print(f"   gyro K(diag) src {np.round(np.diag(K_s),4)} | tgt(first 60s) {np.round(np.diag(K_t60),4)} | tgt(all) {np.round(np.diag(K_tall),4)}")
    print(f"   gyro K corr(src, tgt-all) over 9 entries: {np.corrcoef(K_s.ravel(), K_tall.ravel())[0,1]:.3f}   max|dK| {np.abs(K_s-K_tall).max():.4f}   b src {np.round(b_s,4)} tgt {np.round(b_tall,4)}")
    d = runs[TGT].ctrl[c]
    variants = {"zero": d["gyro"], f"K from {SRC}": corrected(d["gyro"], np.zeros(3), K_s), f"b+K from {SRC}": corrected(d["gyro"], b_s, K_s),
                "K own first-60s": corrected(d["gyro"], np.zeros(3), K_t60), "b+K own first-60s": corrected(d["gyro"], b_t60, K_t60)}
    t0 = prep_t["ts"][prep_t["ok"]][0]; t_hi = prep_t["ts"][-1]
    print(f"   TARGET {TGT}: gyro rotation-prediction error median/p90 (deg), anchors after t0+60 s      gap(s): " + "".join(f"{g:>14.2f}" for g in GAPS_G))
    for lab, gy in variants.items():
        cells = []
        for g in GAPS_G:
            e = gap_err_gyro(runs[TGT], c, prep_t, gy, g, t0 + int(60e9), t_hi)
            cells.append(f"{np.median(e):.2f}/{np.percentile(e,90):.2f}" if len(e) > 10 else "n/a")
        print(f"   {lab:26s}" + "".join(f"{x:>14s}" for x in cells))
    # ---- accel
    gyro_src = corrected(runs[SRC].ctrl[c]["gyro"], np.zeros(3), K_s)
    b_a_s, bb_s, KK_s = accel_fit(runs[SRC], c, prep_s, gyro_src)
    gyro_tgt = corrected(d["gyro"], np.zeros(3), K_s)
    b_a_t, bb_t, KK_t = accel_fit(runs[TGT], c, prep_t, gyro_tgt, t_hi_s=60)
    b_a_tall, _, _ = accel_fit(runs[TGT], c, prep_t, gyro_tgt)
    print(f"   accel b: {SRC}(all) {np.round(b_a_s,3)} | {TGT}(first 60s) {np.round(b_a_t,3)} | {TGT}(all) {np.round(b_a_tall,3)}  m/s^2   factory bias0 already applied: {np.round(d['calib'].accel.bias0,3)}")
    trk_v = track_vision(runs[TGT], c, prep_t); trk_m = track_mocap(runs[TGT], c); tsv = trk_v.ts
    split = t0 + int(60e9)
    params = {"accel b=0": (np.zeros(3), np.zeros((3, 3))), f"b from {SRC}": (b_a_s, np.zeros((3, 3))), f"b+K_a from {SRC}": (bb_s, KK_s),
              "b own first-60s": (b_a_t, np.zeros((3, 3))), "b+K_a own first-60s": (bb_t, KK_t), "b own all-time [UB]": (b_a_tall, np.zeros((3, 3)))}
    def anchors(gap):
        out = []
        for i in range(0, len(tsv), 3):
            ta = tsv[i]; kk = np.searchsorted(tsv, ta + gap * 1e9)
            cs = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
            if not cs: continue
            k = min(cs, key=lambda k: abs(tsv[k] - (ta + gap * 1e9)))
            if abs(tsv[k] - (ta + gap * 1e9)) > max(0.006, 0.35 * gap) * 1e9 or np.max(np.diff(tsv[i:k + 1])) > 0.1e9: continue
            out.append((ta, tsv[k], trk_v.R[i], trk_v.P[i], trk_v.P[k]))
        return out
    vel_fn = lambda t: trk_v.v_at(t - int(0.03e9), 0.03, 3)
    print(f"   TARGET {TGT}: accel position-prediction error median/p90 (m), t>60 s, v0 from vision      gap(s): " + "".join(f"{g:>14.2f}" for g in GAPS_A))
    store = {}
    for g in GAPS_A:
        anc = [a for a in anchors(g) if a[0] >= split]
        store[g] = (np.array([a[0] for a in anc]), predict(runs[TGT], c, anc, gyro_tgt, d["accel"], d["lever"], vel_fn, params))
    for nm in ["naive"] + list(params):
        cells = []
        for g in GAPS_A:
            times, e = store[g]; x = e[nm][np.isfinite(e[nm])]
            cells.append(f"{np.median(x):.3f}/{np.percentile(x,90):.3f}" if len(x) > 10 else "n/a")
        print(f"   {nm[:40]:40s}" + "".join(f"{x:>14s}" for x in cells))
    print("   paired mean-error difference vs 'accel b=0' (m) [95% CI]:")
    for nm in [k for k in params if k != "accel b=0"]:
        line = []
        for g in (0.5, 1.0):
            times, e = store[g]; m = np.isfinite(e[nm]) & np.isfinite(e["accel b=0"]); diff = e[nm][m] - e["accel b=0"][m]
            lo, hi = block_bootstrap_ci(diff, times[m]); line.append(f"{g}s: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
        print(f"     {nm[:40]:40s} " + "   ".join(line))
