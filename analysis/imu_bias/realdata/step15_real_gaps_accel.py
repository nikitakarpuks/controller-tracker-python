"""Step 15: accel position dead-reckoning over REAL tracking-loss gaps (consecutive strong frames 0.15-1.5 s apart), lever arm = bridge-derived.
Methods: naive | accel b=0, K_a=0 | factory-lever accel b=0 (what main.py does today) | + const (b,K_a) fit first 60 s (mocap windows) | + causal RLS (vision) b+K_a.
All (b,K_a) fits use ONLY data before the evaluated gap (first-60 s fit: gaps after t0+60 s; causal RLS: state at the gap start)."""
import sys, os
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step6_gyro_estimators import block_bootstrap_ci
from step10b_accel_K import fit_bK, predict
from step11_accel_causal import AccelRLS, series
assert LEVER_MODE == "bridge", "run with LEVER=bridge"
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; prep = prepare(run, c, ego="mocap")
    trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
    t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); t_end = prep["ts"][-1]
    K_source_run = run
    st0, _ = make_steps(run, c, prep); bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
    gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
    AW_m = AccelWindows(run, c, trk_m, gyro=gyro_c); AW_v = AccelWindows(run, c, trk_v, gyro=gyro_c)
    b_c, K_c, _ = fit_bK(AW_m, t0, split, 5, True, True)
    T, P = series(AW_v, t0, t_end, 40, True)
    tsv = trk_v.ts
    gaps = [(tsv[i], tsv[i + 1], trk_v.R[i], trk_v.P[i], trk_v.P[i + 1]) for i in range(len(tsv) - 1) if 0.15e9 < tsv[i + 1] - tsv[i] <= 1.5e9]
    gaps = [g for g in gaps if g[0] >= split]
    vel_fn = lambda t: trk_v.v_at(t - int(0.03e9), 0.03, 3)
    zero = (np.zeros(3), np.zeros((3, 3)))
    # per-gap causal RLS params
    res = {k: [] for k in ["naive", "accel b=0 (bridge lever)", "+const b,K_a (fit first 60 s)", "+causal RLS b,K_a (tau=40 s)"]}
    for g in gaps:
        k = np.searchsorted(T, g[0], side="right") - 1
        prm = {"accel b=0 (bridge lever)": zero, "+const b,K_a (fit first 60 s)": (b_c, K_c), "+causal RLS b,K_a (tau=40 s)": P[k] if k >= 0 else zero}
        o = predict(run, c, [g], gyro_c, d["accel"], d["lever"], vel_fn, prm)
        for kk in res: res[kk].append(o[kk][0])
    res = {k: np.array(v) for k, v in res.items()}
    # factory lever (what main.py uses today): temporarily swap lever
    lev_bridge = d["lever"].copy(); d["lever"] = d["lever_factory"]
    o = predict(run, c, gaps, gyro_c, d["accel"], d["lever"], vel_fn, {"accel b=0 (FACTORY lever = main.py today)": zero}); d["lever"] = lev_bridge
    res["accel b=0 (FACTORY lever = main.py today)"] = o["accel b=0 (FACTORY lever = main.py today)"]
    ok = np.all([np.isfinite(v) for v in res.values()], axis=0)
    dur = np.array([(g[1] - g[0]) / 1e9 for g in gaps])[ok]
    print(f"\n[{name}/{c}] REAL tracking-loss gaps after t0+60 s: n={ok.sum()}  gap duration median {np.median(dur):.2f}s (p10 {np.percentile(dur,10):.2f}, p90 {np.percentile(dur,90):.2f})   [position error to the next strong vision frame, m]")
    order = ["naive", "accel b=0 (FACTORY lever = main.py today)", "accel b=0 (bridge lever)", "+const b,K_a (fit first 60 s)", "+causal RLS b,K_a (tau=40 s)"]
    for nm in order:
        e = res[nm][ok]
        print(f"   {nm:46s} median {np.median(e):.3f}  p90 {np.percentile(e,90):.3f}  mean {e.mean():.3f}")
    times = np.array([g[0] for g in gaps])[ok]
    for nm in order[2:]:
        diff = res[nm][ok] - res["accel b=0 (FACTORY lever = main.py today)"][ok]
        lo, hi = block_bootstrap_ci(diff, times, block_s=5.0)
        print(f"   paired mean diff vs FACTORY-lever accel: {nm:38s} {diff.mean():+.3f} m [{lo:+.3f},{hi:+.3f}]")
