import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).parent
def _f(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v): return "-"
        return f"{v:.3g}" if (abs(v) >= 0.01 and abs(v) < 1000) or v == 0 else f"{v:.2e}"
    return str(v)
def md(df, fmt=None):
    cols = list(df.columns); s = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(_f(v) for v in r.values) + " |\n"
    return s
# ---- Exp1 CRB
c = pd.read_csv(OUT / "exp1_crb.csv"); c = c[c.cadence.str.startswith("strong")]
t1 = c[(c.rec == "static_dark") & (c.dev == "right")].pivot_table(index="W", columns="gravity", values=["sd_bg_median", "sd_ba_median"]).reset_index()
t1.columns = ["W (s)", "b_a sd, gravity known (m/s²)", "b_a sd, gravity ±1° (m/s²)", "b_g sd (rad/s)", "b_g sd dup"]
t1 = t1[["W (s)", "b_g sd (rad/s)", "b_a sd, gravity known (m/s²)", "b_a sd, gravity ±1° (m/s²)"]]
cl = c[(c.rec == "static_dark") & (c.dev == "left") & (c.gravity == "gravity known")][["W", "sd_bg_median", "sd_ba_median"]].rename(columns={"W": "W (s)", "sd_bg_median": "left b_g sd", "sd_ba_median": "left b_a sd"})
ex = pd.read_csv(OUT / "exp1b_excitation.csv"); ex["rot_amp_deg"] = ex.rot_amp_deg.astype(str)
tex = ex[ex.gravity != "gravity known"].pivot_table(index="rot_amp_deg", columns="W", values="sd_ba").reindex(["0", "5", "30", "120", "180", "3axis-15", "3axis-30", "3axis-60", "3axis-90"]).reset_index()
tex.columns = ["rotation excitation", "b_a sd W=10 s (gravity ±1°)", "b_a sd W=30 s (gravity ±1°)"]
# ---- Exp2
d = pd.read_csv(OUT / "exp2_convergence.csv"); sd = d[(d.rec == "static_dark") & (d.dev == "right")]
def tab(sc, kind, keep):
    x = sd[(sd.scenario == sc) & (sd.kind == kind)].groupby("estimator").agg(rms=("rms", "mean"), sd=("rms", "std"), ratio=("ratio_vs_zero", "mean")).reindex(keep).reset_index()
    x.columns = ["estimator", "steady RMS err (60-123 s)", "±seed sd", "ratio vs no correction"]; return x
gk = ["zero", "A_naive_ema_tau30", "A_dt_weighted_tau30", "A_dt_weighted_tau60", "A_median_tau30", "C_gyro_W30", "C_gyro_W60", "B_q3e-06", "B_q1e-05", "B_q3e-05"]
ak = ["zero", "A_accel_naive_ema", "A_accel_dt_weighted", "C_accel_W30", "B_q1e-05"]
# ---- Exp3
b = pd.read_csv(OUT / "exp3_benefit.csv"); b = b[(b.rec == "static_dark") & (b.dev == "right")]
def ben(sc, est):
    x = b[(b.scenario == sc) & (b.estimator == est)].groupby(["horizon_s", "opt"])[["rot_err_deg_median", "pos_err_mm_median"]].mean().unstack("opt")
    x.columns = [f"{a.split('_')[0]} {o}" for a, o in x.columns]; x = x.reset_index()
    return x[["horizon_s", "rot zero", "rot est", "rot true", "pos zero", "pos est", "pos true"]].rename(columns={"horizon_s": "horizon (s)", "rot zero": "rot° none", "rot est": "rot° est", "rot true": "rot° oracle", "pos zero": "pos mm none", "pos est": "pos mm est", "pos true": "pos mm oracle"})
# ---- Exp4
s4 = pd.read_csv(OUT / "exp4_sensitivity_table.csv")
s4.columns = ["failure mode (true bias = 0)", "gyro A(dtw τ30)", "gyro C(W30)", "gyro B", "accel B"]
a4 = pd.read_csv(OUT / "exp4b_attitude_only.csv").pivot_table(index="case", columns="variant", values="gyro_artifact").reset_index()
att = a4[["case", "B_full", "B_attitude_only"]].rename(columns={"case": "failure mode", "B_full": "B full update (rad/s)", "B_attitude_only": "B attitude-only (rad/s)"})
att = att.set_index("failure mode").reindex(["none", "timing 8ms", "headset ignored", "lever -50%", "lever ignored (-100%)", "gravity tilt 2deg", "gravity mag +1%", "axis misalign 2deg", "gyro scale 0.5%", "identity swap 3s", "corr vision err 1deg/4mm"]).reset_index()
# ---- Exp5
e5 = pd.read_csv(OUT / "exp5_design.csv"); b5 = e5[(e5.estimator == "B") & (e5.kind == "gyro")]
g5 = b5.groupby(["scenario", "q_bg", "meas_infl"]).rms.mean().reset_index()
g5 = g5[g5.q_bg.isin([1e-6, 1e-5, 3e-5, 1e-4])].pivot_table(index=["scenario", "q_bg"], columns="meas_infl", values="rms").reset_index()
g5.columns = ["scenario", "q_bg", "infl 1.0", "infl 1.5", "infl 2.5"]
qa = e5[(e5.estimator == "B") & (e5.kind == "accel")].groupby(["scenario", "q_ba"]).rms.mean().unstack().reset_index()
ag = e5[e5.scenario.isin(["factory", "large", "drift"]) & (e5.estimator == "A_dtw")].groupby(["gate", "tau"]).rms.mean().unstack().reset_index()
lf = pd.read_csv(OUT / "exp2c_left_C_gyro_fix.csv").groupby(["scenario", "estimator"])[["rms", "rms_zero", "ratio_vs_zero"]].mean().reset_index()
T = json = None
import json as _j
ts = _j.load(open(OUT / "test_sim_results.json")); te = _j.load(open(OUT / "test_estimators_results.json"))
rep = f"""# IMU bias estimation: simulation and estimator study (investigator S)

Scope: design and validate causal 6-parameter bias estimators (3 gyro + 3 accel) on **synthetic data with known truth**, built from
real mocap trajectories and the real sample timestamps of `static_dark` (and `walk_medium`), so that any later real-data failure can be
attributed to data/model mismatch rather than to the estimator. No repo source was modified. Everything is in `analysis/imu_bias/simulation/`.

## 1. Executive summary

1. **Which parameters are recoverable.** Gyro bias is observable from vision attitude alone, needs no special motion, and its precision improves as 1/W
   (CRB with real cadence and 0.4° vision noise: {t1.iloc[3,1]:.1e} rad/s at W=20 s, {t1.iloc[4,1]:.1e} at 30 s, {t1.iloc[5,1]:.1e} at 60 s). Accel bias is observable only
   if **gravity is known to well under 1° and the rotation excites 3 axes**: with one rotation axis the bias component along that axis is confounded with gravity error
   regardless of amplitude (sd 2e-2 m/s² at W=10 s), 3-axis rotation restores it (1e-3 at 90°). In this project's real motion (multi-axis, 130–180°) it is observable in principle.
2. **The mentor scheme, as literally described (per-frame prediction-vs-solution residual, running average), works only when the bias is large.**
   Gyro part: at mocap2gt-scale bias (|b_g|≈0.02 rad/s) it removes ~80–90 % of the error (best variant), but at **factory scale (2e-4 rad/s) it is 12–80× WORSE than applying no
   correction at all**, because a single vision-pose pair carries ~0.3–0.9 rad/s of noise and averaging only telescopes if weights are ∝ interval length.
   The naive per-frame EMA is 1.6× worse than the dt-weighted (telescoping) form; the windowed median is worst. The **accel part fails outright** (position residual vs a velocity taken from
   the previous vision difference is biased by a truncation error 100× larger than the bias signal even with perfect vision).
3. **The estimator that works is an error-state Kalman filter** (15 states: attitude, velocity, position, b_g, b_a; vision pose update). Steady-state error
   {sd[(sd.scenario=='factory')&(sd.kind=='gyro')&(sd.estimator=='B_q1e-05')].rms.mean():.1e} rad/s gyro and {sd[(sd.scenario=='factory')&(sd.kind=='accel')&(sd.estimator=='B_q1e-05')].rms.mean():.1e} m/s² accel; converges in ~1–2 s for large bias.
   It is the only estimator that beats "no correction" at factory scale (gyro ratio 0.75, accel ratio 0.30). It is the most robust to timing shifts, vision outliers/identity swaps and correlated vision error,
   but its gyro estimate is contaminated by lever-arm/gravity errors because the accel channel shares the filter (lever ignored: 2e-2 rad/s). An **attitude-only update variant** (no position update) removes that coupling
   (1.3e-4 rad/s regardless of lever/gravity errors) but is worse under gyro scale error and useless if headset motion is ignored (§6).
4. **Accel-bias estimates are a calibration-error absorber more than a bias measurement.** With true bias = 0, a 0.5° gravity tilt (0.018 m/s²), 3 ms timing error (0.014),
   20 % lever-arm error (0.048) or 0.5° sensor misalignment (0.015) each produce a *spurious* accel bias comparable to or larger than the factory bias (0.01). Ignoring headset ego-motion produces 0.12 m/s² and 1.5e-2 rad/s.
5. **Benefit is real but only for gaps ≥ 0.25 s.** From the true state, dead-reckoning position error at 1 s: 108 → 3.4 mm (mocap2gt-scale bias), 22 → 3.1 mm (warm-up drift), 7.4 → 2.2 mm (factory scale).
   Rotation at 1 s: 0.85° → 0.013° (large), 0.24° → 0.07° (drift), no gain (0.013°) at factory scale. Per-frame (22–88 ms) prediction is already far below vision noise (0.4°/4 mm).
6. **Recommendation:** try B with `q_bg=1e-5` (3e-5–1e-4 if a warm-up transient is expected), `q_ba=1e-3`, measurement-noise inflation 2.0–2.5, χ² gate 22.5, strong vision nodes only,
   headset motion from the headset pose, and precondition checks listed in §8. Do **not** ship the literal per-frame EMA. Real-bias magnitude decides whether any of this is worth it (§8).

## 2. Setup and validation of the simulator

* **Truth**: real mocap poses (marker → IMU frame via `T_imu_marker`, camera clock = mocap − fine offset) low-passed at 8 Hz; rotation defined by integrating the smoothed angular rate exactly.
* **Realism check against REAL gyro**: mocap-derived synthetic |ω| vs real controller gyro |ω| — static_dark median 2.94 vs 2.96 rad/s (corr 0.994), walk_medium 7.55 vs 7.68 rad/s (corr 0.981); |accel| 10.6 vs 10.4 m/s².
* **IMU**: real 200 Hz timestamps (+ lag −7.65/−7.60 ms), white noise gyro 7e-4 rad/s / accel 6.5e-3 m/s² (real quiet-segment measurement and factory `Noise`), bias = constant + random walk + optional exponential warm-up.
* **Vision**: real frame timestamps and strong-pose mask (n_inliers≥8 & error_px≤0.15/0.2): right 2095 nodes, left 533 nodes (p90 gap 0.5 s, max 8 s); noise 0.4° / 4 mm, 1 % gross outliers; optional AR(1) correlated error; headset ego-motion in the truth.
* **Known-answer tests**: `test_sim.py` {sum(t['ok'] for t in ts)}/{len(ts)} pass, `test_estimators.py` {sum(t['ok'] for t in te)}/{len(te)} pass. They verify against the **project's own** `integrate_gyro_segment` / `integrate_accel_to_position`
  (gyro 0.008° over 0.5 s; accel dead-reckoning 0.014 mm median vs 0.70 mm if the lever arm is ignored, so the test is sensitive to the lever-arm sign and gravity convention), the marker→IMU chain against `src.mocap_data.world_pose`,
  bias sign, exact recovery of constant/zero/negated/ramp bias with no noise, and that the estimators do not invent bias. Two real findings from the tests: (a) the window solver needs Gauss-Newton re-linearisation (0.5 rad accumulated angle at 0.026 rad/s × 20 s);
  (b) the first-order bias Jacobian error scales as b² (2.6e-5 rad worst interval at b=0.01, 2.6e-7 at 1e-3).

## 3. Observability (Exp1, analytic Cramér-Rao bounds on real trajectories; `exp1_crb.csv`, `plots/fig1_crb.png`)

Right controller, static_dark, real strong-node cadence, vision 0.4°/4 mm white, attitude known for the accel model (optimistic, see §4):

{md(t1)}
Left controller (only 533 strong nodes, sparse): 

{md(cl)}
Using all vision rows (22 ms) instead of strong-only improves the bound ~2× (e.g. gyro W=30 s: 4.1e-5 vs 9.2e-5).
Rotation excitation for accel bias (synthetic rotation, real cadence, gravity uncertain ±1° = 0.17 m/s²; `exp1b_excitation.csv`):

{md(tex)}
Gyro-bias sensitivity does not require excitation; very fast rotation slightly *reduces* it (W=10 s: 2.3e-4 static → 7.5e-4 at ±120° amplitude) because the body-fixed bias direction averages out.
In static_dark the controller always rotates about many axes (rotation span 131–180° in every 10 s window, mean |ω| 2.4–3.9 rad/s), so no real window is poorly excited.

## 4. Convergence and steady-state accuracy (Exp2; `exp2_convergence.csv`, `exp2b_settling_from_traces.csv`, `plots/fig2_traces_*.png`)

static_dark / right, 3 seeds, error over t = 60–123 s. Scenarios: **factory** (|b_g|≈2e-4 rad/s, |b_a|≈0.01 m/s², random walk 1e-5 / 1e-3), **large** (mocap2gt-scale 0.019 rad/s, 0.26 m/s²), **drift** (factory + warm-up 0.004 rad/s, 0.03 m/s², τ=40 s).
"ratio vs no correction" < 1 means the estimator helps.

Gyro, factory scale:

{md(tab('factory','gyro',gk))}
Gyro, large bias:

{md(tab('large','gyro',gk))}
Gyro, warm-up drift:

{md(tab('drift','gyro',gk))}
Accel, factory / large / drift:

{md(tab('factory','accel',ak))}
{md(tab('large','accel',ak))}
{md(tab('drift','accel',ak))}
Convergence (seed 0, large bias, time to be within 30 % of |b| for good): B 1.7 s (gyro) / 1.2 s (accel); C_gyro 5 s (needs a full window); A dt-weighted τ=30–60: 49 s; A naive EMA τ=30: 111 s; τ=10: never.
At factory scale no estimator reaches 30 % accuracy (the noise floor 7e-5 is ~35 % of |b_g|); B still reduces the error by 25 % (gyro) / 70 % (accel), while C is 1.8–3.7× and A 12–80× worse than doing nothing.
Notes: (i) C_accel's own reported sd (3e-5) is wildly over-confident because vision attitude noise (0.4° × 9.8 m/s² ≈ 0.07 m/s² per node) is not in its model; the ESKF smooths attitude with the gyro, which is why it is 30× better.
(ii) Left controller (sparse nodes): B gyro/large reaches only 5.6e-3 (ratio 0.5) — A dt-weighted τ=60 (2.0e-3) does better; B accel still helps (0.65 factory, 0.5 large). C_gyro needs `max_gap` ≥ 12 s to survive 3–8 s vision gaps; fixed:

{md(lf)}
(iii) walk_medium (median |ω| 7.7 rad/s, headset walking): B gyro/large 7.9e-4 (ratio 0.07) but at factory scale everything is worse than zero (B 7.8e-4 vs 1.1e-4): the **midpoint-rule gyro integration error at 200 Hz is equivalent to a pseudo-bias of 1.4e-3 rad/s** here (0.08° per second; 8.7e-5 in static_dark). A second-order Magnus term halves it (0.078° → 0.039°/s).
This is a property of the integrator (also the project's `integrate_gyro_segment`), not of the estimators.

## 5. Prediction benefit (Exp3; `exp3_benefit.csv`, `plots/fig4_benefit.png`)

Dead-reckoning from the TRUE state at t₀ over horizon h using the noisy simulated IMU, with no bias correction / estimated (causal, at t₀) / oracle bias. Median over start times every 0.4 s in 60–120 s. static_dark/right, mean of 3 seeds.

**large bias, ESKF (B):**

{md(ben('large','B_q1e-05'),'{:.3g}')}
**drift, ESKF (B):**

{md(ben('drift','B_q1e-05'),'{:.3g}')}
**factory, ESKF (B):**

{md(ben('factory','B_q1e-05'),'{:.3g}')}
**factory, mentor gyro scheme A (dt-weighted τ=30) + no accel correction:** rotation at 1 s 0.105° (est) vs 0.013° (none): the bias estimate makes prediction ~8× worse.

## 6. Sensitivity / failure modes (Exp4; `exp4_sensitivity.csv`, `plots/fig3_sensitivity.png`)

True bias = **0**, so every entry is a *spurious* estimated bias (mean over t>60 s, 2 seeds). Reference scales: factory 1e-4 rad/s and 0.01 m/s²; mocap2gt 0.011 rad/s and 0.2 m/s².

{md(s4)}
Reading: timing error δt gives artifact ≈ |a|·δt (accel) and ≈ ω·δt for the non-telescoping estimator A; the window/ESKF gyro estimators are robust to a pure timing shift. Headset ego-motion **must** be modelled (ignored: 0.12 m/s², 1.5e-2 rad/s). Gyro scale error hurts window solver C most (1.6e-2 at 0.5 %).
A gravity-tilt error is constant in the world frame, not the body frame, so it is only partly absorbed (2° gives 0.33 m/s², non-linear, ESKF gating interacts). Identity swap / outliers: B's χ² gate handles them, A and C do not.

### 6b. Gyro-bias artifact, ESKF full update vs attitude-only update (`exp4b_attitude_only.csv`; true bias 0, meas. inflation 2.0, 2 seeds)

{md(att)}
Attitude-only makes the gyro-bias estimate independent of the accel model (lever arm, gravity, accel scale) but loses the position/velocity constraint, so it suffers more from gyro scale error (5.9e-3 vs 8.5e-4 rad/s) and from ignored headset motion (3.7e-2 vs 1.4e-2).
A practical design is two tiers: attitude-only filter for b_g, full filter for b_a, with b_a frozen unless the preconditions in §8 hold.

## 7. Design choices (Exp5; `exp5_design.csv`, `plots/fig5_design_B.png`)

ESKF gyro-bias steady RMS (rad/s), mean over q_ba and 2 seeds:

{md(g5)}
* **Measurement-noise inflation is critical with correlated vision error**: at inflation 1.0 the filter becomes over-confident (1195 gate rejections, 7e-3 rad/s); 1.5 → 1.2e-4, 2.5 → 1.0e-4. Use 2.0–2.5.
* **q_bg trades steady noise against warm-up tracking**: constant bias favours ≤1e-5 (8.4e-5), a 0.004 rad/s warm-up prefers 1e-4 (2.9e-4 vs 1.0e-3 at 1e-6) but costs steady noise (1.7e-4). 3e-5 is a sensible compromise; adapt if the real drift is known.
* **q_ba = 1e-3 m/s²/√s** is best for accel bias (RMS by q_ba: 1e-4 → 5.6e-3, 1e-3 → 3.3e-3, 1e-2 → 5.5e-3 at factory scale):

{md(qa)}
* For A (dt-weighted): residual gate 0.05–0.12 rad best (0.02 rejects real data, 0.5 lets outliers in), longer τ is better (RMS 3.0e-3 at τ=10, 1.5e-3 at 30, 1.25e-3 at 60 for gate 0.05–0.12):

{md(ag)}
## 8. Recommendation and what to check on real data

Real-data plan (not run here): (1) use another investigator's mocap-derived oracle bias trajectory as truth; (2) run B causally on static_dark with the parameters below; (3) check the estimated b_g against the oracle and the prediction benefit of §5 on real occlusion gaps.
Estimator: **B (ESKF)**, two-tier: attitude-only update (`update_pos=False`) for b_g whenever lever arm / gravity / accel calibration are not trusted to the precision below (its gyro artifact stays 1.3e-4 rad/s), full update for b_a, with `q_bg=1e-5` (3e-5 if warm-up drift is seen in the oracle), `q_ba=1e-3`, `sig_bg0=0.02`, `sig_ba0=0.3`, measurement noise = 2.0–2.5 × (0.4°, 4 mm), χ² gate 22.5 (6 dof), strong nodes only (using all rows with per-row noise should help ~2× per the CRB).
Preconditions that must hold before the accel channel is meaningful (spurious-bias table): gravity direction within ≲0.3°, |g| within 0.3 %, timing within ≲1 ms, lever arm within ≲10 %, axis misalignment ≲0.5°, headset motion supplied from a pose source (not ignored). If they do not hold, freeze b_a or treat it as an empirical correction, not a bias.
If the mentor's literal scheme is still wanted: use the dt-weighted accumulation (Σ residual / Σ J, exponential forgetting τ ≥ 60 s), gate 0.05–0.12 rad, gyro only; expect benefit only if |b_g| ≳ 3e-3 rad/s.
**Decision rule from the oracle numbers**: if the real gyro bias is ≲ 3e-4 rad/s and accel bias ≲ 0.02 m/s², a correction changes 1 s dead-reckoning by mm/0.01° — probably not worth the risk; if it is mocap2gt-scale (0.01 rad/s, 0.2 m/s²) it is worth 30–100× less position error at 1 s.

## 9. Caveats / not modelled

White vision noise (+ optional AR(1)); real vision has pose-dependent systematic error (≈1°, 4–5 mm floor, right-controller −0.78 % scale, correlated over ~1 s), so absolute numbers are optimistic. Bias is body-constant + RW + exponential warm-up: no temperature-dependent
scale/mixing, g-sensitivity, vibration rectification, accel nonlinearity. Gravity, lever arm, headset pose are given to the estimator (perturbed only in Exp4). Truth is smoothed at 8 Hz. Seeds: 2–3 per cell, so sd is indicative. C_accel used vision-derived attitude only (a fixed-lag solver with gyro-smoothed attitude would do better).
Estimator B was tuned on the same simulated scenarios (Exp5) — check on real data. ESKF cost: ~2 s per 124 s recording in Python.

## 10. Files

`simlib.py` simulator · `estlib.py` estimators (A/B/C) · `test_sim.py`, `test_estimators.py` known-answer tests (+ `*_results.json`) · `crb.py`, `crb_excitation.py` observability · `run_experiments.py` (exp2/3/4/5) · `make_plots.py`, `make_report.py` ·
CSVs: `exp1_crb.csv`, `exp1_crb_windows.csv`, `exp1_excitation_split.csv`, `exp1b_excitation.csv`, `exp2_convergence.csv`, `exp2b_settling_from_traces.csv`, `exp2c_left_C_gyro_fix.csv`, `exp3_benefit.csv`, `exp4_sensitivity.csv`, `exp4_sensitivity_table.csv`, `exp4b_attitude_only.csv`, `exp5_design.csv` · `plots/`, `traces/` (seed-0 bias trajectories).
"""
(OUT / "REPORT.md").write_text(rep); print("REPORT.md written", len(rep))
