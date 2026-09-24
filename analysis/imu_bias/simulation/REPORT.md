# IMU bias estimation: simulation and estimator study (investigator S)

Scope: design and validate causal 6-parameter bias estimators (3 gyro + 3 accel) on **synthetic data with known truth**, built from
real mocap trajectories and the real sample timestamps of `static_dark` (and `walk_medium`), so that any later real-data failure can be
attributed to data/model mismatch rather than to the estimator. No repo source was modified. Everything is in `analysis/imu_bias/simulation/`.

## 1. Executive summary

1. **Which parameters are recoverable.** Gyro bias is observable from vision attitude alone, needs no special motion, and its precision improves as 1/W
   (CRB with real cadence and 0.4° vision noise: 1.4e-04 rad/s at W=20 s, 9.2e-05 at 30 s, 2.8e-05 at 60 s). Accel bias is observable only
   if **gravity is known to well under 1° and the rotation excites 3 axes**: with one rotation axis the bias component along that axis is confounded with gravity error
   regardless of amplitude (sd 2e-2 m/s² at W=10 s), 3-axis rotation restores it (1e-3 at 90°). In this project's real motion (multi-axis, 130–180°) it is observable in principle.
2. **The mentor scheme, as literally described (per-frame prediction-vs-solution residual, running average), works only when the bias is large.**
   Gyro part: at mocap2gt-scale bias (|b_g|≈0.02 rad/s) it removes ~80–90 % of the error (best variant), but at **factory scale (2e-4 rad/s) it is 12–80× WORSE than applying no
   correction at all**, because a single vision-pose pair carries ~0.3–0.9 rad/s of noise and averaging only telescopes if weights are ∝ interval length.
   The naive per-frame EMA is 1.6× worse than the dt-weighted (telescoping) form; the windowed median is worst. The **accel part fails outright** (position residual vs a velocity taken from
   the previous vision difference is biased by a truncation error 100× larger than the bias signal even with perfect vision).
3. **The estimator that works is an error-state Kalman filter** (15 states: attitude, velocity, position, b_g, b_a; vision pose update). Steady-state error
   7.3e-05 rad/s gyro and 3.3e-03 m/s² accel; converges in ~1–2 s for large bias.
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
* **Known-answer tests**: `test_sim.py` 9/9 pass, `test_estimators.py` 12/12 pass. They verify against the **project's own** `integrate_gyro_segment` / `integrate_accel_to_position`
  (gyro 0.008° over 0.5 s; accel dead-reckoning 0.014 mm median vs 0.70 mm if the lever arm is ignored, so the test is sensitive to the lever-arm sign and gravity convention), the marker→IMU chain against `src.mocap_data.world_pose`,
  bias sign, exact recovery of constant/zero/negated/ramp bias with no noise, and that the estimators do not invent bias. Two real findings from the tests: (a) the window solver needs Gauss-Newton re-linearisation (0.5 rad accumulated angle at 0.026 rad/s × 20 s);
  (b) the first-order bias Jacobian error scales as b² (2.6e-5 rad worst interval at b=0.01, 2.6e-7 at 1e-3).

## 3. Observability (Exp1, analytic Cramér-Rao bounds on real trajectories; `exp1_crb.csv`, `plots/fig1_crb.png`)

Right controller, static_dark, real strong-node cadence, vision 0.4°/4 mm white, attitude known for the accel model (optimistic, see §4):

| W (s) | b_g sd (rad/s) | b_a sd, gravity known (m/s²) | b_a sd, gravity ±1° (m/s²) |
|---|---|---|---|
| 2 | 3.28e-03 | 0.0117 | 0.069 |
| 5 | 1.20e-03 | 1.42e-03 | 7.86e-03 |
| 10 | 3.93e-04 | 1.97e-04 | 1.46e-03 |
| 20 | 1.41e-04 | 3.29e-05 | 3.09e-04 |
| 30 | 9.16e-05 | 1.43e-05 | 1.08e-04 |
| 60 | 2.83e-05 | 2.35e-06 | 2.23e-05 |

Left controller (only 533 strong nodes, sparse): 

| W (s) | left b_g sd | left b_a sd |
|---|---|---|
| 2 | 3.90e-03 | 0.0111 |
| 5 | 2.06e-03 | 2.30e-03 |
| 10 | 1.22e-03 | 7.97e-04 |
| 20 | 3.82e-04 | 1.02e-04 |
| 30 | 2.34e-04 | 4.65e-05 |
| 60 | 8.00e-05 | 6.80e-06 |

Using all vision rows (22 ms) instead of strong-only improves the bound ~2× (e.g. gyro W=30 s: 4.1e-5 vs 9.2e-5).
Rotation excitation for accel bias (synthetic rotation, real cadence, gravity uncertain ±1° = 0.17 m/s²; `exp1b_excitation.csv`):

| rotation excitation | b_a sd W=10 s (gravity ±1°) | b_a sd W=30 s (gravity ±1°) |
|---|---|---|
| 0 | 0.0357 | 7.56e-03 |
| 5 | 0.0247 | 6.64e-03 |
| 30 | 0.0207 | 4.61e-03 |
| 120 | 0.0206 | 4.42e-03 |
| 180 | 0.0207 | 4.54e-03 |
| 3axis-15 | 5.24e-03 | 1.97e-03 |
| 3axis-30 | 2.54e-03 | 7.82e-04 |
| 3axis-60 | 1.31e-03 | 2.99e-04 |
| 3axis-90 | 1.05e-03 | 1.90e-04 |

Gyro-bias sensitivity does not require excitation; very fast rotation slightly *reduces* it (W=10 s: 2.3e-4 static → 7.5e-4 at ±120° amplitude) because the body-fixed bias direction averages out.
In static_dark the controller always rotates about many axes (rotation span 131–180° in every 10 s window, mean |ω| 2.4–3.9 rad/s), so no real window is poorly excited.

## 4. Convergence and steady-state accuracy (Exp2; `exp2_convergence.csv`, `exp2b_settling_from_traces.csv`, `plots/fig2_traces_*.png`)

static_dark / right, 3 seeds, error over t = 60–123 s. Scenarios: **factory** (|b_g|≈2e-4 rad/s, |b_a|≈0.01 m/s², random walk 1e-5 / 1e-3), **large** (mocap2gt-scale 0.019 rad/s, 0.26 m/s²), **drift** (factory + warm-up 0.004 rad/s, 0.03 m/s², τ=40 s).
"ratio vs no correction" < 1 means the estimator helps.

Gyro, factory scale:

| estimator | steady RMS err (60-123 s) | ±seed sd | ratio vs no correction |
|---|---|---|---|
| zero | 9.76e-05 | 2.74e-05 | 1 |
| A_naive_ema_tau30 | 2.23e-03 | 6.75e-04 | 22.8 |
| A_dt_weighted_tau30 | 1.44e-03 | 1.78e-04 | 15.6 |
| A_dt_weighted_tau60 | 1.14e-03 | 2.18e-04 | 12.4 |
| A_median_tau30 | 7.47e-03 | 1.27e-03 | 81 |
| C_gyro_W30 | 3.49e-04 | 3.37e-05 | 3.74 |
| C_gyro_W60 | 1.67e-04 | 2.50e-05 | 1.78 |
| B_q3e-06 | 7.27e-05 | 2.29e-05 | 0.746 |
| B_q1e-05 | 7.29e-05 | 2.42e-05 | 0.745 |
| B_q3e-05 | 8.94e-05 | 2.15e-05 | 0.926 |

Gyro, large bias:

| estimator | steady RMS err (60-123 s) | ±seed sd | ratio vs no correction |
|---|---|---|---|
| zero | 0.0112 | 4.46e-05 | 1 |
| A_naive_ema_tau30 | 2.26e-03 | 4.83e-04 | 0.201 |
| A_dt_weighted_tau30 | 1.44e-03 | 1.78e-04 | 0.128 |
| A_dt_weighted_tau60 | 1.14e-03 | 2.17e-04 | 0.102 |
| A_median_tau30 | 7.47e-03 | 1.27e-03 | 0.665 |
| C_gyro_W30 | 3.51e-04 | 3.43e-05 | 0.0313 |
| C_gyro_W60 | 1.66e-04 | 2.45e-05 | 0.0148 |
| B_q3e-06 | 8.02e-05 | 2.52e-05 | 7.14e-03 |
| B_q1e-05 | 8.08e-05 | 2.60e-05 | 7.19e-03 |
| B_q3e-05 | 9.77e-05 | 2.16e-05 | 8.69e-03 |

Gyro, warm-up drift:

| estimator | steady RMS err (60-123 s) | ±seed sd | ratio vs no correction |
|---|---|---|---|
| zero | 3.20e-03 | 4.35e-05 | 1 |
| A_naive_ema_tau30 | 2.13e-03 | 4.71e-04 | 0.666 |
| A_dt_weighted_tau30 | 1.60e-03 | 1.57e-04 | 0.499 |
| A_dt_weighted_tau60 | 1.52e-03 | 2.54e-04 | 0.475 |
| A_median_tau30 | 7.47e-03 | 1.28e-03 | 2.33 |
| C_gyro_W30 | 6.45e-04 | 1.82e-05 | 0.202 |
| C_gyro_W60 | 1.44e-03 | 3.73e-05 | 0.449 |
| B_q3e-06 | 1.02e-03 | 2.74e-05 | 0.318 |
| B_q1e-05 | 9.64e-04 | 2.47e-05 | 0.301 |
| B_q3e-05 | 6.92e-04 | 1.20e-05 | 0.216 |

Accel, factory / large / drift:

| estimator | steady RMS err (60-123 s) | ±seed sd | ratio vs no correction |
|---|---|---|---|
| zero | 0.0115 | 3.85e-03 | 1 |
| A_accel_naive_ema | 0.422 | 0.136 | 37 |
| A_accel_dt_weighted | 1.01 | 0.361 | 95.3 |
| C_accel_W30 | 0.0906 | 0.0569 | 7.23 |
| B_q1e-05 | 3.29e-03 | 5.97e-04 | 0.301 |

| estimator | steady RMS err (60-123 s) | ±seed sd | ratio vs no correction |
|---|---|---|---|
| zero | 0.156 | 3.43e-03 | 1 |
| A_accel_naive_ema | 0.379 | 0.151 | 2.41 |
| A_accel_dt_weighted | 1.01 | 0.365 | 6.49 |
| C_accel_W30 | 0.0907 | 0.0569 | 0.577 |
| B_q1e-05 | 3.42e-03 | 7.40e-04 | 0.0219 |

| estimator | steady RMS err (60-123 s) | ±seed sd | ratio vs no correction |
|---|---|---|---|
| zero | 0.0312 | 2.84e-03 | 1 |
| A_accel_naive_ema | 0.414 | 0.139 | 13.1 |
| A_accel_dt_weighted | 1.01 | 0.359 | 32.9 |
| C_accel_W30 | 0.0905 | 0.0569 | 2.82 |
| B_q1e-05 | 4.25e-03 | 5.76e-04 | 0.137 |

Convergence (seed 0, large bias, time to be within 30 % of |b| for good): B 1.7 s (gyro) / 1.2 s (accel); C_gyro 5 s (needs a full window); A dt-weighted τ=30–60: 49 s; A naive EMA τ=30: 111 s; τ=10: never.
At factory scale no estimator reaches 30 % accuracy (the noise floor 7e-5 is ~35 % of |b_g|); B still reduces the error by 25 % (gyro) / 70 % (accel), while C is 1.8–3.7× and A 12–80× worse than doing nothing.
Notes: (i) C_accel's own reported sd (3e-5) is wildly over-confident because vision attitude noise (0.4° × 9.8 m/s² ≈ 0.07 m/s² per node) is not in its model; the ESKF smooths attitude with the gyro, which is why it is 30× better.
(ii) Left controller (sparse nodes): B gyro/large reaches only 5.6e-3 (ratio 0.5) — A dt-weighted τ=60 (2.0e-3) does better; B accel still helps (0.65 factory, 0.5 large). C_gyro needs `max_gap` ≥ 12 s to survive 3–8 s vision gaps; fixed:

| scenario | estimator | rms | rms_zero | ratio_vs_zero |
|---|---|---|---|---|
| factory | C_gyro_W30 | 2.12e-04 | 1.17e-04 | 1.82 |
| factory | C_gyro_W60 | 6.94e-04 | 1.17e-04 | 5.85 |
| factory | C_gyro_W90 | 7.45e-04 | 1.17e-04 | 6.27 |
| large | C_gyro_W30 | 2.22e-04 | 0.0112 | 0.0198 |
| large | C_gyro_W60 | 7.22e-04 | 0.0112 | 0.0644 |
| large | C_gyro_W90 | 7.66e-04 | 0.0112 | 0.0684 |

(iii) walk_medium (median |ω| 7.7 rad/s, headset walking): B gyro/large 7.9e-4 (ratio 0.07) but at factory scale everything is worse than zero (B 7.8e-4 vs 1.1e-4): the **midpoint-rule gyro integration error at 200 Hz is equivalent to a pseudo-bias of 1.4e-3 rad/s** here (0.08° per second; 8.7e-5 in static_dark). A second-order Magnus term halves it (0.078° → 0.039°/s).
This is a property of the integrator (also the project's `integrate_gyro_segment`), not of the estimators.

## 5. Prediction benefit (Exp3; `exp3_benefit.csv`, `plots/fig4_benefit.png`)

Dead-reckoning from the TRUE state at t₀ over horizon h using the noisy simulated IMU, with no bias correction / estimated (causal, at t₀) / oracle bias. Median over start times every 0.4 s in 60–120 s. static_dark/right, mean of 3 seeds.

**large bias, ESKF (B):**

| horizon (s) | rot° none | rot° est | rot° oracle | pos mm none | pos mm est | pos mm oracle |
|---|---|---|---|---|---|---|
| 0.022 | 0.0245 | 1.56e-03 | 1.56e-03 | 0.065 | 3.37e-03 | 3.04e-03 |
| 0.044 | 0.0491 | 2.50e-03 | 2.49e-03 | 0.26 | 0.0117 | 0.0105 |
| 0.088 | 0.0977 | 3.74e-03 | 3.72e-03 | 1.04 | 0.042 | 0.0351 |
| 0.25 | 0.272 | 5.66e-03 | 5.44e-03 | 8.19 | 0.304 | 0.235 |
| 0.5 | 0.502 | 8.52e-03 | 7.91e-03 | 31 | 1.09 | 0.795 |
| 1 | 0.852 | 0.0126 | 0.0116 | 108 | 3.43 | 2.26 |

**drift, ESKF (B):**

| horizon (s) | rot° none | rot° est | rot° oracle | pos mm none | pos mm est | pos mm oracle |
|---|---|---|---|---|---|---|
| 0.022 | 6.96e-03 | 2.51e-03 | 1.56e-03 | 0.0128 | 3.00e-03 | 2.60e-03 |
| 0.044 | 0.014 | 4.90e-03 | 2.49e-03 | 0.0514 | 0.0103 | 7.62e-03 |
| 0.088 | 0.0276 | 9.07e-03 | 3.72e-03 | 0.207 | 0.0343 | 0.0226 |
| 0.25 | 0.0769 | 0.0247 | 5.44e-03 | 1.65 | 0.237 | 0.0968 |
| 0.5 | 0.141 | 0.0437 | 7.90e-03 | 6.16 | 0.853 | 0.294 |
| 1 | 0.238 | 0.0736 | 0.0115 | 22 | 3.11 | 0.914 |

**factory, ESKF (B):**

| horizon (s) | rot° none | rot° est | rot° oracle | pos mm none | pos mm est | pos mm oracle |
|---|---|---|---|---|---|---|
| 0.022 | 1.54e-03 | 1.56e-03 | 1.56e-03 | 4.95e-03 | 2.82e-03 | 2.49e-03 |
| 0.044 | 2.56e-03 | 2.49e-03 | 2.49e-03 | 0.0196 | 9.31e-03 | 7.44e-03 |
| 0.088 | 3.89e-03 | 3.71e-03 | 3.72e-03 | 0.0762 | 0.0304 | 0.0209 |
| 0.25 | 6.19e-03 | 5.68e-03 | 5.44e-03 | 0.603 | 0.179 | 0.0732 |
| 0.5 | 8.54e-03 | 8.22e-03 | 7.91e-03 | 2.23 | 0.636 | 0.204 |
| 1 | 0.0129 | 0.0123 | 0.0116 | 7.37 | 2.16 | 0.686 |

**factory, mentor gyro scheme A (dt-weighted τ=30) + no accel correction:** rotation at 1 s 0.105° (est) vs 0.013° (none): the bias estimate makes prediction ~8× worse.

## 6. Sensitivity / failure modes (Exp4; `exp4_sensitivity.csv`, `plots/fig3_sensitivity.png`)

True bias = **0**, so every entry is a *spurious* estimated bias (mean over t>60 s, 2 seeds). Reference scales: factory 1e-4 rad/s and 0.01 m/s²; mocap2gt 0.011 rad/s and 0.2 m/s².

| failure mode (true bias = 0) | gyro A(dtw τ30) | gyro C(W30) | gyro B | accel B |
|---|---|---|---|---|
| none | 1.04e-03 | 5.48e-04 | 9.55e-05 | 8.70e-04 |
| timing 1ms | 2.32e-03 | 5.05e-04 | 8.63e-05 | 4.35e-03 |
| timing 3ms | 7.45e-03 | 4.59e-04 | 1.01e-04 | 0.014 |
| timing 8ms | 0.0187 | 6.25e-04 | 1.28e-04 | 0.0279 |
| headset ignored | 6.45e-03 | 0.0101 | 0.0153 | 0.123 |
| headset vio drift 1e-3 | 1.02e-03 | 4.66e-04 | 1.29e-04 | 2.98e-03 |
| headset vio drift 3e-3 | 1.16e-03 | 6.47e-04 | 2.68e-04 | 0.0154 |
| lever +20% | 1.04e-03 | 5.48e-04 | 1.08e-04 | 0.0479 |
| lever -50% | 1.04e-03 | 5.48e-04 | 5.96e-03 | 0.0519 |
| lever ignored (-100%) | 1.04e-03 | 5.48e-04 | 8.73e-03 | 0.12 |
| gravity tilt 0.5deg | 1.04e-03 | 5.48e-04 | 1.20e-04 | 0.0176 |
| gravity tilt 1deg | 1.04e-03 | 5.48e-04 | 1.61e-04 | 0.0339 |
| gravity tilt 2deg | 1.04e-03 | 5.48e-04 | 2.26e-03 | 0.327 |
| gravity mag +1% | 1.04e-03 | 5.48e-04 | 4.40e-03 | 0.142 |
| axis misalign 0.5deg | 0.0107 | 5.09e-04 | 1.82e-04 | 0.015 |
| axis misalign 1deg | 0.0203 | 6.19e-04 | 3.37e-04 | 0.03 |
| axis misalign 2deg | 0.0396 | 1.01e-03 | 8.73e-03 | 0.2 |
| gyro scale 0.1% | 1.67e-03 | 2.79e-03 | 5.89e-04 | 5.00e-03 |
| gyro scale 0.5% | 5.05e-03 | 0.0157 | 8.63e-04 | 0.0161 |
| accel scale 0.3% | 1.04e-03 | 5.48e-04 | 5.79e-04 | 0.0378 |
| vision outliers 5% | 2.41e-03 | 5.56e-04 | 6.82e-05 | 2.25e-04 |
| identity swap 3s | 0.0182 | 4.94e-03 | 9.79e-05 | 7.97e-04 |
| corr vision err 0.5deg/2mm | 3.29e-03 | 5.15e-04 | 1.40e-04 | 2.84e-03 |
| corr vision err 1deg/4mm | 5.81e-03 | 8.93e-04 | 3.95e-04 | 7.39e-03 |
| vision noise x2 | 2.02e-03 | 5.84e-04 | 9.78e-05 | 8.37e-04 |
| dt jitter 0.5ms | 2.16e-03 | 7.68e-04 | 2.04e-04 | 1.33e-03 |
| dt jitter 1ms | 2.18e-03 | 1.09e-03 | 3.39e-04 | 1.19e-03 |

Reading: timing error δt gives artifact ≈ |a|·δt (accel) and ≈ ω·δt for the non-telescoping estimator A; the window/ESKF gyro estimators are robust to a pure timing shift. Headset ego-motion **must** be modelled (ignored: 0.12 m/s², 1.5e-2 rad/s). Gyro scale error hurts window solver C most (1.6e-2 at 0.5 %).
A gravity-tilt error is constant in the world frame, not the body frame, so it is only partly absorbed (2° gives 0.33 m/s², non-linear, ESKF gating interacts). Identity swap / outliers: B's χ² gate handles them, A and C do not.

### 6b. Gyro-bias artifact, ESKF full update vs attitude-only update (`exp4b_attitude_only.csv`; true bias 0, meas. inflation 2.0, 2 seeds)

| failure mode | B full update (rad/s) | B attitude-only (rad/s) |
|---|---|---|
| none | 9.84e-05 | 1.27e-04 |
| timing 8ms | 1.53e-04 | 2.73e-04 |
| headset ignored | 0.0136 | 0.0369 |
| lever -50% | 1.53e-04 | 1.27e-04 |
| lever ignored (-100%) | 0.0198 | 1.27e-04 |
| gravity tilt 2deg | 1.31e-03 | 1.27e-04 |
| gravity mag +1% | 4.72e-03 | 1.27e-04 |
| axis misalign 2deg | 2.02e-03 | 6.93e-04 |
| gyro scale 0.5% | 8.53e-04 | 5.90e-03 |
| identity swap 3s | 1.01e-04 | 1.25e-04 |
| corr vision err 1deg/4mm | 2.58e-04 | 5.39e-04 |

Attitude-only makes the gyro-bias estimate independent of the accel model (lever arm, gravity, accel scale) but loses the position/velocity constraint, so it suffers more from gyro scale error (5.9e-3 vs 8.5e-4 rad/s) and from ignored headset motion (3.7e-2 vs 1.4e-2).
A practical design is two tiers: attitude-only filter for b_g, full filter for b_a, with b_a frozen unless the preconditions in §8 hold.

## 7. Design choices (Exp5; `exp5_design.csv`, `plots/fig5_design_B.png`)

ESKF gyro-bias steady RMS (rad/s), mean over q_ba and 2 seeds:

| scenario | q_bg | infl 1.0 | infl 1.5 | infl 2.5 |
|---|---|---|---|---|
| drift | 1.00e-06 | 1.03e-03 | 1.03e-03 | 1.03e-03 |
| drift | 1.00e-05 | 9.62e-04 | 9.68e-04 | 9.72e-04 |
| drift | 3.00e-05 | 6.82e-04 | 6.98e-04 | 7.14e-04 |
| drift | 1.00e-04 | 2.93e-04 | 3.00e-04 | 3.07e-04 |
| factory | 1.00e-06 | 8.93e-05 | 8.39e-05 | 8.27e-05 |
| factory | 1.00e-05 | 9.00e-05 | 8.54e-05 | 8.44e-05 |
| factory | 3.00e-05 | 1.06e-04 | 1.03e-04 | 1.01e-04 |
| factory | 1.00e-04 | 1.80e-04 | 1.68e-04 | 1.58e-04 |
| factory+corr_vision | 1.00e-06 | 7.01e-03 | 1.25e-04 | 1.02e-04 |
| factory+corr_vision | 1.00e-05 | 7.01e-03 | 1.25e-04 | 1.02e-04 |
| factory+corr_vision | 3.00e-05 | 7.04e-03 | 1.48e-04 | 1.16e-04 |
| factory+corr_vision | 1.00e-04 | 7.17e-03 | 3.20e-04 | 2.28e-04 |
| large | 1.00e-06 | 9.50e-05 | 9.04e-05 | 8.88e-05 |
| large | 1.00e-05 | 9.63e-05 | 9.21e-05 | 9.05e-05 |
| large | 3.00e-05 | 1.15e-04 | 1.11e-04 | 1.07e-04 |
| large | 1.00e-04 | 1.89e-04 | 1.75e-04 | 1.62e-04 |

* **Measurement-noise inflation is critical with correlated vision error**: at inflation 1.0 the filter becomes over-confident (1195 gate rejections, 7e-3 rad/s); 1.5 → 1.2e-4, 2.5 → 1.0e-4. Use 2.0–2.5.
* **q_bg trades steady noise against warm-up tracking**: constant bias favours ≤1e-5 (8.4e-5), a 0.004 rad/s warm-up prefers 1e-4 (2.9e-4 vs 1.0e-3 at 1e-6) but costs steady noise (1.7e-4). 3e-5 is a sensible compromise; adapt if the real drift is known.
* **q_ba = 1e-3 m/s²/√s** is best for accel bias (RMS by q_ba: 1e-4 → 5.6e-3, 1e-3 → 3.3e-3, 1e-2 → 5.5e-3 at factory scale):

| scenario | 0.0001 | 0.001 | 0.01 |
|---|---|---|---|
| drift | 0.0104 | 4.36e-03 | 7.95e-03 |
| factory | 5.57e-03 | 3.33e-03 | 5.46e-03 |
| factory+corr_vision | 0.0131 | 0.0118 | 0.0189 |
| large | 5.56e-03 | 3.41e-03 | 5.65e-03 |

* For A (dt-weighted): residual gate 0.05–0.12 rad best (0.02 rejects real data, 0.5 lets outliers in), longer τ is better (RMS 3.0e-3 at τ=10, 1.5e-3 at 30, 1.25e-3 at 60 for gate 0.05–0.12):

| gate | 10.0 | 30.0 | 60.0 |
|---|---|---|---|
| 0.02 | 6.97e-03 | 4.98e-03 | 4.59e-03 |
| 0.05 | 2.98e-03 | 1.50e-03 | 1.25e-03 |
| 0.12 | 2.99e-03 | 1.51e-03 | 1.25e-03 |
| 0.5 | 3.18e-03 | 2.04e-03 | 1.88e-03 |

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
