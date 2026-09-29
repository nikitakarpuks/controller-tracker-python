# IMU bias oracle study (investigator O) — mocap as ground truth

Data: current evaluation set (Sep 22, vision offset 7.65/7.60 ms + drift correction), 8 recordings x 2 controllers, project loaders
(`load_and_calibrate_controller_imu`, lag from `controller_imu_lag_ns`, `DeviceMocap` lookup incl. offsets). All scripts/CSVs in this folder.
"Bias" = residual on top of the factory T=0 calibration the loader already applies (mix @ raw + bias0). Sensor frame = mocap-IMU frame (body = diag(1,-1,-1) @ sensor).

## 0. Sanity gates (both pass)
* Frame map (`gate_mocap_gyro.py`): gyro vs mocap relative rotation over 0.1 s: `sensor` frame hypothesis residual 0.57–0.69 deg/0.1 s (~4% of the 16–18 deg/0.1 s motion); identity hypothesis 26 deg. So mocap-IMU frame == sensor frame.
* Gyro vs vision (`gate_vision_gyro.py`, static_dark strong frames): median rotation error 0.292 (left) / 0.297 deg (right) per frame interval (README finding 9: 0.34/0.51 at the old lag).
* Accel: mocap "IMU" position IS the accelerometer position (fitted extra lever arm 0.5–7 mm vs the 85 mm factory lever arm).
* Data reality check: the "static" recordings are not static. Median |omega| = 2.3–3 rad/s (static_dark/easy, walk_dark/easy), 5–7 (medium), 9–10 rad/s (hard). Rest (|w|<0.3 rad/s, v<5 cm/s) is 1–8% of the time, and the long rests at the start (mocap t=1–4.5 s) lie BEFORE the IMU stream begins (IMU covers mocap-track time ~4 s..127 s). Only a handful of >=1 s rests exist inside IMU coverage.

## 1. Gyro bias (per controller, rad/s; 1 rad/s = 57.3 deg/s)
Method (`gyro_oracle.py`): exact gyro integration over 0.5 s intervals vs mocap relative rotation, first-order model gyro = (I+K) w + b with analytic Jacobians; K (3x3 gain/misalignment) pooled over the 8 recordings (SE 0.0004), bias per recording, 4-MAD robust, block bootstrap CIs. Cross-checked: results stable for tau 0.25/0.5/1 s; rest-interval means (K-independent) agree in sign/size but are noisy (+-0.005).

| controller | mean bias over 8 recordings (x, y, z) | between-recording sd | |b| |
|---|---|---|---|
| left  | (-0.0011, -0.0026, **-0.0074**) | ~(0.003, 0.003, 0.002) | 0.008 rad/s = 0.46 deg/s |
| right | (**+0.0033, +0.0057, -0.0040**) | ~(0.001, 0.002, 0.003) | 0.007 rad/s = 0.41 deg/s |

Per-recording CIs (bootstrap, K free) are +-0.002–0.005 rad/s (wider for the hard recordings, +-0.01). Per-recording values with pooled K are in the report body above / `gyro_bias_global.csv`, `gyro_bias_windows.csv`.
* **Magnitude vs factory**: 70–80x the factory BiasUncertainty (1e-4 rad/s); same scale as the mocap2gt figures in the README (0.008–0.014). Timing sensitivity: a +-4 ms IMU timestamp shift moves the estimate by <=0.0015 (left) / up to 0.004 (right z); the shipped lag is good to <1 ms.
* **Time behaviour**: window estimates (10/20 s, SE 0.005–0.008 rad/s) scatter exactly as their standard errors (chi2/dof 0.75–1.2 at 10–20 s; 1.0–1.8 at 40 s) => statistically a CONSTANT offset within each recording; any variation is < ~0.003 rad/s at 40 s. Left: flat across all time bins (z = -0.0078 +-0.001, no ramp, no warm-up transient). Right: marginal ramp x: -0.003 -> +0.004, y: +0.005 -> +0.008 over 2 min (slope +0.004+-0.0018 / +0.003+-0.0011 rad/s per min, ~2 sigma). Between recordings (25 min): no trend beyond scatter. => a stable per-session offset, not a fast random walk.
* **Gain/misalignment K matters more than bias**: pooled K (left diag ~[0.007,0.008,0.000], right ~[0.012,0.005,0.006], off-diagonals up to +-0.01) i.e. ~0.5–1.2% gain and <=0.6 deg misalignment (includes any T_imu_marker rotation error of the mocap calibration). At the typical 3 rad/s this is 0.02–0.03 rad/s, 3–4x the bias.
* Residual after fitting b and K is still 0.73 deg per 0.5 s interval (MAD): dominated by mocap orientation noise (~0.17 deg/sample -> 0.24 deg per difference) and few-ms timing; not modelled.

## 2. Accelerometer bias (m/s^2)
Method (`accel_oracle.py`): position-level matched model, no differentiation of mocap: per 2 s window p_mocap = p0 + v0 t + double-integral[R(f - b - S f) + g0 + dg], p0/v0 nuisance projected out; unknowns b(3), S(3x3 gain/misalign), dg(3 gravity correction); bootstrap over windows. VALID only for the 6 moderate recordings (position misfit 4–11 mm); static_hard/walk_hard misfit 21–24 mm and inconsistent biases (motion too fast for this model) — excluded from bias inference.
Pooled S,dg over 6 recordings, bias per recording (95% CI ~ +-0.01–0.02; +-0.035 walk_medium):

| controller | b (x, y, z) at the first three recordings | drift over the 25 min session |
|---|---|---|
| left  | (0.083, 0.179, -0.038) / (0.081, 0.181, -0.046) / (0.071, 0.177, -0.046) | x 0.083 -> 0.026, z -0.04 -> -0.11 (~-0.002 m/s^2/min) |
| right | (-0.077, 0.102, -0.276) / (-0.079, 0.110, -0.280) / (-0.079, 0.119, -0.277) | x -0.08 -> 0.00, z -0.28 -> -0.32 |

* |b| ~ 0.20 (left) / 0.30 (right) m/s^2 = 20–30x the factory BiasUncertainty (0.01); README's mocap2gt scale (0.17–0.21) reproduced for left.
* Gain S: diag 1.8–2.5% (left ~[0.021,0.021,0.023], right ~[0.019,0.025,0.018]); dg <= 0.06 m/s^2 (<0.35 deg gravity tilt). b is only moderately correlated with S_yy (-0.3..-0.4) and dg (-0.2..-0.35); cond(A^T A) 1e3–1e4 => identifiable in these dynamic recordings because orientation varies (b is body-fixed, g is world-fixed).
* Time behaviour: within a recording, 20 s block estimates scatter 0.02–0.04 (chi2/dof 1–6 vs bootstrap SE ~0.017: some excess, likely model error rather than true drift), no systematic within-recording trend; slow session drift ~0.002 m/s^2/min (left x, right x/z; corr with time 0.6–0.8 over only 6 recordings — suggestive).
* Rest-based accel check was not possible: <1 s of true rest inside IMU coverage per recording.

## 3. Noise and stability (`noise_allan.py`; only 4 quiet runs >=1 s inside coverage)
* Per-sample white noise (lag-1 differences): gyro 0.0026–0.0037 rad/s (0.15–0.21 deg/s; ~15–21 mdps/rtHz at 100 Hz BW vs ICM-20602 datasheet ~4 mdps/rtHz), accel 0.010–0.012 m/s^2 (~100–120 ug/rtHz, matches datasheet ~100). Factory 'Noise' field: gyro 6.8e-4, accel 7.6e-3 (units unstated; ~4x / 0.8x below the measured per-sample sigma).
* Allan deviation rises with tau (16 mrad/s at 0.1–0.3 s): the quiet runs contain hand tremor, so a bias-instability floor CANNOT be measured from these recordings; use the window-consistency bound instead (gyro < ~0.003 rad/s at 40 s; accel ~0.02–0.03 at 20 s).
* No temperature channel; factory BiasTemperatureModel/mixing temperature terms are zero beyond the constant term, so warm-up cannot be attributed. No warm-up transient visible in the first 25 s (left flat; right first-bin slightly lower, within errors).

## 4. Materiality
Bias-induced error vs gap T (b_g = 0.008 rad/s, b_a = 0.2 m/s^2): rotation b*T = 0.009 / 0.023 / 0.046 / 0.14 / 0.46 deg at T = 0.02/0.05/0.1/0.3/1.0 s; position 0.5 b T^2 = 0.04 / 0.25 / 1 / 9 / 100 mm.
Oracle upper bounds (score = angle(Rm^T Rg) against mocap; the mocap floor ~0.24 deg + timing is included in all numbers), 250 random starts per rec/ctrl/T, all 16 recording-controller pairs (`gyro_upperbound.csv`):

| median rot. error (deg) | T=0.022 | 0.044 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|
| factory (current) | 0.564 | 0.720 | 0.954 | 1.306 | 1.871 |
| const oracle BIAS only | 0.563 | 0.719 | 0.949 | 1.292 | 1.833 |
| K only (gain/misalign) | 0.558 | 0.701 | 0.894 | 1.125 | 1.259 |
| const bias + K | 0.556 | 0.700 | 0.893 | 1.113 | 1.211 |
| time-varying bias (10 s) + K | 0.557 | 0.698 | 0.893 | 1.115 | 1.243 |
| ONE bias for all recordings + K | 0.557 | 0.700 | 0.891 | 1.113 | 1.207 |

* **Bias alone removes only 0.2–2% of gyro prediction error at any gap** (even the non-causal per-recording oracle); K removes 1.4% (22 ms) to 35% (1 s). A time-varying bias adds nothing over one constant (0.0 to -3% worse). Real vision-loss gaps (556 gaps 0.15–1.2 s, `gyro_realgaps.csv`): factory median 1.38 deg -> 1.09 (bias+K) / 1.10 (K only) / 1.12 (time-varying bias+K).
* Short gaps (1–4 frames, 11–44 ms): error 0.56–0.72 deg is mostly the mocap floor and timing; nothing (bias or K) can improve it by more than ~3%.
* **Accel** (`accel_upperbound.csv`; mocap initial state and orientation, 6 moderate recordings, median position error, mm): factory 1.84 / 3.53 / 21.2 / 196 at T = 0.05/0.1/0.3/1.0 s; bias only 1.81 / 3.05 / 13.8 / 112; gain(S,dg) only 1.71 / 2.89 / 14.8 / 115; bias+S+dg 1.68 / 2.56 / 8.2 / 44. Accel correction matters from ~0.1 s (-27%) and strongly from 0.3 s (-61%); gain (2%) is as important as bias.
* Answers: is bias a leading error term? **Gyro: no** (gain/misalignment K, mocap-floor/timing and noise-free residuals are larger). **Accel: bias+gain are the leading terms for gaps >= 0.3 s**, negligible for 1–4 frame gaps.

## 5. Implications for a live estimator (targets)
* Gyro target: a constant ~0.007–0.008 rad/s offset (|b|), per-controller signature above, stable per session; to matter it must be estimated to < ~0.002 rad/s — mocap needs >= 20–40 s windows to reach 0.004–0.005 SE, so vision-based live estimation must average over tens of seconds; a fast-tracking estimator would just track noise. Expected benefit if perfect: <= 2% of gyro prediction error. Adding 3 gain terms (diag K, ~1%) gives 10–35% at 0.1–1 s gaps.
* Accel target: b ~ 0.2–0.3 m/s^2 slowly drifting (~0.002/min), jointly identifiable with gain S (~2%) and gravity only under orientation excitation; benefit only for gaps >= 0.1–0.3 s.
* Do not trust the hard recordings (walk_hard/static_hard) for accel fits; gyro fits there have +-0.01 CIs.

## Caveats
Mocap orientation noise (~0.17 deg) and few-ms timing limit interval-level precision; K absorbs any constant T_imu_marker rotation error (fine for correcting against the mocap/inertial frame, but a vision-referenced estimator sees K relative to the bridge frame); no long static log exists so Allan bias-instability is unmeasured; temperature unknown; results are from one ~25 min session (8 recordings) of one pair of controllers.

Files: `REPORT.md`, `common.py` (loaders), `gate_*.py`, `gyro_oracle.py` (+`gyro_all.py`, `gyro_global.py`, `gyro_windows.py`, `gyro_sensitivity.py`, `rest_gyro.py`), `accel_oracle.py` (+`accel_all.py`, `accel_fits.py`, `accel_time.py`), `noise_allan.py`, `gyro_upperbound.py`, `accel_upperbound.py`, `gyro_realgaps.py`, `make_figs.py`; CSVs: `gyro_bias_global.csv`, `gyro_bias_windows.csv`, `gyro_bias_window_stats.csv`, `gyro_bias_rest.csv`, `gyro_upperbound.csv`, `gyro_realgaps.csv`, `accel_bias_per_recording.csv`, `accel_bias_blocks.csv`, `accel_upperbound.csv`, `noise_rest_runs.csv`; figures `fig_*.png`; cached terms `*.npz/*.pkl`.
