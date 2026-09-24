# IMU bias research: consolidated report (2026-09-23)

Question: can we run a live, causal bias estimator (3 gyro + 3 accel parameters, time-varying) by
comparing the IMU-predicted pose to the vision solution each frame and feeding the correction into the
next prediction (mentor's scheme)? Start with static_dark.

Five independent investigations, each with its own artifacts (paths below). UPDATE 2026-09-23: the loader and
lever-arm fixes (section 4, items 1-2) were then drafted in the working tree with tests (fix_impl/README.md); they are
NOT committed. Everything else in this report is analysis only.

## 1. Answer in one paragraph

The literal scheme cannot work at the size of bias we have (per-frame SNR ~0.01), and, more importantly,
**bias is not the leading IMU error**. Three deterministic calibration errors are larger and are fixable
once, without any live estimation: (a) the loader applies the factory mix+bias a second time (the recorded
CSV is already corrected by the Monado driver), (b) accel is scaled by a hard-coded driver constant
(10.0 instead of 9.80665 m/s^2 per g, ~1.9 %), (c) the accel lever arm used by the tracker points ~97 deg
away from the true accelerometer position (right length, wrong frame). What is left after those is a small,
nearly constant per-session bias, for which a live estimator adds roughly 0 to 0.16 deg at 1 s (gyro) and
matters only for gaps of ~0.3 s and longer.

## 2. Evidence table (who found what)

| # | Finding | Source | Verified how |
|---|---|---|---|
| 1 | Residual gyro bias after factory cal: ~0.007-0.012 rad/s (70-80x factory BiasUncertainty), constant within a recording, stable per session | Oracle O, Audit F, Real R | mocap-derived angular velocity, 6 moderate recordings |
| 2 | A perfect (even non-causal) gyro bias removes only 0.2-2 % of rotation prediction error; time-varying bias adds nothing over a constant | O, R | oracle ceiling experiments |
| 3 | Recorded controller CSV is already factory-corrected by Monado (wmr_controller_hp.c:245-276: mix, bias, P_oxr); loader `correct()` applies mix+bias again | Audit F | source reading + "CSV as-is beats loader in 12/12 pairs"; "CSV is raw" rejected at 3-5 sigma |
| 4 | Accel gain error ~1.9 % = Monado divisor 49000 -> 10.000 m/s^2 per g (own @todo "confirm the scale"); x0.980665 collapses fitted gain to ~0 | F (O and R see the same gain) | fit + unit test of the constant |
| 5 | Deterministic fix (3)+(4) alone: accel position error 21.2 -> 10.3 mm at 0.3 s (-52 %), 196 -> 75 mm at 1 s (-62 %); oracle ceiling 8.1 / 44.6 mm; gyro median rotation error at 1 s 1.70 -> 1.35 deg | F | oracle scoring loops, 12 recording-controller pairs |
| 6 | Tracker accel lever arm (main.py:199, factory Rt composition) = (34.6,-77.5,2.8) mm left / (-33.2,-80.2,2.9) mm right; bridge.t (= accelerometer origin in the LED frame; the bridge maps IMU-frame points to LED-frame points) = (5.6,7.4,-83.9) / (-5.7,7.6,-82.8) mm. Same length, **~95 deg apart**; the factory vector read in the right frame, -R^T t = (4.8,9.9,-84.2) / (-5.4,9.6,-86.2) mm, agrees with bridge.t to 2.6 / 3.9 mm. (An earlier draft of this report used -R_b^T t_b, which had the bridge direction backwards; corrected after independent review.) | R; I re-verified with the code and current bridges | direct computation (this report) + R's independent regression (agrees to 10-17 mm/axis) |
| 7 | Fixing the lever arm: accel position error at 1 s 0.42 -> 0.23 m (left), 0.44 -> 0.23 m (right); today's accel prediction loses to naive constant velocity on short real gaps | R | held-out (t > 60 s), paired CIs on real gaps exclude 0 |
| 8 | Gyro static misalignment/scale K: ~1 deg about x + ~1 % scale; K from static_dark transfers to walk_medium (1 s: 2.07 -> 1.51 deg left, 2.60 -> 1.29 deg right) | R | held-out + cross-recording |
| 9 | Literal per-frame scheme: gyro SNR ~0.01/frame (~40 k frames), accel ~0.02; consecutive residuals anti-correlated (lag-1 -0.44..-0.48) so only accumulation (sum e / sum dt, length-weighted) works; naive per-frame EMA 24x worse with the 11/22 ms alternating cadence | R, Literature L, Sim S | real data + Monte Carlo |
| 10 | Literal scheme at factory-scale bias is 12-80x WORSE than no correction; only a 15-state error-state Kalman filter beats no correction (gyro 7e-5 rad/s, accel 3e-3 m/s^2 steady state); accel channel absorbs calibration errors (0.5 deg gravity tilt = 0.018 m/s^2, 3 ms timing = 0.014, 20 % lever error = 0.048, ignoring headset motion = 0.12) | Sim S | simulator validated against project integrators (9/9 + 12/12 exact tests) |
| 11 | Earlier batch attempt failed because residuals were dominated by the wrong lever arm, accel scale error and gyro K (44-150 sigma before bias mattered); its "bias irrelevant" conclusion was right for the wrong reason | R (matches README findings 3-5) | residual decomposition |

Numbers tagged with the recording set: 6 moderate recordings (static_dark, walk_dark, static_easy,
static_medium, walk_easy, walk_medium), one ~25 min session, one pair of controllers. static_hard/walk_hard
fit poorly and were excluded by the oracle.

## 3. Targets after the deterministic fixes (per-session offsets, from F)

Gyro b (rad/s): left (-0.0008, -0.0083, 0.0007), right (-0.0002, 0.0111, -0.0016).
Accel b (m/s^2): left (0.045, 0.079, 0.021), right (-0.067, -0.108, -0.085).
Both stable within a session (drift <= 1.5e-4 rad/s/min gyro, <= 1.3e-3 m/s^2/min accel), inside ICM-20602
datasheet initial-offset tolerances. No temperature channel exists in the CSV; the factory models have only
constant terms, so a temperature effect cannot be tested.

## 4. Recommended plan, in payoff order

1. **Lever arm** (main.py:199): DONE 2026-09-23 (uncommitted draft) as -R_acc^T t_acc (frame fix, mocap-independent) -- see fix_impl/README.md.
2. **Loader** (`load_and_calibrate_controller_imu`): DONE 2026-09-23 (uncommitted draft): no second `correct()`, `_DIAG_FLIP` kept, accel x0.980665, config switches, startup guard. Re-measure all oracle biases afterwards (not yet done).
3. **Gyro K / accel gain**: apply only after 1-2 and only vision-referenced (see risks). Candidate: the gyro "1 deg about x" may just be the bridge rotation's deviation from an exact diag(1,-1,-1) (left bridge R off-diagonals 0.008/0.0155 ~ 0.5/0.9 deg; hypothesis, untested).
4. **Live bias estimator, only if still wanted**: error-state Kalman filter, attitude-only update for gyro bias, accel channel frozen unless gravity within ~0.3 deg / timing within ~1 ms / lever arm within ~10 %; q_bg 1e-5 (3e-5 if drift), q_ba 1e-3, vision-noise inflation 2.0-2.5, chi2 gate 22.5, per-step |e| < 3 deg gate, ~2 s hold-off after any gate trip or tracker flag, raw strong vision as the reference (never the fused pose), headset ego-motion supplied from a pose source. Initialise at the section 3 offsets. Expected payoff: ~0 to -0.16 deg at 1 s gyro; none on fast motion.

## 5. Risks and open items

- All of 1-3 change the LIVE loader/fusion. Per the project's standing rule they need an independent physics review and exact-value numeric tests before any code changes.
- Most fits are mocap-referenced. Controller T_imu_marker came from a non-converged mocap2gt solve (cost 1.0e6 / 3.1e6), so a constant mocap-frame rotation error is absorbed into K and the lever arm. The lever-arm claim is protected by the bridge (fit against vision) and by R's regression; the gyro K claim is not yet vision-referenced.
- Which Monado build recorded the data is unverified (data agree with the source; "already corrected" is strongly supported, not proven).
- RESOLVED (fix_validation/REPORT.md section 3): the +4.5 % rest |f| was scale (1.97 %) plus the doubled factory bias projecting onto gravity; after both loader fixes quiet |f| is 9.81..9.93 m/s^2 (+0.03..+1.2 %), the remainder being the real per-session bias.
- R used the current (double-corrected) loader stream; its accel/gyro K and bias numbers must be re-derived on the corrected stream.
- Deployable headset ego-motion: imu0 works as ego source once its own gyro bias is calibrated per session (0.013/0.012/0.001 rad/s static_dark, 0.020/0.007/0.002 walk_medium); uncorrected it shifts the controller estimate by ~0.04 rad/s. Mocap headset motion is dev-only.
- Not tested: temperature, sessions longer than ~25 min, other recordings for R's fixes, hard recordings.

## 6. Where the details are

- Oracle: analysis/imu_bias/oracle/REPORT.md (+ CSVs, figures)
- Simulation: analysis/imu_bias/simulation/REPORT.md
- Factory audit: analysis/imu_bias/factory_audit/ (scripts, *_fits.txt, payoff_*.txt/csv; its written report is in the session transcript only, summarised in section 2 rows 3-5 and section 3)
- Real data: analysis/imu_bias/realdata/ (run_all.sh, proposed_calibration.json, figures/, results_csv/, logs/, 12 exact-value tests)
- Literature: analysis/imu_bias/literature/ (sources/ extracts, sim_observability output; its written report is summarised in section 7)

## 7. Literature and reference-code summary (investigator L)

- The mentor's scheme is VINS-Mono's gyro-bias initialisation (eq. 15, arXiv 1708.03852) run continuously; accel bias is ignored there because it is "coupled with gravity ... hard to be observed".
- Correct form: e_k = Log(R_pred^T R_v,k) ~ -(b - b_hat) * dt_k; accumulate sum(e)/sum(dt), never mean(e/dt). Vision orientation noise 0.3 deg, ~1 s correlation time (project's own numbers): achievable gyro-bias error 3.5e-4 rad/s (random-walk model), ~1e-3 or worse with correlated vision error; error-state KF formula 2^(1/4) q^(3/4) R_c^(1/8).
- Accel bias floor from orientation error: g*sin(dtheta) = 0.017 / 0.051 / 0.086 / 0.171 m/s^2 for 0.1 / 0.3 / 0.5 / 1 deg (same size as the biases themselves).
- Stationary detection is essentially impossible on these data (<= 2.3 % of windows have gyro std < 0.05 rad/s; 0 % in *_hard), so no zero-velocity updates.
- Local reference code: Monado fork has no bias estimation (m_imu_3dof tilt correction gated on |a|~9.82 and |w|<0.1 rad/s for 20 ms; gyro bias = manual 300 ms mean; temperature field parsed and never used). Basalt has b_g/b_a states with random-walk residual (imu_block.hpp:64-84; defaults gyro noise 2.82e-4, accel 1.6e-2, bias 1e-4 / 1e-3). Only OpenHMD's Rift UKF (thaytan) estimates bias online in constellation-controller code, treating it as nearly constant and near-ignoring optical orientation. Beyley's sensor_fusion.cpp is gone (404).
- ICM-20602 datasheet: gyro 0.004 dps/sqrt(Hz), initial ZRO +-1 dps, ZRO vs temperature +-0.01 dps/C, scale +-1 %, cross-axis +-1 %; accel 100 ug/sqrt(Hz), zero-g +-25/40 mg, +-0.5/1 mg/C. No bias-instability figure published.
- Factory Noise fields (6.76e-4 rad/s, 7.6e-3 m/s^2) are per-sample RMS, not densities; Basalt's defaults are deliberately inflated (6x / 30x).
- Basalt's `cam_time_offset_ns` is commented out in the VIO (sqrt_keypoint_vio.cpp:256-257).
