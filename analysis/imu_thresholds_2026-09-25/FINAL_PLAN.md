# FINAL PLAN - IMU thresholds fix (coordinator's reconciliation of A1/A2/A3, C1/C2, D1, K1/K2/K3; 2026-09-25)
Status: PLAN ONLY. Nothing in src/, config/, main.py, tests/ has been changed. Requires user approval per commit stage.
Evidence base: analysis/imu_thresholds_2026-09-25/{a1_live_sweep,a2_candidate_rules,a3_inventory,c1_physics,c2_stats,d1_spec,k1_code,k2_adversarial,k3_safety,saturation}/ + A1_REPORT.md, A2_REPORT.md, CRITIC_FINDINGS.md, d1_spec/SPEC.md (v1, superseded where this plan differs).

## 1. What changes and what does not
- ROTATION (gyro) decisions: replaced by ONE continuous gate valid at all elapsed times (no dt<=budget fail-open switch). Position gates/budgets: NOT changed (evidence does not support loosening: position innovation useless beyond ~50 ms; hard/medium recordings violate looser policies; results depend on dev-only mocap headset).
- Every new config key defaults to legacy behaviour (byte-identical default path); shipped config flips flags in separate stages.

## 2. Final rotation gate (reconciled constants)
T(dt, window) = 
- headset ego-motion correction ACTIVE:  T = 40 + 25*sat        (cap 75, never binds live)
- headset correction NOT available for this call:  T = min(150, 40 + 25*sat + 500*dt)
- sat = 1 if the window [last_update_ts - 10 ms pad, frame_ts + pad] contains a gyro-clip sample (|x|>=0.95*2000, |y|>=0.95*2000, |z|>=0.95*2722 dps on the filter's BODY-frame gyro; mix diag 1.000-1.010, so no inverse mix needed - K1 verified 0 missed of 387 truth samples) OR peak gyro >= 1800 dps.
- state age > 0.35 s (expiry): gate not applied (shipped behaviour), covers the reachable band dt in (0.3, 0.33] and makes the 0.3 s coast reset explicit.
- Veto iff rot_innov_deg > T. Applied to: cold-reacquire veto (rot_pred_implausible, replaces the dt<=0.25-0.00018*G switch), warm hard-gate rotation ceiling (replaces 21.3+6.54|v|+0.0046(G-900)+2200*stale), and the ceiling reused by Case A / _vision_weight / _agreement_* (K1 verified those are the only consumers).
- Exemption kept: _rotation_seed_grace_frames > 0 (grace exists only after coverage_fallback seeds).
- Why these numbers: a=40 clears the leave-one-recording-out max unflagged GOOD innovation (25.2 deg) by >=14.8 deg and the between-pair predictive q99 (44.4) is covered only in flagged windows by +25; the fitted 0.05*G*dt term is DROPPED (fit to ~3 walk_hard saturation events, not physics, unbounded); the saturation allowance is tied to the measured cause (gyro clip => burst under-integration 30-45 deg). H=500 (>=400 needed, max needed 304 deg/s in A1 live_rig, headroom ~1.65x).
- Measured (in-sample, 8 recordings, state_ok, 83,279 GOOD / 313 BAD): FP 0/83,279 (worst GOOD margin 14.8 deg HS; 15+ deg NOHS), catches 235/313 over all dt (144/217 at dt<=0.35; 132/204 at dt<=0.3); shipped rules catch 22-36/313. Honest FP upper bound in the relevant stratum (dt 50-350 ms, n=871): 3.4e-3 per candidate. Per-pair: FP 0 on all 16 (recording, controller) pairs.
- Problem frames (HS): 348 (172 vs 40), 5333 (87), 5337 (95), 5339 (154; also via pair rule), 5629 (126), 1941 (58.4, dt 0.322 s: live-reachable band), static_hard 4310, 5504, 5872, 5812(72), walk_hard 3959/3963/3969/4169/5626 caught; 6235 (38.4 vs 40) NOT caught; NOHS misses 1941, 5403, 5629, 6235, static_hard 5873.
- NOT IMU-catchable (documented): follower frames after a wrong lock (5337/5338 followers, 5340, 5404-5410 ...), position-only-bad candidates (50/313), wrong candidates within ~25-35 deg of the gyro prediction, identity swaps with similar orientation.

## 3. Weak-confirm and override (gyro consistency)
- In _try_cold_reacquire confirm step (only log_prefix != BOOTSTRAP): keep shipped position/rotation budget as an extra AND; add gyro residual = angle( R_pending^-1-relative-rotation vs gyro-integrated relative rotation over the pair interval ) <= 20 deg (+15 if pair window flagged) with headset correction, <= 20 + 400*pair_dt without. Use rotation-only algebra R_hc1 = R_wh1^T R_wh0 R_pending Rg (no predict_headset_relative_pose: avoids accel/gravity/position-seed dependence) - K1.
- Measured pairs (1078 both-good / 52 both-bad / 297 mixed): shipped 1056/24/151; new HS 1050/0/15; NOHS 1053/5/62. Lost both-good pairs delayed 1-2 frames (11-33 ms).
- Reject-streak override: add the same gyro residual on its two strong candidates (flag default off; 175/175 strong both-good pairs pass).

## 4. Rotation coast/anchor budgets (STAGE 4, optional, lowest priority)
- Widen only the ROTATION budgets (degenerate w_sum<=0 gate, _mark_all_lost anchor): HS 0.30 s for G<1000 dps ramping to 0.035 s at 1500; NOHS 0.05 s (0.15 s calm). Position budgets unchanged. Only in fusion_heuristic; matching gets one mode key coast_rot_budget_mode (getattr guard for Kalman filter). Ship LAST; revert first if walk_hard frame 39/71 anchor/coast regress.

## 5. Mandatory additions from the critics
1. Saturation flag helper (src/imu_data.py gyro_window_clip_flags(gyro_data, ts_lo, ts_hi, ceil_rad_s, frac, pad_ns)) + startup diagnostic printing per-axis max of the loaded gyro stream (a wrong ceiling fails silent).
2. Compute headset inputs ONCE per predict()/gate call (stamp _pred_mode=(ts0,ts1,hs)); a naive _headset_inputs is 0.83 ms/call (4-8 ms/frame). Keep world_pose/headset_* module-level names in pose_fusion_heuristic (HeadsetCorrectionTests patch them and count calls). Legacy mode must not call any new code.
3. NOHS fail-open guard (K3 major): when predict() returns None (rig-frame g_world not converged / lever missing) every candidate is accepted unchecked (walk_hard NOHS: 254/254 fail_open, incl. 110-154 deg candidates). Add a gyro-only rotation prediction for the gate in that case (R_state @ gyro_rel, headset-corrected if available) and log the fail_open count per run.
4. dt window/pad: _peak_gyro_accel uses pad 0 (2-3 samples in 11 ms, G=0 if accel_data None): use pad >= 10 ms for the clip/gap flag; clamp negative dt; _headset_inputs(None, ts) returns None.
5. Persistent bad stream (K1): after 6 vetoes -> reset -> BOOTSTRAP path (no gyro check) re-anchors the same mirror lock: the veto DELAYS the lock; validation must look >= 0.5 s past the first veto. Lock-out bound: <= 6 rejects / 0.3 s (should_force_cold_start), then 1-3 slow brute frames.
6. Seed-grace accepts (first two after coverage_fallback seed) stay ungated: monitor count.
7. Optional last commit: headset IMU (imu0) rotation ego-motion source, default off (headset_ego_source: mocap|imu0|none). C1: innovation shift median 0.00, p99 1.67, max 11.3 deg; R/FP/catches identical. Nothing loads imu0 today; ~3 ms timing offset noted. Makes NOHS a fallback only.

## 6. Tests
- Baseline BEFORE any change (K1): 7 failed + 3 errors (5 test_camera_kb4_rpmax, 2 test_proximity_strong_match Frame41, 3 errors test_brute_aux_coverage) - freeze in k1_code/baseline_failures.txt; 306 related tests pass.
- Existing tests needing explicit rot_gate_mode: legacy: RotPredImplausibleSignalTests.test_past_its_own_credibility_window..., WarmGateSpeedAwareHardRejectTests test_2/test_3/test_6, StaleTimeWideningTests (rotation half), ImplausibleJump*/QualityAware*/CaseA*/AgreementWeakBounds* (pin legacy numbers). No test reads shipped fusion_heuristic; tests/test_imu_loader.py compares imu_loader_kwargs dict exactly (do not extend it).
- New tests in NEW files (dirty test files by others must not be touched): threshold function table (a=40,S=25,H=500 values computed from the formula, not literals baked to old a=35), real-case regressions (348: dt 0.2441 G 300.8 innov 172.1; 5333: 0.0555/799.7/87.3; 5337: 0.0666/530/95.0; 5339 pair 469 mm/170.4 deg over 88.8 ms; 4310, 5872, 4169, 1941), FP guards (real GOOD candidates near threshold incl. the 7 with innov>25 and 4873/4874 in flagged windows), HS/NOHS per-call mode + mocap-hole A/B/A, NOHS unconverged-g fallback, expiry, clip-flag synthetic cases, override/weak-confirm. Template: k1_code/test_k1_harness.py (fails on current code as intended: 348-like accepted, 5333-like fused; GOOD guard passes). HeadsetCorrectionTests constraints: keep single lookup set per predict().

## 7. Commit order (each independently revertible; default legacy; hunk-staged)
- C0 (no code): freeze baseline failures + baseline replay artefacts (walk_easy 700-1700 pose.csv/vision_pose.csv).
- C1 refactor: _headset_inputs + _pred_mode stamp + disable_headset_ego_motion (dev switch, default false). Legacy byte-identical (replay diff = 0 or stop).
- C2 pure functions: _rot_gate_threshold_deg, gyro_window_clip_flags + tests/test_rot_gate_threshold.py; config defaults only.
- C3 wire rot_gate_mode (cold veto + warm ceiling + Case A/_vision_weight/_agreement_*), logging (INFO: rule, mode, T, innov, dt, G, sat, rejects) + NOHS gyro-only fallback.
- C4 weak-confirm + override gyro residual (flags default off).
- C5 kernel per-prefix floors + rot_coast_budget_s + degenerate-gate budget mode (optional).
- C6 controller _mark_all_lost coast_rot_budget_mode (getattr guard; stage only that hunk).
- C7 config flips (contiguous fusion_heuristic hunk; three flags, each with its own validation): rot_gate first, then weak_confirm, then budgets.
- C8 optional: headset imu0 source.
- Staging: src/pose_fusion_heuristic.py, src/imu_data.py, tests/test_pose_fusion_heuristic.py are clean (stage whole); src/controller.py, config/config.yml (user hunks: data root/frame_range 6000-6130, proximity_vis_score_threshold 0.9, mesh alignment ...) and 3 test files are dirty (+471 lines by others): use HEAD copy + edit + `git diff --no-index` patch -> `git apply --cached` (+ `git apply` to worktree); verify `git diff --cached --stat`; never commit .idea/ __pycache__/ analysis run dirs/pkl.

## 8. Validation (subsequence windows; both modes; before/after)
- Windows: walk_medium 0-700 (348; K3 emulation: 3 true vetoes, 348 not reported, correct candidate accepted at dt 0.277 s, 0 good->bad), walk_medium 5200-5450 with warm-up from ~3700, walk_hard 2400-4010 and 4600-5000, static_hard 4300-5919, calm A/B: walk_easy 700-1700, walk_dark 800-1350, static_dark 3000-4000.
- Acceptance: 0 vetoes on GOOD state_ok candidates in calm windows; named bad frames (348, 5333, 5337, 4310, 5872, 4169) not reported as accepted; no good->bad frames vs baseline (bridge-composed); coverage loss <=0.5 pp on calm windows; lock-out <= 6 vetoes / 0.3 s; forced cold starts <= baseline + 2 per window; per-call time <= baseline + 0.5 ms; NOHS: report fail_open count (must be ~0 with the gyro-only fallback on violent windows).
- Stop conditions: any GOOD state_ok veto in a calm window; lock-out > 0.35 s; forced cold start following a GOOD veto; slowdown > 0.5 ms/call.
- Post-hoc logging: veto counts by rule with dt/G/innov/T/mode, override count, forced cold starts, HS/NOHS transitions, fail_open counts.

## 9. Open decisions for the user
1. Deployment target: runs use mocap headset ego-motion (config mocap.enabled true, dev-only). Is the intended runtime with mocap, or must the no-headset path (and imu0 source, C8) be first-class? (Drives whether C8 and the NOHS constants matter.)
2. Rotation budget widening (C5/C6): defer to a later change (recommended) or include now?
3. Approve implementation stage by stage (C1..C8) or C1-C3 first with validation, then the rest?

## Status 2026-09-25 (post C1-C3, decisions by user)
- Deployment target: headset ego-motion from dev mocap headset + fixed mocap room g_world. No-headset case (imu0 gyro / SLAM-VIO headset pose, converged g_world_estimator, fail-open in predict()) = FUTURE WORK (C8), noted in config.yml next to rot_gate_* keys.
- C1-C3 implemented and validated (rot_gate_mode: continuous; default still legacy). Results: bad reported frames walk_medium 17->0 / 13->1, walk_hard 19->1 / 23->1; 0 good->bad; coverage within +-0.8 pt.
- Rotation budget widening implemented as coast_rot_budget_mode: headset (default legacy): tiny coverage gain (+0..1.3 pt) but 6 good->bad frames vs 1 bad->good across walk_hard 0-400/2400-4010 and static_hard 4300-5000 (max error 174 mm). Widened budget lets the degenerate gate accept bare p_pred at moderate gyro. NOT recommended to enable as-is.

## Update 2026-09-25 (evening)
- rot_gate_mode default is now `auto` (continuous when headset ego-motion data is available, else legacy).
- C4 (weak-confirm gyro residual) measured on all 8 full recordings in legacy mode (shadow run) and DROPPED: 74 confirmed pairs with mocap truth, 6 wrong; C4 would catch 5 but delays 8/68 good confirms by a frame, and in the continuous-mode windows no wrong-rotation confirmation remained, so its marginal benefit is ~0. No state older than 0.35 s occurred. Side finding: the legacy confirm rotation budget (15.6 deg + 2000 deg/s * dt) exceeds 180 deg for pair intervals above ~80 ms, i.e. it does not check rotation.
- C5/C6 (coast_rot_budget_mode: headset) stay implemented but off (negative result above). C8 (headset imu0) is future work.
