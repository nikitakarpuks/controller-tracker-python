# Data package for testing bias/mocap calibration with Basalt

Recording: `euroc_recording_20260826173103_static_dark` (the "static_dark" recording
memory flags as the one to use for orientation/bias work — Step 3's Finding 7 found
its vision orientation trustworthy, unlike static_hard's).

## What's in here

```
mav0/imu{0,1,2}/data.csv      raw, uncalibrated IMU (EuRoC format: timestamp_ns,
                               w_x,w_y,w_z [rad/s], a_x,a_y,a_z [m/s^2])
                               imu0 = headset, imu1 = left controller, imu2 = right
                               controller. FULL recording (~123.7s), not trimmed —
                               see "Timestamp window" below for how much of it
                               actually has matching vision tracking.

mocap_filtered/{headset,ctrlleft,ctrlright}/data.csv
                               filtered/aligned mocap ground-truth trajectory per
                               device (schema: #timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z).
                               Also FULL recording, same span as the imu*.csv above.
mocap_filtered/*/drift_check/1chunk/drift_check.json, final_offset.txt
                               basalt_mocap_time_sync output for that device: the
                               IMU<->mocap time-offset fit this project actually
                               uses (DRIFT_CHECK_VARIANT="1chunk" in src/mocap_data.py).
                               flat_stats.mean_offset_ns is the FINE (sub-second)
                               offset; final_offset.txt's total is coarse+fine,
                               log-only, never applied directly (see that module's
                               docstring for why).

vision/pose_log_2765frames.csv
                               THE CONSTELLATION TRACKING RESULT — per-accepted-frame
                               LED-constellation pose from this project's own vision
                               pipeline (main.py), in vision's own T_world_ctrl (=
                               T_headsetImu_controllerLedRef, UN-bridged — see
                               configs/config.yml's controllers.*.mocap_bridge_path
                               comment). Columns: timestamp_ns, ctrl_name, qx,qy,qz,qw,
                               px,py,pz, reproj_err_px, inlier_count. Both controllers
                               interleaved (filter by ctrl_name).

                               CONFIRMED (main.py:1023-1030): this is T_world_ctrl
                               written raw, with NEITHER correction applied — not the
                               mocap-bridge above, and not the Rt-transpose fold-in
                               either. That fold-in (main.py:1031-1038, T_world_ctrl.
                               compose(_algo_log_T_ref_ic[ctrl_name])) only happens in
                               a SEPARATE writer gated on debug.algorithm_log_dir,
                               which was null for this recording -- no
                               <ctrl_name>_algorithm_log.csv exists in the source data,
                               confirming that path never ran. Whatever consumes this
                               CSV needs to apply the Rt-transpose itself (direction
                               still unverified -- see algorithm_log_rt_transpose in
                               configs/config.yml's debug section) or you regenerate a
                               proper algorithm_log CSV by rerunning main.py with that
                               option set, which is the format basalt_controller_mocap_
                               calib's input comment actually documents.

configs/
  config.yml                  this project's full pipeline config, for path/convention
                               reference (data.root, mocap.*, controllers.*.config_path
                               etc. all point at the ORIGINAL absolute paths, not this
                               folder — treat as documentation, not a runnable config).
  mocap_calibration_headset.json, controller_{left,right}_calib.json
                               per-device T_imu_marker (mocap-marker <-> device-IMU
                               rigid extrinsic) — only value0.T_imu_marker is meaningful,
                               everything else in these files is unused by this project.
  controller_{left,right}_mocap_bridge_basalt01.json
                               this project's own empirically-fit LED-reference-frame ->
                               mocap-accelerometer-IMU-frame bridge (compare_vision_mocap.py
                               output) — NOT derived from the controller JSON's factory Rt,
                               which was tried and found off by several mm/degrees.
  left_controller_A85K5091630091L.json, right_controller_A85K6081930636R.json
                               full factory controller calibration (LED positions/normals,
                               InertialSensors Rt/bias/mixing/noise/BiasUncertainty for
                               both Gyro and Accelerometer, entry_index=1 is the one this
                               project actually uses — see src/imu_data.py's module docstring
                               for the Id=Undefined vs Id=ICM20602 caveat).
  calibration_basalt.json     camera intrinsics/T_imu_cam currently active in this
                               project's config.yml — itself Basalt-calibration output
                               from an earlier pass, included for reference/consistency.
  prior_basalt_output_controller_{left,right}.yaml
                               CORRECTED ATTRIBUTION: this is NOT basalt_controller_
                               mocap_calib output. Its fields (p_MI, q_MI, q_WG, toff_MI,
                               cost, gradient) don't match what our own tool writes --
                               src/mocap_data.py's docstring lists our tool's output
                               fields as T_imu_marker, T_mocap_world, mocap_time_offset_ns
                               /mocap_to_algorithm_offset_ns, and residual_* stats, which
                               is exactly what controller_{left,right}_calib.json /
                               mocap_calibration_headset.json actually contain. q_WG
                               (world-to-gravity) is the clearest tell: this project's own
                               calib never touches gravity. Most likely this is mocap2gt
                               output -- a separate, pure IMU+mocap spline fit with NO
                               vision/LED involved at all. Its bad cost/gradient (left:
                               cost=1.01e6, gradient=8.65e3; right: cost=3.12e6,
                               gradient=1.05e5) is therefore evidence about THIS
                               CONTROLLER'S RAW IMU DATA specifically -- useful as an
                               independent "is the IMU stream itself well-behaved" signal
                               to compare against basalt_controller_mocap_calib's result,
                               but not a prior data point about that tool.
```

## Timestamp window — how much to actually use

The raw `imu*.csv`/`mocap_filtered/*.csv` files span the **full ~123.7s recording**,
but `vision/pose_log_2765frames.csv` (the constellation tracking result) only covers
its **first ~46.0s** (timestamps `100235449353745` .. `100281421786875`). Everything
past that point in the IMU/mocap files has no corresponding vision pose to calibrate
against, so for a first test you don't need more than:

| file | rows covering the vision-tracked span (+1s pad) | full file |
|---|---|---|
| `mav0/imu0/data.csv` (headset, ~250Hz) | **first ~11,800 rows** | 31,022 |
| `mav0/imu1/data.csv` (left ctrl, ~200Hz) | **first ~9,350 rows** | 24,600 |
| `mav0/imu2/data.csv` (right ctrl, ~200Hz) | **first ~9,380 rows** | 24,671 |
| `mocap_filtered/headset/data.csv` | **first ~6,120 rows** | 15,850 |
| `mocap_filtered/ctrlleft/data.csv` | **first ~6,050 rows** | 15,615 |
| `mocap_filtered/ctrlright/data.csv` | **first ~6,030 rows** | 15,701 |

Files here are left FULL (un-trimmed) since they're small (2-3MB IMU, ~1-2MB mocap)
and trimming risks cutting off history a calibration tool wants for its own
initialization/bootstrap — but if a tool complains about vision/IMU coverage
mismatch, this is the window to point it at.

## Findings from this project's own bias-estimation attempt (relevant context)

This project already tried a from-scratch joint batch least-squares solve for
per-node accel/gyro bias (`bias_estimation_check.py`, `scipy.optimize.least_squares`,
same window as above) before reaching for Basalt. Worth knowing before/while
comparing against whatever Basalt produces:

1. **Step 3 baseline** (memory: `project_imu_fusion_step3_accel.md`): accel
   dead-reckoning with bias assumed zero LOSES to naive constant-velocity
   extrapolation 53-67% of the time (Finding 4) — corroborated independently by
   a CV1 sensor-fusion developer (Finding 10), who said this is expected until
   bias is estimated online rather than fixed at zero.

2. **This session's Step 4 solve (80-node window, both controllers) did not
   converge** using sigma weights derived from the factory JSON's own
   `Noise`/`BiasUncertainty` fields (residual RMS stuck at 73-86σ, effectively
   unmoved over 2000 solver evaluations).

3. **Root cause, found by breaking the residual down by family at x0**: ALL
   THREE physical residual terms (gyro rotation, accel velocity, accel position)
   were 44-150σ out simultaneously — not one buggy term. Converted to physical
   units: gyro rotation mismatch ~0.7-2.7° per gap (matches Step 1's already-known
   ~2-3° figure, not new), and accel velocity mismatch implies a **~4 m/s²
   systematic discrepancy — ~400x bigger than the factory `BiasUncertainty`
   (0.01 m/s²)** used to budget how far bias is allowed to drift. With
   `loss="huber", f_scale=1.0`, residuals this far out sit in huber's
   near-flat linear regime, starving the solver of gradient signal.

4. **After uniformly loosening every sigma 100x** (pure rescale, doesn't change
   relative family weighting, just moves the operating point back into huber's
   quadratic region): bias now moves substantially (~50-80x bigger, e.g. gyro
   bias mean reaching ~7-8x the *original* factory uncertainty), but residual
   RMS only drops 2-4% (left: 0.735→0.719σ, right: 0.860→0.827σ) before
   plateauing — still doesn't converge. **A generously-budgeted smooth bias term
   can only explain a few percent of the original mismatch.**

5. **CONFIRMED (2026-09-02): the dominant error source was the accel/gyro
   LEVER ARM, not IMU bias.** `accel.Rt ∘ gyro.Rt⁻¹`, computed directly from
   the factory JSON's InertialSensors entries -- no mocap, no vision, no new
   recording -- gives a pure translation with negligible rotation (sub-0.2°,
   noise): left `[34.635mm, -77.529mm, 2.809mm]` (84.96mm), right
   `[-33.227mm, -80.239mm, 2.945mm]` (86.90mm), using the `Id=Undefined`
   entries (entry_index=1, what `create_imu_calib_from_config` actually loads
   -- the `Id=ICM20602` entries give translations a few mm off on individual
   axes, e.g. right's z is nearly 2x, so use the Undefined-entry numbers
   above, not whatever the ICM20602 entries say, to stay consistent with the
   rest of the accel/gyro calibration chain). Cross-checked three independent
   ways (factory Rt composition, mocap2gt's raw IMU+mocap spline fit, and a
   vision+mocap LED bundle-adjustment chain using vision/led_detections_
   2765frames.csv) -- all converge on ~83-87mm. `integrate_accel_segment`/
   `integrate_accel_to_position` gained an optional lever-arm correction
   (ω̇×r + ω×(ω×r)) for this -- WIRED IN and unit-tested (matches known
   rigid-body physics exactly), and threaded through `bias_estimation_check.py`
   as a fixed constant (not a solved unknown, per the recommendation not to
   let a sliding-window bias estimator also re-estimate extrinsics).

   RETRACTED: "the ~140° accel-vs-gyro rotation discrepancy is resolved" (an
   earlier claim relayed via Discord, not verified here at the time) does NOT
   hold up. A peer session's fresh, independent derivation (LED-reference-frame
   rotation via vision+mocap bundle adjustment, composed through mocap2gt's own
   T_M_I -- no factory Rt involved at all) still finds ~139-140° off factory Rt
   in BOTH the as-stored and transposed direction. The earlier "resolution"
   conflated two different questions: accel.Rt and gyro.Rt agreeing with EACH
   OTHER (true, sub-0.2°) is not the same as the LED-reference-frame relating to
   either one via a simple Rt.R/Rt.R.T (still unresolved, real ~140° mismatch,
   translation magnitude matches (~83mm mine vs ~86mm factory) but rotation
   doesn't match either convention). Treat as open.

   ALSO CONFIRMED EMPIRICALLY NOT SUFFICIENT ON ITS OWN: re-running Step 4's
   80-node solve with the lever arm applied (same 100x-loosened sigma scaling)
   changed almost nothing -- residual RMS left 0.735→0.719σ (before) vs.
   0.736→0.720σ (after), right 0.860→0.827σ vs. 0.862→0.829σ, still DID NOT
   CONVERGE, recovered bias still ~10-2000x smaller than mocap2gt's cited
   sanity envelope (accel 0.17-0.21 m/s², gyro 0.008-0.014 rad/s). The
   correction's own magnitude on this window (~0.07-0.9 m/s², scales with
   ω which averages ~0.95 rad/s here) is simply too small to explain a
   ~4 m/s²-scale mismatch. Lever arm is real and now modeled, but was NOT
   the dominant error source on this recording -- see finding 6. The ~140°
   LED-frame-vs-Rt mismatch above remains the strongest open candidate for
   what actually is the dominant error source (see finding 6's empirical
   result, which rules out timing too and points back at this).

6. **STALE TIMING CONSTANT -- process concern, RESOLVED as a non-issue
   (2026-09-02).** `bias_estimation_check.py`/`main.py` hardcode a per-controller
   vision<->IMU timestamp lag (-5ms left, -7ms right) whose own comment says
   it's "Stage 1's measured controller<->camera clock offset on THIS
   recording (imu_vision_sync_check.py) -- a single-clip estimate, re-measure
   if this ever runs against different data." Git archaeology: that constant
   was committed 2026-08-04, when config.yml's data.root pointed at
   euroc_recording_20260729173447_still_easy (2026-07-29) -- NOT static_dark
   (recorded 2026-08-26, three-plus weeks later). imu_vision_sync_check.py
   itself was never committed (lost scratch script), so it was never re-run
   against static_dark. A peer session independently measured the REAL offset
   on static_dark twice, by different methods (basalt_controller_mocap_calib
   angular-velocity cross-correlation, and their own LED bundle-adjustment
   time-offset search): ~134ms (left), ~134-147ms (left/right) -- both far
   from the -5/-7ms this project has been silently using through every Step
   1-4 diagnostic on this recording. At this controller's peak angular rates
   (up to 850°/s), a 130-150ms uncorrected offset alone produces 100°+ of
   instantaneous misalignment -- enough to blow out gyro-rotation,
   accel-velocity, AND accel-position residuals simultaneously and
   identically, which is exactly the failure signature Step 4 has shown from
   the start.

   EMPIRICAL RESULT: tested anyway (±134ms, ±peer per-controller 134/147ms,
   both signs, 80-node x0 residual, lever arm included) -- r_gyro improves
   modestly (~6-7%) at +134/+147ms but r_vel/r_pos stay flat or fractionally
   WORSE. A genuine ~140ms timing bug at these angular rates should blow out
   gyro-rotation, accel-velocity, AND accel-position together and shrink them
   together when corrected -- it doesn't. The stale constant is still a real,
   separate bug worth fixing (confirmed via git archaeology above), but it is
   NOT the dominant driver of Step 4's residual. That points back at finding
   5's ~140° LED-frame-vs-Rt mismatch as the stronger remaining candidate: a
   large, fixed rotation error produces a large, roughly fixed-magnitude
   residual largely insensitive to small timing shifts -- matching what's
   actually observed here.

   PROPERLY RE-MEASURED (2026-09-02), no solver needed: re-implemented Stage
   1's lost imu_vision_sync_check.py methodology directly -- angular-velocity
   cross-correlation between vision's per-frame-gap rotation-rate MAGNITUDE
   (Rotation.from_matrix(R0.T@R1).magnitude() / dt) and gyro's own |omega|
   magnitude, swept over lag candidates -50ms..+300ms in 2ms steps, using
   the FULL recording (~2500-2600 frame-gap samples per controller, not an
   80-node window). Using magnitude rather than the raw vector sidesteps the
   axis-convention question (finding 5) entirely: ||R@w|| = ||w|| for any
   proper rotation R, so this measurement is correct regardless of whatever
   _DIAG_FLIP/Rt debate is happening elsewhere -- a genuinely independent
   check of JUST timing.

   RESULT: clean, single-peaked correlation curve for both controllers,
   monotonically decreasing in both directions away from the peak, no
   secondary bump anywhere near the peer's ~134-147ms region. Peak lag:
   -8ms (left, corr=0.8487) / -8ms (right, corr=0.8605) -- both sit right on
   the plateau the CURRENT -5ms/-7ms constants already occupy (corr ~0.848/
   ~0.860 at the current values, indistinguishable from the peak). Correlation
   at +140ms is only ~0.52 (left) / ~0.59 (right) -- dramatically worse, not
   better. **CONCLUSION: the stale-constant concern was a legitimate process
   gap (genuinely unverified, could have been wrong) but turns out to be
   harmless in practice -- the -5ms/-7ms constants are already essentially
   optimal for static_dark too, by luck or because the underlying hardware
   latency is stable across recordings. NO CODE CHANGE NEEDED.** The peer's
   ~134-147ms figure was very likely measuring a different quantity (e.g. a
   mocap-related epoch offset) despite being described as vision<->IMU --
   not verified further since it's moot for this project's own pipeline.

   THIRD INDEPENDENT CONFIRMATION + full explanation (peer session,
   2026-09-02): peer reproduced this finding with a fresh EPnP solve
   straight from raw LED detections (no mocap, no basalt_controller_
   mocap_calib) for ~2440 frames, same magnitude-based cross-correlation
   against raw gyro -- peak at 8ms, corr=0.98 (corr at 134ms only 0.65).
   Three independent methods now agree the vision<->IMU offset is a few ms,
   not ~140ms. Peer also explained the earlier discrepancy: both their
   ~134ms figures (basalt_controller_mocap_calib's cross-correlation and
   their own LED-BA tool) actually correlated vision against MOCAP-derived
   ground truth, not raw gyro -- a genuinely different, real quantity (this
   project's own mocap_calibration_headset.json carries a documented
   ~-361ms mocap_time_offset_ns, confirming mocap has its own separate,
   nontrivial clock-alignment problem distinct from vision/IMU timing --
   see cameras.mocap_calib_path's usage in src/mocap_data.py). Their
   ~134/147ms correction is real and needed for THEIR tool's vision-mocap
   anchor residual (empirically fixed 4.7°/47mm -> 0.13°/2.25mm there) --
   it just isn't the right number for THIS project's vision-vs-gyro
   residual, which is what caused the mismatch. Fully reconciled, no open
   contradiction remains. TIMING QUESTION CLOSED.

7. **BIGGEST EFFECT FOUND SO FAR, still partial (2026-09-02).** Tested the
   peer session's bundle-adjustment-derived LED-frame<->true-IMU rotation as a
   replacement for the current `_Y_FLIP @ Rt.R^(±1)` frame transform in
   `load_and_calibrate_controller_imu` (tested via an ISOLATED script, nothing
   in src/imu_data.py changed -- this swap has real consequences for live
   tracking too and shouldn't be made without more certainty). Caught an
   error in the peer's own report first: they described their rotation as
   "close to IDENTITY (sub-1°)" but their posted quaternion
   (x≈1.0, w≈-0.006, both controllers) is actually a **~179.3° rotation about
   x** (angle = 2·acos(|w|)) -- their prose contradicted their own numbers.
   Testing literal identity (what their prose said) made r_gyro WORSE, as
   expected; testing their ACTUAL matrix gave a real, substantial effect:

   | | r_gyro | r_vel | r_pos |
   |---|---|---|---|
   | left, current | 143.84 | 67.26 | 44.56 |
   | left, peer's actual matrix (both accel+gyro) | **34.78 (-76%)** | 50.18 (-25%) | 38.76 (-13%) |
   | right, current | 150.75 | 86.95 | 83.74 |
   | right, peer's actual matrix (both accel+gyro) | **71.54 (-53%)** | 81.21 (-7%) | 80.34 (-4%) |

   Isolated further: swapping ONLY accel's transform (gyro left at its
   current Rt-based one) gets almost the identical r_vel/r_pos improvement as
   swapping both -- the accel-side gain is real and independent of gyro; the
   r_gyro improvement only appears when gyro's transform is also swapped.

   NOT a clean fix: 34.78σ/71.54σ is still far above 1σ, and the improvement
   is notably asymmetric (left improves much more than right) -- something
   else is still going on (peer flagged their own rotation as an unvalidated
   composition of two independently-fit pieces, not ground truth). Biggest
   lead by far, still open.

   FOLLOW-UP: peer noticed both matrices land within ~1° of the exact clean
   value diag(1,-1,-1) (a precise 180° flip, X unchanged/Y,Z reversed --
   physically unsurprising, a common deliberate PCB mounting choice, not a
   bug) and proposed this as the real physical answer, their composed
   rotation being just a noisy measurement of it. CONFIRMED: swapping in
   diag(1,-1,-1) gives near-identical numbers to their noisy matrix (left
   34.83 vs 34.78σ, right 71.57 vs 71.54σ) -- good, clean explanation for
   the ROTATION value itself. But it REFUTES their asymmetry theory: the
   clean value doesn't shrink the left/right gap at all (same ~35σ vs ~72σ
   split). The asymmetry is real and per-controller, not composition noise
   -- likely related to right moving faster (peak 852°/s vs left's 692°/s)
   and fitting worse elsewhere in the peer's own unrelated methods too.

   IMPLEMENTED (2026-09-02): extended the joint solve with a GLOBAL (not
   per-node) 3-DOF rot_delta -- an axis-angle correction on top of
   _DIAG_FLIP, applied identically to both accel and gyro (one shared
   sensor->body transform, replacing _Y_FLIP @ Rt.R^(±1) entirely), solved
   jointly with bias, initialized at rot_delta=0. See module docstring of
   bias_estimation_check.py and _build_residual_fn's docstring for the
   implementation. gyro_raw_corr/accel_raw_corr (mix+bias corrected, NOT
   axis-transformed) are now what's loaded/windowed instead of pre-transformed
   gyro_body/accel_body, since the transform now depends on solved state.

   RESULT -- the biggest confirmation of this whole investigation: x0
   residual RMS with _DIAG_FLIP alone (before the solve even touches
   rotation) is 0.322σ (left) / 0.600σ (right) -- down from 73-86σ
   pre-_DIAG_FLIP, a ~100-200x reduction, now confirmed via THIS project's
   own residual formulation, not just the peer's externally-composed
   estimate. The joint solve then refines rot_delta to only 0.585°
   (left) / 0.744° (right) beyond _DIAG_FLIP -- same order of magnitude and
   same left<right pattern as the peer's own bundle-adjustment estimate
   (0.83°/1.27°), a second independent method landing in the same
   neighborhood. Solve still technically DID NOT CONVERGE (RMS only
   0.322→0.280 left, 0.600→0.552 right, ~8-13% further reduction, 2000
   evals) -- a real but much smaller remaining gap than anything found
   before this fix (candidates: the still-unfixed stale timing constant
   from finding 6, or genuine residual noise). Recovered bias (b_a std
   ~1-2e-4 m/s², b_g std ~1-6e-5 rad/s) is still nowhere near mocap2gt's
   cited envelope (0.17-0.21 m/s² / 0.008-0.014 rad/s) -- likely reflects a
   real difference in what "bias" means between mocap2gt's whole-recording
   spline fit and this project's short-window batch solve, not necessarily
   a remaining bug.

   CHECKED, not just guessed: peer proposed the gap might be the bias
   random-walk prior being too tight to reach mocap2gt's ~0.2 m/s² over an
   80-node/1.3s window (inferring mocap2gt's own config sigma,
   1.0356e-3 m/s³/√Hz, might be what this solver uses too). Verified against
   the actual code: this solver's bias_walk_sigma_a is `imu_calib.accel.
   bias_uncertainty * _SIGMA_SCALE` = 0.01 * 100 = 1.0 m/s² -- about 1000x
   looser than the peer's guess. At that sigma, reaching 0.2 m/s² via a
   smooth ramp over the window costs ~0.02σ/gap in the random-walk term --
   negligible, not restrictive. So that specific mechanism doesn't explain
   the gap here. More likely explanation: the residual is already down to
   ~0.28-0.55σ after the axis fix, so there's little unexplained signal left
   demanding a large bias correction -- the solver isn't being capped, it's
   just not finding a reason to move bias further.

   CONFIRMED WITH A DIRECT OBSERVABILITY TEST (2026-09-02): does this
   window's data actively REJECT a mocap2gt-scale (~0.2 m/s²) bias, or is it
   just invisible to it? Cheap, no-solver test: evaluate residual_fn at x0
   (baseline RMS 0.3223 left / 0.6003 right) vs. the same point with b_a
   forced to a CONSTANT 0.2 m/s² along each of 6 directions (±x/±y/±z).
   Result: RMS barely moves at all (0.3222-0.3228 left, 0.6001-0.6008
   right -- differences in the 4th decimal place, 1.00x ratio every time).
   **The data doesn't reject a mocap2gt-scale bias -- it's essentially free
   to add, confirming this is a genuine observability gap, not the prior
   capping anything and not a sign the true bias is actually ~0.**

   MECHANISM (more precise than "window too short"): each gap here is only
   ~15-20ms. integrate_accel_to_position's own docstring already documents
   why short-horizon integration is deliberately robust to calibration
   error -- a 0.2 m/s² bias integrated TWICE over 15ms works out to a
   sub-millimeter position perturbation, and a single-integration velocity
   perturbation of only ~0.004 m/s per gap -- both tiny next to sensor noise
   and real inter-frame motion. This is a property of each constraint's GAP
   LENGTH, not the window's total length: a longer window (more nodes)
   would NOT by itself fix this, because every individual r_vel/r_pos
   constraint still only "sees" bias over its own ~15-20ms slice, no matter
   how many such slices are chained together. What mocap2gt does
   differently, structurally: it fits ONE smooth trajectory model against
   the WHOLE recording, so a bias has to stay consistent with predicting
   motion correctly across thousands of gaps simultaneously against one
   shared trajectory -- this solver's independent per-gap velocity states
   have no such long-baseline anchor forcing consistency. This is exactly
   what the peer's original recommendation (a proper sliding-window/online
   filter, not an isolated short batch) was pointing at from the start --
   now with a concrete mechanism, not just an appeal to authority. Axis-
   convention question closed; solve convergence and the bias-magnitude gap
   remain open, but the gap's mechanism is now well understood, not guessed.

8. **PROPAGATED to the shared loader (2026-09-02)** -- everything above was,
   until now, confined to `bias_estimation_check.py`'s own local reimplementation
   of the raw-IMU load. `load_and_calibrate_controller_imu` in `src/imu_data.py`
   -- the canonical loader `main.py`'s LIVE tracking (`imu.enabled`) and every
   other diagnostic script (`gyro_preint_check.py`, `accel_short_horizon_check.py`,
   `accel_sign_check.py`, `accel_occlusion_check.py`, `accel_jerk_check.py`,
   `motion_dynamics_check.py`, `orientation_gravity_correction_check.py`,
   `compare_vision_mocap.py`, etc. all call -- was still silently using the OLD,
   now-confirmed-wrong per-sensor `_Y_FLIP @ Rt.R^(±1)` transform this whole time.

   Before changing shared, live-tracking-affecting code, re-verified decisively:
   that module's own docstring claimed the old transform was independently
   validated (gyro: ranked #1 of 16 transpose x sign-flip candidates via
   cross-correlation against vision, imu_vision_sync_check.py; accel:
   gravity-direction self-consistency, imu_accel_exhaustive_search.py) -- a
   real, direct conflict with this session's diag(1,-1,-1) finding that needed
   resolving, not assuming away. Ran a decisive, ROTATION-sensitive (not
   magnitude-only) re-check over the FULL recording (2509/2610 gyro-vs-vision
   rotation-error samples, left/right, 8 candidate transforms): diag(1,-1,-1)
   alone wins by nearly an order of magnitude -- median error 0.342°/0.510°,
   vs. 2.644°/2.480° for the old documented "best" transform, vs. 2.5-4.8° for
   every other candidate tried (both Rt directions alone, various flip x Rt
   combinations, identity). Neither imu_vision_sync_check.py nor
   imu_accel_exhaustive_search.py were ever committed (lost scratch scripts,
   confirmed via git log), so it's unknown which recording that original
   validation ran against -- but exactly like the lag_ns constant (finding 6),
   the most likely explanation is it was validated on an older recording and,
   unlike lag_ns, does NOT hold up on static_dark.

   `load_and_calibrate_controller_imu` now applies `_DIAG_FLIP` directly (same
   transform for both gyro and accel, no Rt rotation involved at all) --
   verified end-to-end to reproduce the exact 0.342°/0.510° result. The old
   `_Y_FLIP`/`_gyro_body_transform`/`_accel_body_transform` are gone (only
   caller was this function). `bias_estimation_check.py` still can't call this
   function directly -- rot_delta needs the raw, pre-transform stream to solve
   a correction on top of `_DIAG_FLIP` -- but its comments now describe the
   current (not superseded) state accurately. src/imu_data.py's module
   docstring rewritten to document this supersession and the evidence, rather
   than silently dropping the old (now-wrong) claim.

9. **`algorithm_log_rt_transpose` RESOLVED (2026-09-02) -- was never going to
   converge either way.** main.py's `basalt_controller_mocap_calib` input
   (`debug.algorithm_log_dir`) computed R_ref_ic (reference-frame ->
   controller-IMU rotation, needed for T_Ih_Ic) from the controller config's
   InertialSensors `Rt`, with an unverified transpose direction (`Rt.R` vs
   `Rt.R.T`) gated by this flag. Checked directly: `_DIAG_FLIP` -- what
   `load_and_calibrate_controller_imu` uses for gyro_body, which by
   construction IS gyro data expressed in this same reference frame, so it's
   exactly what R_ref_ic needs -- is `~137-139°` away from BOTH `Rt.R` and
   `Rt.R.T` for both controllers. Neither transpose direction was ever going
   to be right; this is the SAME ~140° mismatch found independently twice
   elsewhere in this investigation (finding 5's peer LED-BA-vs-Rt comparison,
   and finding 7's decisive full-recording check) -- a third independent
   confirmation that Rt's rotation component just isn't the sensor<->
   reference-frame relationship for this hardware; only its translation
   (used for the lever arm) is. Fixed: `_algo_log_T_ref_ic` now uses
   `_DIAG_FLIP` directly, no Rt involved, no transpose ambiguity.
   `algorithm_log_rt_transpose` removed from `config.yml`/`main.py` --
   the ambiguity it gated no longer exists. Verified end-to-end: ran main.py
   with `debug.algorithm_log_dir` set (50-frame smoke test), confirmed clean
   output with no crash, then regenerated the full 2765-frame `pose_log`/
   `led_detections` CSVs the smoke test had temporarily overwritten (verified
   byte-identical to the pre-smoke-test backup afterward).

10. **`prior_basalt_output_controller_*.yaml` cost/gradient CHECKED, not a
    modeling bug (2026-09-02).** This file is mocap2gt output (finding 5's
    corrected attribution) -- it fits its own marker<->IMU transform (T_M_I)
    from scratch, purely from raw IMU + mocap, NEVER touching this project's
    factory `Rt`/axis-convention question at all. So the ~140°/`_DIAG_FLIP`
    bug findings 5/7/9 uncovered can't be why mocap2gt's right-controller fit
    is worse (cost 3.12e6 vs left's 1.01e6, ~3x; gradient 1.05e5 vs 8.65e3,
    ~12x) -- that's a different mechanism entirely.

    Checked directly against this recording's own mocap quality reports
    (`mocap_filtered/{ctrlleft,ctrlright}/mocap_quality_{pre,post}_refit_*.txt`,
    not previously read): right controller's marker tracking is genuinely
    noisier than left's, independent of anything IMU/vision-related.
    Post-refit (what's actually loaded): right has **10 flagged pose jumps**
    with physically-impossible instantaneous angular rates (4560-18141°/s --
    clearly marker-occlusion/reacquisition glitches) vs left's **zero**; one
    lands at t≈30.9s, within the full recording's span mocap2gt would fit
    over (outside the first-~46s window this project's own vision/bias
    diagnostics use, so it doesn't affect THIS project's own joint solve).
    Right's max per-marker error is 16.13mm vs left's 3.85mm (4x). Pre-refit
    numbers are independently worse across the board too (46.6% of frames
    flagged vs left's 38.0%; p99 error 0.145m vs 0.086m; max 0.824m vs
    0.205m) -- not a refit artifact, right's raw marker tracking on this
    specific recording is measurably noisier from the start. Consistent with
    every other "right fits worse than left" signal in this investigation
    (this project's own axis-check: 0.510° vs left's 0.342°, ~1.5x; the
    joint solve: 0.552σ vs left's 0.280σ, ~2x; the peer's own peak angular
    rate: 852°/s vs left's 692°/s). **CONCLUSION: a real, recording-specific
    mocap data-quality difference, not a modeling bug in mocap2gt, this
    project's residual, or the peer's LED-BA tool.** CLOSED.

11. **REAL-OCCLUSION-GAP VALIDATION (2026-09-02) -- the decisive test, using
    actual tracking-loss gaps instead of synthetic short ones.** Every prior
    bias test in this investigation (findings 4, 7's observability check)
    used adjacent-vision-frame gaps (~15-20ms), where a bias error shrinks
    to a sub-mm effect purely from short integration horizon (see
    `integrate_accel_to_position`'s own docstring) -- not evidence bias is
    unimportant generally, just unobservable at THAT horizon. This project's
    own recording has real tracking-loss gaps (occlusion / out of camera
    view) -- `static_dark`'s `_MAX_PAIR_DT_S=0.15s` threshold finds 7 on
    left_controller (155ms-1076ms) and 6 on right_controller (155-766ms),
    spanning the whole ~46s tracked window. These are 10-70x longer than
    the gaps everything else in this investigation used -- exactly the
    horizon where bias-induced drift (dt² for position, dt for velocity)
    should actually become visible, if it's real.

    Method: BLIND dead-reckoning through each real gap [t0,t1] -- v0 from a
    backward finite difference (only data available in real time), R1
    predicted via gyro alone (`R0 @ integrate_gyro_segment`), p1 predicted
    via accel double-integration using the GYRO-PREDICTED R1 (not vision's
    true R1) as the SLERP endpoint -- genuinely no peeking at the ground
    truth during the gap, unlike the original short-gap residual checks
    (which legitimately use both true endpoints to isolate accel error).
    Compared against vision's actual recovered pose at t1, and against the
    same naive constant-velocity baseline Step 3's Finding 4 used. Tested 7
    bias candidates: zero, and mocap2gt-scale (0.2 m/s² accel / 0.011 rad/s
    gyro) along each of ±x/±y/±z.

    HEADLINE RESULT -- directly reverses Step 3's original finding: Step 3
    found accel dead-reckoning LOST to naive constant-velocity 53-67% of the
    time (bias assumed zero, pre-any-of-this-session's-fixes). On these 13
    real occlusion gaps, with the axis-convention fix + lever arm now in
    place (bias still fixed at zero), accel dead-reckoning **BEATS naive
    constant-velocity 9/13 times (69%)** -- right_controller 6/6 (100%),
    left_controller 3/7 (43%). This session's fixes turned accel prediction
    from a net loser into a net winner on real, substantial occlusion gaps.

    BIAS'S ROLE, CONFIRMED ON REAL DATA (not just short-gap residuals):
    rotation error (0.5-7° across all 13 gaps) varies by <1° across every
    bias candidate tested. Position win/loss verdict against naive is
    **completely unanimous across all 7 bias candidates in 11/13 gaps**
    (either all 7 beat naive or all 7 lose) -- the 2 exceptions are
    razor-thin, sub-centimeter flips around the naive baseline, noise-level,
    not a real bias effect. Bias choice, even at mocap2gt's cited scale in
    any tested direction, essentially never changes whether dead-reckoning
    succeeds on a given real gap. This reconfirms finding 7's observability
    conclusion on genuinely long, real gaps, not just an artifact of the
    short synthetic ones -- bias is not the lever for THIS project's
    dead-reckoning-through-occlusion use case on this recording.

    ONE REAL OUTLIER, not a bias issue: the single longest gap (1076ms,
    left_controller, t=43.18s) loses badly to naive for every bias
    candidate (~2.0-2.1m error vs naive's 0.74m) -- unanimous across all 7
    candidates just like the wins, so not bias-related. Most likely the
    SLERP-based orientation-interpolation model (designed for ~15-30ms
    gaps, per its own docstring) or the single backward-finite-difference
    v0 estimate breaking down over a full second, or genuinely complex real
    motion during that specific occlusion. Worth a closer look if this
    recording's occlusion-gap performance needs to be robust to >1s losses,
    but does not change the overall 69% win-rate conclusion or the
    bias-doesn't-matter conclusion.
