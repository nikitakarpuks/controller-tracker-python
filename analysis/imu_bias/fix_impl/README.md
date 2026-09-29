# Implementation of the loader + lever-arm fixes (drafted 2026-09-23, UNCOMMITTED)

Design/derivation: ../fix_design.md (Change 2 superseded, see its header). Independent review: found the bridge
convention slip in the first design. Validation A/B: ../fix_validation/REPORT.md.

## What changed (working tree)
- src/imu_data.py
  - `load_and_calibrate_controller_imu(..., factory_corrected_input=True, accel_scale=ACCEL_DRIVER_SCALE)`: no second
    mix+bias on the recorded (already driver-corrected) CSVs; accel x 0.980665; `factory_corrected_input=False,
    accel_scale=1.0` reproduces the old chain exactly. `imu_loader_kwargs(imu_cfg)` reads config.
  - `accel_lever_arm_body(calib)` = accel.T_rt.inverse().t - gyro.T_rt.inverse().t (= -R_acc^T t_acc; the factory
    translation read in the body frame). Old value was the raw t_acc (sensor frame): right length, ~95 deg wrong direction.
  - `median_quiet_accel_magnitude`, `ACCEL_QUIET_MAGNITUDE_BAND = (9.70, 10.02)`: stream sanity guard.
  - module docstring: "RECORDED CONTROLLER IMU STREAMS ARE ALREADY FACTORY-CORRECTED" section; correct() warns.
- main.py: passes the loader kwargs, uses accel_lever_arm_body, logs lag / loader switches / lever arm at startup,
  warns at startup if the quiet |accel| is outside the band. (main.py also contains an UNRELATED uncommitted
  thesis-figure block from another session -- do not stage it with these hunks.)
- config/config.yml: `imu.recorded_stream_factory_corrected: true`, `imu.accel_driver_scale: 0.980665`.
  (config.yml also has other uncommitted edits from other sessions.)
- Offline scripts switched to accel_lever_arm_body: imu_trust_analysis, visualize_position_orientation,
  pose_fusion_heuristic_jump_check, visualize_pose_fusion, visualize_pose_fusion_validation, bias_estimation_check.
  bias_estimation_check.py and check_controller_imu_axis.py build their own correct() chain: annotated, not changed.
- tests/test_imu_loader.py (17 tests) and tests/test_accel_lever_arm.py (12 tests): exact-value, incl. the spinning-body
  "prove the bug, prove the fix" test through the project's integrate_accel_to_position, the bridge-convention tests
  (a wrong -R_b^T t_b design would fail them), a mocap-free factory-vs-bridge cross-check, the legacy-switch bit
  exactness, and the quiet-|accel| band on all real recordings (new stream inside, old chain outside).

## Results
- Tests: 556 run, 550 pass; 6 errors are the pre-existing missing data/cameras/kb4_calib.json ones.
- Held-out gap prediction (A/B, fix_validation): accel position error at a 1 s gap 399 -> 137 mm (left), 548 -> 234 mm
  (right); real tracking-loss gaps 221 -> 110 mm, 192 -> 72 mm; gyro rotation error at 1 s 1.71 -> 1.36 deg.
- End-to-end (main.py, static_dark frames 2900-3500, fused and raw vision vs mocap, same timestamps, first 60 frames
  skipped): no crash, no tracking loss, mean position error changed by +0.01 / +0.02 mm, i.e. UNCHANGED in normal
  tracking (compare_runs.py).
- Startup guard: silent with the fix (both controllers), fires with legacy settings (median |a| 10.27 / 10.25).
- How much the live pipeline leans on the IMU (old Sep-22 runs): only 0.04-0.95 % of frames are IMU-coasted and gaps
  >= 0.1 s number 1-17 per recording. The fixes improve IMU prediction quality; they do not change reported accuracy
  while vision is present.

## Not done / open
- Downstream tuned constants (coast_trust_*, jump thresholds) were fit on the old stream; expected second-order,
  not re-tuned. A live comparison on windows with real losses/reacquires (walk_medium, static_medium) is still open.
- Per-session gyro/accel bias, gyro gain K, right-controller accel anisotropy: untouched.
- Oracle biases were measured on the OLD stream; re-measure on the corrected one.
- Recorder provenance strongly supported, not proven (no git metadata for the recording build).
