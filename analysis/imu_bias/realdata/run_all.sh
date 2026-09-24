#!/bin/bash
# Reproduction order for the IMU-bias real-data study (investigator R). Repo-relative paths are resolved inside common.py (it chdir's to the repo).
# Inputs: visualization/evaluate_2026-09-22/<rec>/{vision_pose.csv,config.yml} (raw vision + run config) and the raw recordings under ~/Downloads/recordings-aug26.
# Run from anywhere:  bash analysis/imu_bias/realdata/run_all.sh
set -e
cd "$(dirname "$0")"

# ---- 0. unit tests (exact-value; synthetic on real timestamps) ----
python3 test_estimators.py
python3 test_accel_tools.py

# ---- 1. baseline + residual anatomy (static_dark) ----
python3 step1_baseline.py static_dark            # reproduces README finding 9 (gyro-vs-vision ~0.3 deg median)
python3 step1b_residual_anatomy.py static_dark   # lag-1 autocorrelation (-0.44..-0.48 => differenced white noise), rate dependence, imu0 lag scan
python3 step3c_timing_scan.py static_dark        # gyro<->vision time shift per quarter (optimum 0 ms, no drift)
python3 step3d_mocap_quality.py static_dark      # mocap-orientation reference quality
python3 step3e_scaling.py static_dark            # accumulated world-frame residual S(T) ~ sqrt(T) (not telescoping, not linear)

# ---- 2. what explains the residual: K (scale/misalignment) not bias ----
python3 step3f_scale_misalign.py static_dark
python3 step4b_regressor_check.py static_dark    # errors-in-variables artifact (vision-noisy regressor -> spurious negative K); gyro regressor fixes it
python3 step4a_payoff_KB.py static_dark          # (kept for the erroneous-K comparison; uses vision-derived regressor)
python3 step5_bias_after_K.py static_dark
python3 step19_K_decomposition.py

# ---- 3. gyro-bias oracle and causal estimators ----
python3 step3_oracle.py static_dark
python3 step6b_gyro_fast.py static_dark mocap own60
python3 step6b_gyro_fast.py static_dark imu0 own60           # deployable headset ego (imu0, bias calibrated on first 10 s vs mocap)
python3 step6b_gyro_fast.py walk_medium mocap static_dark    # K from static_dark -> walk_medium
python3 step6b_gyro_fast.py walk_medium mocap own60
python3 step6b_gyro_fast.py walk_medium imu0 static_dark
python3 step7_imu0.py static_dark
python3 step17_quality_gate.py static_dark
python3 step16_outlier_swap.py

# ---- 4. accelerometer ----
python3 step8_accel_explore.py static_dark       # literal per-frame scheme SNR
python3 step9_accel_oracle.py static_dark
python3 step10_accel_payoff.py static_dark
python3 step10b_accel_K.py static_dark
python3 step14_accel_budget.py static_dark
python3 step14b_lever.py static_dark             # lever-arm regression (factory lever) -> dr ~ 0.1 m
LEVER=bridge python3 step9_accel_oracle.py static_dark
LEVER=bridge python3 step10b_accel_K.py static_dark
LEVER=bridge python3 step14b_lever.py static_dark
LEVER=bridge python3 step11_accel_causal.py static_dark
LEVER=bridge python3 step11_accel_causal.py walk_medium
python3 step11_accel_causal.py walk_medium
LEVER=bridge python3 step13_transfer.py static_dark walk_medium
python3 step13_transfer.py static_dark walk_medium
LEVER=bridge python3 step15_real_gaps_accel.py static_dark
LEVER=bridge python3 step15_real_gaps_accel.py walk_medium
LEVER=bridge python3 step18_accel_bias_variability.py static_dark

# ---- 5. figures ----
python3 make_figures.py
