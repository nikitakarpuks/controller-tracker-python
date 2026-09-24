"""Step 20: export the static calibration constants this study implies (PROPOSAL ONLY -- nothing in the repo is changed).
Gyro K (3x3, e0 = -dt (b + K w_gyro) convention; omega_corrected = (I+K)^-1 (omega_meas - b)): whole-recording fit, gyro regressor, both references, both recordings.
Accel: lever arm (bridge-derived), K_a and b from ALL-time mocap windows (b,K_a fit), lever = bridge."""
import sys, json, os
os.environ["LEVER"] = "bridge"
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step10b_accel_K import fit_bK
out = {"convention": {"gyro": "omega_corrected = (I+K_g)^-1 (omega_factory_corrected_body - b_g)  [after factory T=0 mix+bias and diag(1,-1,-1)]",
                      "accel": "f_used = (I+K_a)^-1 (f_factory_corrected_body - lever_terms(r) - b_a) ; r = accelerometer position in LED/body frame (bridge-derived)"}, "controllers": {}}
runs = {n: Run(n) for n in ("static_dark", "walk_medium")}
for c in CTRLS:
    e = {"gyro_K": {}, "accel": {}}
    for n, run in runs.items():
        prep = prepare(run, c, ego="mocap"); sv, _ = make_steps(run, c, prep); sm = mocap_steps(run, c)
        for lab, st in (("vision", sv), ("mocap", sm)):
            beta, se, _ = fit_body(st, key="omega_g")
            e["gyro_K"][f"{n}/{lab}"] = {"K": np.round(beta[3:].reshape(3, 3), 5).tolist(), "b_rad_s": np.round(beta[:3], 5).tolist()}
    run = runs["static_dark"]; d = run.ctrl[c]
    Ks = [np.array(v["K"]) for k, v in e["gyro_K"].items() if k.startswith("static_dark")]
    Kw = [np.array(v["K"]) for k, v in e["gyro_K"].items() if k.startswith("walk_medium")]
    K_mean = np.mean(Ks, axis=0)
    e["gyro_K_recommended_static_dark_mean_of_refs"] = np.round(K_mean, 5).tolist()
    e["gyro_K_spread_max_abs"] = {"vision_vs_mocap_static_dark": float(np.abs(Ks[0] - Ks[1]).max()), "static_dark_vs_walk_medium": float(np.abs(K_mean - np.mean(Kw, axis=0)).max())}
    # accel (static_dark, all-time, mocap windows, bridge lever)
    prep = prepare(run, c, ego="mocap"); trk_m = track_mocap(run, c); t0 = prep["ts"][prep["ok"]][0]; t_end = prep["ts"][-1]
    gyro_c = corrected(d["gyro"], np.zeros(3), K_mean); AW = AccelWindows(run, c, trk_m, gyro=gyro_c)
    b, K, nw = fit_bK(AW, t0, t_end, 5, True, True)
    e["accel"] = {"lever_arm_bridge_m": np.round(d["lever_bridge"], 5).tolist(), "lever_arm_factory_used_by_main_py_m": np.round(d["lever_factory"], 5).tolist(),
                  "delta_lever_m": np.round(d["lever_bridge"] - d["lever_factory"], 5).tolist(), "K_a": np.round(K, 5).tolist(), "b_a_m_s2": np.round(b, 4).tolist(), "n_windows": nw}
    out["controllers"][c] = e
    print(c, "\n  gyro K (static_dark mean of vision/mocap):\n", np.round(K_mean, 4), "\n  spread", e["gyro_K_spread_max_abs"], "\n  accel lever bridge (mm)", np.round(d["lever_bridge"]*1000, 1),
          "vs factory (mm)", np.round(d["lever_factory"]*1000, 1), "\n  K_a diag", np.round(np.diag(K), 4), " b_a", np.round(b, 3))
with open(OUT_DIR / "proposed_calibration.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved proposed_calibration.json")
