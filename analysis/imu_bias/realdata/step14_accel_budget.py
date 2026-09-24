"""Step 14: error budget of accel position dead-reckoning (static_dark, t>60 s, gaps 0.5/1 s). Swap ONE ingredient at a time for ground truth:
   R(t): gyro-integrated (K-corrected)  vs  mocap orientation;  v0: vision LS fit vs mocap;  a-bias: zero vs const(train mocap windows).
Then the residual gap to zero error is what accelerometer scale/misalignment, gravity, lever arm, vision-position noise etc. leave."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step10b_accel_K import fit_bK

name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
    t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9)
    st0, _ = make_steps(run, c, prep); bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
    gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
    AW = AccelWindows(run, c, trk_m, gyro=gyro_c); b_tr, _, _ = fit_bK(AW, t0, split, 5, True, False)
    tsv = trk_v.ts; t = d["t"]
    def run_case(gap, R_src, v_src, bias, target="vision"):
        errs = []
        for i in range(0, len(tsv), 3):
            ta = tsv[i]
            if ta < split: continue
            kk = np.searchsorted(tsv, ta + gap * 1e9); cs = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
            if not cs: continue
            k = min(cs, key=lambda k: abs(tsv[k] - (ta + gap * 1e9)))
            if abs(tsv[k] - (ta + gap * 1e9)) > max(0.006, 0.35 * gap) * 1e9 or np.max(np.diff(tsv[i:k + 1])) > 0.1e9: continue
            tb = tsv[k]; i0, i1 = np.searchsorted(t, ta), np.searchsorted(t, tb)
            ts_ = np.concatenate(([ta], t[i0:i1][(t[i0:i1] > ta) & (t[i0:i1] < tb)], [tb])).astype(np.int64)
            if len(ts_) < 4: continue
            g_ = np.stack([np.interp(ts_, t, gyro_c[:, q]) for q in range(3)], 1); f_ = np.stack([np.interp(ts_, t, d["accel"][:, q]) for q in range(3)], 1)
            dts = np.diff(ts_) / 1e9
            if R_src == "gyro":
                R = np.empty((len(ts_), 3, 3)); R[0] = trk_v.R[i]
                for q in range(len(dts)): R[q + 1] = R[q] @ Rotation.from_rotvec(0.5 * (g_[q] + g_[q + 1]) * dts[q]).as_matrix()
            else:
                R = trk_m.R_at(ts_)
            v0 = trk_v.v_at(ta - int(0.03e9), 0.03, 3) if v_src == "vision" else trk_m.v_at(ta, 0.04, 3)
            if v0 is None: continue
            p0 = trk_v.P[i] if target == "vision" else trk_m.P[np.searchsorted(trk_m.ts, ta)]
            pb = trk_v.P[k] if target == "vision" else trk_m.P[np.searchsorted(trk_m.ts, tb)]
            fc = f_ - _lever_arm_correction(ts_, ts_, g_, d["lever"])
            aw = np.einsum("nij,nj->ni", R, fc - bias) + G_ABS
            v = np.empty_like(aw); v[0] = v0; p = p0.copy()
            for q in range(len(dts)):
                v[q + 1] = v[q] + 0.5 * (aw[q] + aw[q + 1]) * dts[q]; p = p + 0.5 * (v[q] + v[q + 1]) * dts[q]
            errs.append(np.linalg.norm(p - pb))
        return np.array(errs)
    print(f"\n[{name}/{c}] median position error (m), t>60 s   [target: vision position unless noted]")
    print(f"   {'case':62s}{'0.5 s':>10s}{'1.0 s':>10s}")
    cases = [("R=gyro-integrated, v0=vision, b=0 (as deployed)", "gyro", "vision", np.zeros(3), "vision"),
             ("R=gyro-integrated, v0=vision, b=const(train)", "gyro", "vision", b_tr, "vision"),
             ("R=MOCAP, v0=vision, b=0", "mocap", "vision", np.zeros(3), "vision"),
             ("R=MOCAP, v0=vision, b=const(train)", "mocap", "vision", b_tr, "vision"),
             ("R=MOCAP, v0=MOCAP, b=0", "mocap", "mocap", np.zeros(3), "vision"),
             ("R=MOCAP, v0=MOCAP, b=const(train)", "mocap", "mocap", b_tr, "vision"),
             ("R=MOCAP, v0=MOCAP, b=const(train), TARGET=mocap position", "mocap", "mocap", b_tr, "mocap"),
             ("R=MOCAP, v0=MOCAP, b=0, TARGET=mocap position", "mocap", "mocap", np.zeros(3), "mocap")]
    for lab, Rs, vs, b, tg in cases:
        print(f"   {lab:62s}" + "".join(f"{np.median(run_case(g, Rs, vs, b, tg)):10.3f}" for g in (0.5, 1.0)))
