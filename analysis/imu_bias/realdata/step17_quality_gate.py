"""Step 17: does gating the estimator input by vision quality (n_inliers / error_px) matter?  Estimators are fed steps from frames passing the
gate; EVALUATION pairs are fixed (strong (8,0.5) frames, t>60 s, gaps 1 s and 2 s) so settings are comparable. K-corrected gyro (K from first 60 s)."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from fastgap import integrate_with_C, fast_error_deg
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step6b_gyro_fast import gap_pairs, Cache
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
GATES = [("all frames (n>=0, any err)", (0, 1e9)), ("loose (n>=5, err<=1.0)", (5, 1.0)), ("default (n>=8, err<=0.5)", (8, 0.5)), ("strict (n>=12, err<=0.3)", (12, 0.3)), ("very strict (n>=16, err<=0.2)", (16, 0.2))]
for c in CTRLS:
    d = run.ctrl[c]; prep_eval = prepare(run, c, ego="mocap")
    t0 = prep_eval["ts"][prep_eval["ok"]][0]; split = t0 + int(60e9)
    st0, _ = make_steps(run, c, prep_eval); bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
    gy = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
    pairs = {g: gap_pairs(prep_eval, g) for g in (1.0, 2.0)}; caches = {g: Cache(run, c, prep_eval, gy, pairs[g]) for g in pairs}
    zero = {g: caches[g].errors(lambda t: np.zeros(3)) for g in pairs}
    print(f"\n[{name}/{c}]  paired mean-error change vs zero bias (deg) on fixed test pairs t>60 s  [negative = better]; median error")
    for lab, gate in GATES:
        prep = prepare(run, c, ego="mocap", strong=gate)
        steps, n_out = make_steps(run, c, prep, gyro=gy)
        out = []
        for mk_name, mk in (("worldRLS tau=80", lambda: WorldRLS(tau_s=80)), ("ratioEMA tau=40", lambda: RatioEMA(tau_s=40))):
            T, B = run_estimator(mk(), steps); cells = []
            for g in (1.0, 2.0):
                e = caches[g].errors(lambda t, T=T, B=B: b_at(T, B, t)); m = (caches[g].times >= split) & np.isfinite(e) & np.isfinite(zero[g])
                cells.append(f"{g}s: {np.mean(e[m]-zero[g][m]):+.3f} (med {np.median(e[m]):.2f})")
            out.append(f"{mk_name}: " + "  ".join(cells))
        print(f"   {lab:32s} steps {len(steps):5d} (|e0|-gated {n_out:3d})   " + "   |   ".join(out))
