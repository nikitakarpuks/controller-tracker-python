"""Step 16: robustness. Inject an IDENTITY SWAP episode into the left controller's vision stream (frames t0+70..t0+74 s take the RIGHT
controller's pose at the same timestamps), then measure how far each causal estimator's bias output is pushed away from its clean-run
trajectory, with and without the gates:  (a) per-step |e0| > 3 deg gate  (b) none.  Gyro estimators (K-corrected) and the accel RLS
(b,K_a) with a vision position-jump gate (>8 m/s implied speed inside the window) are tested."""
import sys, copy
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step11_accel_causal import AccelRLS
name = "static_dark"; run = Run(name)
c, o = "left_controller", "right_controller"
d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); prep_o = prepare(run, o, ego="mocap")
t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9)
st0, _ = make_steps(run, c, prep); bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
# --- inject swap: frames in [t0+70, t0+74] use the other controller's orientation/position where the same timestamp exists
lo, hi = t0 + int(70e9), t0 + int(74e9)
other = {int(t): i for i, t in enumerate(prep_o["ts"]) if prep_o["ok"][i]}
prep_sw = copy.deepcopy(prep); n_sw = 0
for i, t in enumerate(prep["ts"]):
    if lo <= t <= hi and int(t) in other:
        j = other[int(t)]; prep_sw["R_wc"][i] = prep_o["R_wc"][j]; prep_sw["p_hc"][i] = prep_o["p_hc"][j]; n_sw += 1
print(f"[{name}/{c}] swap episode t0+70..74 s: {n_sw} frames replaced by {o}'s poses")
def steps_of(pr, gate):
    st, n_out = make_steps(run, c, pr, gyro=gyro_c, outlier_deg=(3.0 if gate else 1e9))
    return st, n_out
for label, mk in (("worldRLS tau=40", lambda: WorldRLS(tau_s=40)), ("ratioEMA tau=40", lambda: RatioEMA(tau_s=40)), ("feedbackEMA tau=40", lambda: FeedbackEMA(tau_s=40)),
                  ("worldRLS tau=10", lambda: WorldRLS(tau_s=10)), ("windowMedian W=20", lambda: WindowMedian(20))):
    Tc, Bc = run_estimator(mk(), steps_of(prep, True)[0])
    for gate in (True, False):
        st, n_out = steps_of(prep_sw, gate)
        T, B = run_estimator(mk(), st)
        common = np.intersect1d(T, Tc)
        Bi = B[np.searchsorted(T, common)]; Bci = Bc[np.searchsorted(Tc, common)]
        dev = np.linalg.norm(Bi - Bci, axis=1)
        after = common > hi
        print(f"   {label:20s} gate={'3deg' if gate else 'none':5s} gated steps {n_out:3d}: max |b - b_clean| during/after swap {dev.max():.4f} rad/s   at +20 s after {dev[after][min(len(dev[after])-1, 1500)]:.4f}   rms over t>swap {np.sqrt(np.mean(dev[after]**2)):.4f}")
# --- accel RLS (b,K_a): swap positions/orientations, window gate on implied speed
trk_c = track_vision(run, c, prep); trk_s = track_vision(run, c, prep_sw)
def rls_series(trk, gate, tau=40, W=5, hop=1.0):
    AW = AccelWindows(run, c, trk, gyro=gyro_c); est = AccelRLS(tau, True); T, P = [], []
    ts = trk.ts; sp = np.linalg.norm(np.diff(trk.P, axis=0), axis=1) / (np.diff(ts) / 1e9); bad_t = ts[1:][(sp > 8.0) & (np.diff(ts) < 0.05e9)]
    for t in np.arange(t0 + int(W * 1e9), prep["ts"][-1], int(hop * 1e9)):
        r = AW.components(int(t - W * 1e9), int(t))
        skip = gate and np.any((bad_t >= t - int(W * 1e9)) & (bad_t <= t))
        if r is not None and r[3] > 0.7 and not skip: est.update(r[0], r[1], r[2])
        T.append(t); P.append(np.concatenate([est.current()[0], est.current()[1].ravel()]))
    return np.array(T), np.array(P)
Tc, Pc = rls_series(trk_c, False)
for gate in (True, False):
    T, P = rls_series(trk_s, gate)
    dev = np.linalg.norm(P[:, :3] - Pc[:, :3], axis=1); after = T > hi
    print(f"   accel RLS b,K_a tau=40 gate={'speed>8m/s' if gate else 'none':10s}: max |b - b_clean| {dev.max():.4f} m/s^2   at +20 s after {dev[after][20]:.4f}   rms over t>swap {np.sqrt(np.mean(dev[after]**2)):.4f}   (clean |b| ~ {np.linalg.norm(Pc[-1,:3]):.3f})")
