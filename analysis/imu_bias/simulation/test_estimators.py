"""Known-answer tests for the ESTIMATORS: noise-free constant / zero / ramp / sign-flipped bias must be
recovered (or not invented). Run after test_sim.py passes."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from simlib import *; from estlib import *
from src.imu_data import integrate_gyro_segment

res = []
def check(name, ok, detail=""):
    res.append(dict(name=name, ok=bool(ok), detail=detail)); print(("PASS " if ok else "FAIL ") + name + "  " + detail)

bg = np.array([0.01, -0.02, 0.015]); ba = np.array([0.10, -0.05, 0.08])
base = dict(noise_g=0, noise_a=0, outlier_frac=0, sig_rot_deg=0, sig_pos_mm=0)
def mk(**kw):
    d = simulate("static_dark", "right", Scenario(**{**base, **kw}), t_max_s=60)
    d.sig_rot = 1e-4; d.sig_pos = 1e-4
    return d
d = mk(b_g0=bg, b_a0=ba); iv = Intervals(d); K = len(d.t_v_ns)

# E1 shared gyro interval == project integrate_gyro_segment (b=0) and first-order Jacobian scales as b^2
mx = 0
for k in range(1, 100):
    ref = integrate_gyro_segment(d.t_imu_ns, d.gyro, d.t_v_ns[k - 1], d.t_v_ns[k]); mx = max(mx, np.abs(ref - iv.dR[k]).max())
check("E1a dR0 identical to src.imu_data.integrate_gyro_segment", mx < 1e-12, f"max diff {mx:.1e}")
errs = {}
for sc in (1e-3, 1e-2):
    w = 0
    for k in range(1, 150):
        b = np.array([1, -1, 1.]) / np.sqrt(3) * sc
        ex = integrate_gyro_segment(d.t_imu_ns, d.gyro - b, d.t_v_ns[k - 1], d.t_v_ns[k])
        w = max(w, np.linalg.norm(Log(ex.T @ (iv.dR[k] @ Exp(-iv.J[k] @ b)))))
    errs[sc] = w
check("E1b first-order bias Jacobian: error ~ b^2 (b=0.01: <1e-4 rad; b=1e-3 100x smaller)", errs[1e-2] < 1e-4 and errs[1e-3] < errs[1e-2] / 30,
      f"b=1e-3: {errs[1e-3]:.1e}, b=1e-2: {errs[1e-2]:.1e} rad (worst interval, up to 1.4 s)")

# E2 constant-bias exact recovery (noise-free)
for mode, tau in (("dt_weighted", 1e5), ("naive_ema", 5.0), ("median", 5.0)):
    e = np.linalg.norm(run_A_gyro(d, iv, mode, tau)[-1] - bg)
    check(f"E2 A_gyro[{mode}] recovers constant b_g", e < 2e-4, f"err {e:.1e} rad/s (|b|={np.linalg.norm(bg):.3f})")
bc, sc = run_C_gyro(d, iv, W=20, every=10)
check("E2 C_gyro recovers constant b_g", np.linalg.norm(bc[-1] - bg) < 5e-4, f"err {np.linalg.norm(bc[-1]-bg):.1e}")
rB = run_B(d, sig_rot=1e-4, sig_pos=1e-4, meas_infl=1.0)
check("E2 B(ESKF) recovers constant b_g and b_a", np.linalg.norm(rB['b_g'][-1] - bg) < 2e-4 and np.linalg.norm(rB['b_a'][-1] - ba) < 3e-3,
      f"err_g {np.linalg.norm(rB['b_g'][-1]-bg):.1e}  err_a {np.linalg.norm(rB['b_a'][-1]-ba):.1e}")
bca, _ = run_C_accel(d, np.tile(bg, (K, 1)), W=30, every=40, sig_pos=1e-4)
check("E2 C_accel recovers constant b_a", np.linalg.norm(bca[-1] - ba) < 5e-3, f"err {np.linalg.norm(bca[-1]-ba):.1e} m/s^2 (|b|={np.linalg.norm(ba):.3f})")
bA = run_A_accel(d, iv, np.tile(bg, (K, 1)), "dt_weighted", 1e5)
eA = np.linalg.norm(bA[-1] - ba)
check("E2x A_accel (frame-level, mentor) is BIASED even with perfect vision (documented negative result)", eA > 5 * np.linalg.norm(bca[-1] - ba),
      f"err {eA:.3f} vs signal |b_a|={np.linalg.norm(ba):.3f}: velocity-from-position-difference truncation error swamps the bias signal")

# E3 zero bias: no invented bias
d0 = mk(); iv0 = Intervals(d0)
m = max(np.linalg.norm(run_A_gyro(d0, iv0, "dt_weighted", 1e5)[-1]), np.linalg.norm(run_C_gyro(d0, iv0, W=20, every=10)[0][-1]))
r0 = run_B(d0, sig_rot=1e-4, sig_pos=1e-4, meas_infl=1.0)
check("E3 zero true bias -> estimates ~0 (no spurious bias)", m < 2e-4 and np.linalg.norm(r0['b_g'][-1]) < 2e-4 and np.linalg.norm(r0['b_a'][-1]) < 3e-3,
      f"A/C gyro {m:.1e}, B gyro {np.linalg.norm(r0['b_g'][-1]):.1e}, B accel {np.linalg.norm(r0['b_a'][-1]):.1e}")

# E4 sign flip: negated truth -> negated estimate
dn = mk(b_g0=-bg, b_a0=-ba); ivn = Intervals(dn)
sA = np.linalg.norm(run_A_gyro(dn, ivn, "dt_weighted", 1e5)[-1] + bg)
rn = run_B(dn, sig_rot=1e-4, sig_pos=1e-4, meas_infl=1.0)
check("E4 sign convention: b -> -b flips estimates (A gyro, B gyro, B accel)", sA < 2e-4 and np.linalg.norm(rn['b_g'][-1] + bg) < 2e-4 and np.linalg.norm(rn['b_a'][-1] + ba) < 3e-3,
      f"A {sA:.1e}, B_g {np.linalg.norm(rn['b_g'][-1]+bg):.1e}, B_a {np.linalg.norm(rn['b_a'][-1]+ba):.1e}")

# E5 ramp tracking (noise-free): warm-up drifting bias, short time constants must track with small lag
dw = mk(b_g0=np.zeros(3), warm_g=np.array([0.02, -0.01, 0.015]), warm_tau=15.0); ivw = Intervals(dw)
tw = dw.t_v_s
bw_true = np.array([dw.scen.warm_g * (1 - np.exp(-(t - dw.t_imu_s[0]) / dw.scen.warm_tau)) for t in tw])
ea = np.abs(run_A_gyro(dw, ivw, "dt_weighted", 4.0) - bw_true)[len(tw) // 2:].max()
eb = np.abs(run_B(dw, sig_rot=1e-4, sig_pos=1e-4, meas_infl=1.0, q_bg=2e-3)['b_g'] - bw_true)[len(tw) // 2:].max()
check("E5 tracks an exponential warm-up drift after convergence (A tau=4 s; B with matching q)", ea < 2e-3 and eb < 2e-3, f"A max err {ea:.1e}, B max err {eb:.1e} rad/s")

nf = sum(1 for r in res if not r['ok'])
print(f"\n{len(res)-nf}/{len(res)} passed")
json.dump(res, open(Path(__file__).parent / "test_estimators_results.json", "w"), indent=1)
sys.exit(1 if nf else 0)
