"""Step 1b: anatomy of the per-step gyro-vs-vision residual (zero extra bias). Questions:
 (a) is it white vision noise (lag-1 autocorrelation ~ -0.5 for differenced noise) or systematic?
 (b) does it scale with rotation rate / angular acceleration (timing/scale) rather than being constant (bias)?
 (c) how big is a plausible bias contribution (b*dt) relative to it?
 (d) imu0 (headset gyro) ego-rotation vs mocap headset rotation: lag scan + bias."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *

name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; v = d["vision"]; ts = v.ts
    ok = v.strong_mask()
    Rw = []
    for i, t in enumerate(ts):
        Rh = run.R_wh(t)
        Rw.append(None if Rh is None else Rh @ v.R[i])
    ok &= np.array([r is not None for r in Rw])
    pairs = step_pairs(ts, ok)
    E, DT, W, ROT = [], [], [], []
    for i, j in pairs:
        Rg = integrate_gyro_segment(d["t"], d["gyro"], int(ts[i]), int(ts[j]))
        if Rg is None: continue
        dRv = Rw[i].T @ Rw[j]
        E.append(Rotation.from_matrix(Rg.T @ dRv).as_rotvec())      # body frame at j, rad
        DT.append((ts[j]-ts[i])/1e9)
        W.append(Rotation.from_matrix(dRv).as_rotvec() / DT[-1])    # body-frame rate (rad/s)
    E = np.array(E); DT = np.array(DT); W = np.array(W)
    en = np.linalg.norm(E, axis=1)
    print(f"\n[{name}/{c}] steps {len(E)}   |e| median {np.degrees(np.median(en)):.3f} deg  (per-step, rad median {np.median(en):.2e})")
    # (a) lag-1 autocorrelation of consecutive-step residual components
    cons = np.array([pairs_i == pairs_j for pairs_i, pairs_j in zip([p[1] for p in pairs[:-1]], [p[0] for p in pairs[1:]])])
    ac = []
    for ax in range(3):
        a, b = E[:-1, ax][cons[:len(E)-1]], E[1:, ax][cons[:len(E)-1]]
        ac.append(np.corrcoef(a, b)[0, 1])
    print(f"   lag-1 autocorr of residual (x,y,z): {np.round(ac,3)}   (differenced white noise -> -0.5; bias/scale -> >0)")
    # (b) dependence on rate and angular acceleration
    wn = np.linalg.norm(W, axis=1)
    alpha = np.linalg.norm(np.diff(W, axis=0), axis=1) / DT[1:]
    print(f"   |rate| median {np.median(wn):.2f} rad/s;  corr(|e|,|rate|) {np.corrcoef(en, wn)[0,1]:.3f}   "
          f"corr(|e|[1:],|ang.accel|) {np.corrcoef(en[1:], alpha)[0,1]:.3f}")
    for lo, hi in ((0, .5), (.5, 1.5), (1.5, 3), (3, 99)):
        m = (wn >= lo) & (wn < hi)
        if m.sum() > 20: print(f"      rate {lo}-{hi} rad/s: n={m.sum():5d}  median |e| {np.degrees(np.median(en[m])):.3f} deg")
    # (c) what a bias of a given size would look like per step
    for b in (0.002, 0.005, 0.01, 0.02):
        print(f"   bias {b} rad/s -> per-step effect {np.degrees(b*np.median(DT)):.4f} deg vs |e| median {np.degrees(np.median(en)):.3f} deg (ratio {b*np.median(DT)/np.median(en):.3f})")
    # mean residual per step normalised by dt (raw bias measurement) -- the naive literal estimator's raw output
    bm = -E / DT[:, None]
    print(f"   raw per-frame bias measurement -e/dt: mean {np.round(bm.mean(0),4)} rad/s   std {np.round(bm.std(0),3)} rad/s "
          f"-> naive per-frame SNR for 0.005 rad/s: {0.005/np.mean(bm.std(0)):.4f}")
    S = -E.sum(0) / DT.sum()
    print(f"   ratio-of-sums bias (-sum e / sum dt): {np.round(S,5)} rad/s   (telescoping-friendly)")

# (d) headset imu0 vs mocap headset rotation
t0, g0, a0 = run.imu0
ts_all = np.arange(t0[0] + 2_000_000_000, t0[-1] - 2_000_000_000, 20_000_000, dtype=np.int64)
def ego_err(lag_ns, bias=np.zeros(3), step=6):
    errs = []
    for k in range(0, len(ts_all) - step, 4):
        a, b_ = int(ts_all[k]), int(ts_all[k + step])
        Ra, Rb = run.R_wh(a), run.R_wh(b_)
        if Ra is None or Rb is None: continue
        Rg = integrate_gyro_segment(t0 + lag_ns, g0 - bias, a, b_)
        if Rg is None: continue
        errs.append(rot_deg(Rg.T @ (Ra.T @ Rb)))
    return np.array(errs)
print("\n[imu0 (headset) gyro vs mocap headset rotation over 120ms windows; raw imu0 (no calib)]")
for lag in (-6e6, -4e6, -2e6, 0, 2e6, 4e6, 6e6):
    e = ego_err(int(lag))
    print(f"   lag {lag/1e6:+.0f} ms: median {np.median(e):.3f} deg  p95 {np.percentile(e,95):.3f}  n={len(e)}")
