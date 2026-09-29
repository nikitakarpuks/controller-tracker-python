"""Step 3c: gyro<->vision (and gyro<->mocap) time-shift scan, per recording quarter. Metric: median and rms per-step
|e0| (zero extra bias). If timing is wrong the residual is dominated by (omega_{k+1}-omega_k)*delta terms, which would
swamp bias. Shifts are applied ON TOP of the shipped lag_ns (camera clock)."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *

name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
SHIFTS_MS = np.arange(-6, 6.01, 1.0)
for c in CTRLS:
    d = run.ctrl[c]
    t0_all = d["vision"].ts[0]
    prep = prepare(run, c, ego="mocap")
    ts, R_wc, ok = prep["ts"], prep["R_wc"], prep["ok"]
    idx = np.flatnonzero(ok)
    pairs = [(a, b) for a, b in zip(idx[:-1], idx[1:]) if 0 < (ts[b]-ts[a])/1e9 <= 0.05]
    # subsample for speed
    pairs = pairs[::2]
    tq = np.array([ts[b] for a, b in pairs])
    quarters = np.digitize(tq, np.quantile(tq, [0.25, 0.5, 0.75]))
    med = np.zeros((len(SHIFTS_MS), 4)); rms = np.zeros_like(med)
    for si, sh in enumerate(SHIFTS_MS):
        E = []
        for (a, b) in pairs:
            Rg = integrate_gyro_segment(d["t"] + int(sh * 1e6), d["gyro"], int(ts[a]), int(ts[b]))
            E.append(np.nan if Rg is None else np.linalg.norm(Rotation.from_matrix(Rg.T @ (R_wc[a].T @ R_wc[b])).as_rotvec()))
        E = np.degrees(np.array(E))
        for q in range(4):
            m = (quarters == q) & np.isfinite(E) & (E < 3.0)
            med[si, q] = np.median(E[m]); rms[si, q] = np.sqrt(np.mean(E[m] ** 2))
    print(f"\n[{name}/{c}] gyro time-shift scan (extra shift on top of lag_ns={d['lag']/1e6:.2f}ms), median |e| per step (deg)   quarters Q1..Q4")
    for si, sh in enumerate(SHIFTS_MS):
        print(f"   {sh:+.0f} ms: median {np.round(med[si],3)}   rms {np.round(rms[si],3)}")
    best = SHIFTS_MS[np.argmin(med, axis=0)]
    print(f"   best shift per quarter (median): {best} ms;  overall best (mean of medians) {SHIFTS_MS[np.argmin(med.mean(1))]:+.0f} ms")
    # gyro-vs-MOCAP-controller timing (mocap steps), whole recording
    st_med = []
    ridx = np.flatnonzero(np.array([run.R_w_ctrl_mocap_led(c, t) is not None for t in ts[::1]]))
