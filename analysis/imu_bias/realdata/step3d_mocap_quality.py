"""Step 3d: how good is the mocap-orientation reference for gyro steps? per-step |e0| median/rms, autocorrelation,
lag scan for mocap, mocap sampling (120Hz) interpolation effect, vs the vision-based steps."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]
    st_m = mocap_steps(run, c)
    prep = prepare(run, c, ego="mocap"); st_v, _ = make_steps(run, c, prep)
    for lab, st in (("mocap-orientation steps", st_m), ("vision steps (strong)", st_v)):
        e = np.array([np.linalg.norm(s["e0"]) for s in st]); E = np.array([s["e0"] for s in st])
        ac = np.mean([np.corrcoef(E[:-1, k], E[1:, k])[0, 1] for k in range(3)])
        print(f"[{name}/{c}] {lab:24s} n={len(st):5d}  |e| median {np.degrees(np.median(e)):.3f} deg  mean {np.degrees(e.mean()):.3f}  rms {np.degrees(np.sqrt((e**2).mean())):.3f}  p95 {np.degrees(np.percentile(e,95)):.3f}  lag1 autocorr {ac:+.3f}")
    # combined "mocap vs vision" per-step relative rotation disagreement (pure measurement noise + bridge)
    tv = {s["t"]: s for s in st_v}
    diffs = []
    for s in st_m:
        if s["t"] in tv:
            diffs.append(np.linalg.norm(Rotation.from_matrix(np.eye(3)).as_rotvec()))  # placeholder
    # lag scan of the mocap steps (shift gyro)
    ts = d["vision"].ts
    Rm = [run.R_w_ctrl_mocap_led(c, t) for t in ts]
    okm = [r is not None for r in Rm]
    idx = [i for i in range(len(ts)) if okm[i]]
    pairs = [(a, b) for a, b in zip(idx[:-1], idx[1:]) if 0 < (ts[b]-ts[a])/1e9 <= 0.05][::3]
    print(f"   gyro-vs-MOCAP shift scan (extra shift on top of lag_ns), median |e| deg over {len(pairs)} steps:")
    line = []
    for sh in np.arange(-8, 8.01, 2.0):
        E = []
        for a, b in pairs:
            Rg = integrate_gyro_segment(d["t"] + int(sh*1e6), d["gyro"], int(ts[a]), int(ts[b]))
            if Rg is None: continue
            E.append(np.linalg.norm(Rotation.from_matrix(Rg.T @ (Rm[a].T @ Rm[b])).as_rotvec()))
        line.append(f"{sh:+.0f}:{np.degrees(np.median(E)):.3f}")
    print("     ", "  ".join(line))
    # mocap-vs-vision consistency: orientation difference statistics (bridge-composed), constant offset removed
    dr = []
    for i in np.flatnonzero(prep["ok"]):
        if okm[i]:
            dr.append(Rotation.from_matrix(prep["R_wc"][i].T @ Rm[i]).as_rotvec())
    dr = np.array(dr)
    print(f"   vision-vs-mocap orientation difference (body frame): mean {np.round(np.degrees(dr.mean(0)),3)} deg   std {np.round(np.degrees(dr.std(0)),3)} deg")
