"""Step 3f: per-step BODY-frame regression  e0_j = -dt_j (b + K omega_j)  (b: 3, K: 3x3), for vision steps and for mocap steps
SEPARATELY (independent references) -- if K is a real gyro scale/misalignment it must agree between them. Then check
whether removing K*omega restores the telescoping behaviour of the accumulated world-frame residual S(T)."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *

def fit_body(st, use_b=True, key="omega"):
    X, y = [], []
    for s in st:
        dt, om = s["dt"], s[key]
        blk = np.zeros((3, 9))
        for a in range(3):
            for b in range(3):
                blk[a, a * 3 + b] = om[b] * dt
        X.append(np.hstack([dt * np.eye(3), blk]) if use_b else blk); y.append(-s["e0"])
    X = np.vstack(X); y = np.concatenate(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # bootstrap-ish uncertainty from residual scatter (independent-step approx) -- crude
    res = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * res.var()
    return beta, np.sqrt(np.diag(cov)), res

def S_scaling(st, Kb=None, b=None, Ts=(0.5, 1, 2, 5, 10, 20)):
    tt = np.array([s["t"] for s in st])
    W = []
    for s in st:
        v = -(s["R_end"] @ s["e0"])
        if Kb is not None:
            v = v - s["dt"] * (s["R_end"] @ (Kb @ s["omega"]))
        if b is not None:
            v = v - s["dt"] * (s["R_end"] @ b)
        W.append(v)
    W = np.array(W); cs = np.vstack([np.zeros(3), np.cumsum(W, 0)])
    out = []
    for T in Ts:
        vals = []
        for k in range(0, len(st), 25):
            hi = np.searchsorted(tt, tt[k] + int(T * 1e9))
            if hi >= len(st): break
            if (tt[hi] - tt[k]) / 1e9 > 1.2 * T: continue
            vals.append(np.linalg.norm(cs[hi + 1] - cs[k]))
        out.append(np.sqrt(np.mean(np.square(vals))))
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
    run = Run(name)
    for c in CTRLS:
        prep = prepare(run, c, ego="mocap"); st_v, _ = make_steps(run, c, prep); st_m = mocap_steps(run, c)
        print(f"\n[{name}/{c}] body-frame per-step regression  e0 = -dt (b + K omega)")
        fits = {}
        for lab, st in (("vision", st_v), ("mocap", st_m)):
            beta, se, res = fit_body(st)
            fits[lab] = beta
            K = beta[3:].reshape(3, 3); Kse = se[3:].reshape(3, 3)
            print(f"   [{lab}] b = {np.round(beta[:3],4)} (+-{np.round(se[:3],4)}) rad/s")
            print(f"          K =\n{np.round(K,4)}\n          (se)\n{np.round(Kse,4)}")
            print(f"          residual rms {np.degrees(np.sqrt((res**2).sum()/len(st)/3)):.3f} deg/comp, vs raw {np.degrees(np.sqrt(sum((s['e0']**2).sum() for s in st)/len(st)/3)):.3f}")
        print(f"   K agreement vision vs mocap: max |dK| {np.abs(fits['vision'][3:]-fits['mocap'][3:]).max():.4f}   corr of 9 entries {np.corrcoef(fits['vision'][3:], fits['mocap'][3:])[0,1]:.3f}")
        Ts = (0.5, 1, 2, 5, 10, 20)
        for lab, st in (("vision", st_v), ("mocap", st_m)):
            K = fits[lab][3:].reshape(3, 3)
            print(f"   S(T) for {lab} steps   T={Ts}")
            print(f"      raw             {np.round(S_scaling(st), 4)}")
            print(f"      minus own K*w   {np.round(S_scaling(st, Kb=K), 4)}")
            print(f"      minus OTHER ref's K*w {np.round(S_scaling(st, Kb=fits['mocap' if lab=='vision' else 'vision'][3:].reshape(3,3)), 4)}")
