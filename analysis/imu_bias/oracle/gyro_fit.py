import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
OUT = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle/"

def ols_bias(om, e, robust=True):
    """e_axis = b + K[axis,:] . omega  per axis -> b (3,), K (3,3). Iteratively trims |resid|>4 MAD."""
    X = np.hstack([np.ones((len(om), 1)), om])
    keep = np.ones(len(om), bool)
    for _ in range(3 if robust else 1):
        coef, *_ = np.linalg.lstsq(X[keep], e[keep], rcond=None)
        r = e - X @ coef
        if not robust: break
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep], 0)), 0) + 1e-9
        keep = (np.abs(r) < 4 * s).all(1)
    return coef[0], coef[1:].T, keep

def block_boot(om, e, t, n=200, block_s=5.0, rng=np.random.default_rng(0)):
    """block bootstrap CI of the regression intercept."""
    sec = ((t - t[0]) / 1e9 / block_s).astype(int); ub = np.unique(sec); bs = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub)); idx = np.concatenate([np.where(sec == u)[0] for u in pick])
        bs.append(ols_bias(om[idx], e[idx], robust=False)[0])
    bs = np.array(bs); return np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)

if __name__ == "__main__":
    for name in (sys.argv[1:] or ["static_dark"]):
        for ctrl in CTRLS:
            a = np.load(OUT + f"gyro_intervals_{name}_{ctrl}.npy"); t = a[:, 0]; om = a[:, 1:4]; e = a[:, 4:7]
            b, K, keep = ols_bias(om, e)
            lo, hi = block_boot(om, e, t)
            wn = np.linalg.norm(om, axis=1)
            stat = wn < 0.5
            bs = e[stat].mean(0) if stat.sum() > 5 else np.full(3, np.nan)
            print(f"{name}/{ctrl}: n={len(a)} kept {keep.sum()}")
            print(f"  plain mean          {e.mean(0).round(4)} rad/s")
            print(f"  regression intercept {b.round(4)}  95%CI lo {lo.round(4)} hi {hi.round(4)}")
            print(f"  K (rows=axis) =\n{K.round(4)}")
            print(f"  rest-only (|omega|<0.5 rad/s, n={stat.sum()}) mean {np.round(bs,4)}  sem {np.round(e[stat].std(0)/np.sqrt(max(stat.sum(),1)),4) if stat.sum()>5 else 'n/a'}")
