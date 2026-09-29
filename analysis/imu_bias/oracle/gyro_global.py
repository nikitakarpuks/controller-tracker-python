import sys, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from gyro_oracle import *
def load(name, ctrl, tau=0.5):
    z = np.load(OUT + f"gyro_terms_{name}_{ctrl}_tau{tau}.npz"); return {k: z[k] for k in z.files}
def boot(d, idx_all, with_K, n=300, block_s=10.0, rng=np.random.default_rng(1)):
    sec = ((d["t0"] - d["t0"][0]) / 1e9 / block_s).astype(int); ub = np.unique(sec); bs = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub)); idx = np.concatenate([np.where(sec == u)[0] for u in pick])
        bs.append(solve(d, idx, with_K=with_K)[0])
    bs = np.array(bs); return np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
rows = []
for name in REC_NAMES:
    for ctrl in CTRLS:
        d = load(name, ctrl); idx = np.arange(len(d["t0"]))
        b0, _, k0 = solve(d, idx, with_K=False); b1, K1, k1 = solve(d, idx, with_K=True)
        lo0, hi0 = boot(d, idx, False); lo1, hi1 = boot(d, idx, True)
        rows.append(dict(rec=name, ctrl=ctrl, n=len(idx), **{f"b0_{a}": b0[i] for i, a in enumerate("xyz")}, **{f"b0lo_{a}": lo0[i] for i, a in enumerate("xyz")}, **{f"b0hi_{a}": hi0[i] for i, a in enumerate("xyz")},
                         **{f"b1_{a}": b1[i] for i, a in enumerate("xyz")}, **{f"b1lo_{a}": lo1[i] for i, a in enumerate("xyz")}, **{f"b1hi_{a}": hi1[i] for i, a in enumerate("xyz")},
                         **{f"K{r}{c}": K1[r, c] for r in range(3) for c in range(3)}))
        print(f"{name:14s}{ctrl[:5]}: bias-only {b0.round(4)}  | bias+K {b1.round(4)} CI[{lo1.round(4)} .. {hi1.round(4)}] Kdiag {np.diag(K1).round(4)}", flush=True)
with open(OUT + "gyro_bias_global.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
