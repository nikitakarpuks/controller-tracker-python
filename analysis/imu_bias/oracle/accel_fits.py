import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from accel_oracle import *
def loadw(n, c): return pickle.load(open(OUT + f"accel_windows_{n}_{c}.pkl", "rb"))
def ne(wins, idx):
    return (sum(wins[i]["AtA"] for i in idx), sum(wins[i]["Aty"] for i in idx), sum(wins[i]["yty"] for i in idx), sum(wins[i]["n"] for i in idx))
rng = np.random.default_rng(3)
rows = []
print("=== per-recording fits (b, S, dg free), window-bootstrap 95% CI on b (m/s^2) ===")
for ctrl in CTRLS:
    for n in REC_NAMES:
        wins = loadw(n, ctrl); idx = np.arange(len(wins))
        AtA, Aty, yty, N = ne(wins, idx); th = np.linalg.solve(AtA, Aty)
        bs = []
        for _ in range(300):
            pick = rng.choice(idx, len(idx)); a, b_, *_ = ne(wins, pick); bs.append(np.linalg.solve(a, b_)[:3])
        bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        rss = yty - 2 * th @ Aty + th @ AtA @ th; rms = np.sqrt(rss / N)
        cov = rss / (N - 15) * np.linalg.inv(AtA); sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
        # identifiability: corr of b_y with S_yy, dg_y ; b_x with S_xx, dg_x ; b_z with S_zz, dg_z  (indices: b0-2, S 3..11 row-major, dg 12..14)
        cy = [corr[1, 7], corr[1, 13]]; cx = [corr[0, 3], corr[0, 12]]; cz = [corr[2, 11], corr[2, 14]]
        print(f"{n:14s}{ctrl[:5]}: nwin={len(wins):2d} rms {rms*1000:5.2f}mm  b={th[:3].round(3)} CI lo{lo.round(3)} hi{hi.round(3)}  Sdiag={np.array([th[3],th[7],th[11]]).round(4)} dg={th[12:].round(3)}  cond(AtA)={np.linalg.cond(AtA):.1e}  corr(b_y,S_yy)={cy[0]:+.2f} corr(b_y,dg_y)={cy[1]:+.2f}")
        rows.append(dict(rec=n, ctrl=ctrl, nwin=len(wins), rms_mm=rms*1000, **{f"b_{a}": th[i] for i, a in enumerate("xyz")}, **{f"blo_{a}": lo[i] for i, a in enumerate("xyz")}, **{f"bhi_{a}": hi[i] for i, a in enumerate("xyz")},
                         **{f"S{r}{c}": th[3+3*r+c] for r in range(3) for c in range(3)}, **{f"dg_{a}": th[12+i] for i, a in enumerate("xyz")}))
with open(OUT + "accel_bias_per_recording.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n=== pooled per controller: shared S(9), dg(3); per-recording b(3) ===")
pooled = {}
for ctrl in CTRLS:
    W = {n: loadw(n, ctrl) for n in REC_NAMES}; R = len(REC_NAMES); P = 3 * R + 12
    for variant in ("full", "dg_fixed0"):
        AtA = np.zeros((P, P)); Aty = np.zeros(P); yty = 0.0; N = 0
        for j, n in enumerate(REC_NAMES):
            for w in W[n]:
                sel = list(range(3 * j, 3 * j + 3)) + list(range(3 * R, 3 * R + 12))
                loc = list(range(15))
                AtA[np.ix_(sel, sel)] += w["AtA"]; Aty[sel] += w["Aty"]; yty += w["yty"]; N += w["n"]
        if variant == "dg_fixed0":
            keep = list(range(P - 3))
            th = np.zeros(P); th[keep] = np.linalg.solve(AtA[np.ix_(keep, keep)], Aty[keep])
        else:
            th = np.linalg.solve(AtA, Aty)
        rss = yty - 2 * th @ Aty + th @ AtA @ th
        S = th[3*R:3*R+9].reshape(3, 3); dg = th[3*R+9:]
        print(f"{ctrl[:5]} [{variant}] rms {np.sqrt(rss/N)*1000:.2f} mm  S=\n{S.round(4)}  dg={dg.round(4)}")
        for j, n in enumerate(REC_NAMES): print(f"     {n:14s} b = {th[3*j:3*j+3].round(3)}")
        pooled[(ctrl, variant)] = (th, R)
pickle.dump({k: v[0] for k, v in pooled.items()}, open(OUT + "accel_pooled.pkl", "wb"))
