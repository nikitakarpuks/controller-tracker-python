import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from accel_oracle import *
def loadw(n, c): return pickle.load(open(OUT + f"accel_windows_{n}_{c}.pkl", "rb"))
GOOD = ["static_dark", "walk_dark", "static_easy", "static_medium", "walk_easy", "walk_medium"]
rng = np.random.default_rng(5)
rows = []; sumrows = []
for ctrl in CTRLS:
    W = {n: loadw(n, ctrl) for n in GOOD}; R = len(GOOD); P = 3 * R + 12
    AtA = np.zeros((P, P)); Aty = np.zeros(P); yty = 0.0; N = 0
    for j, n in enumerate(GOOD):
        for w in W[n]:
            sel = list(range(3 * j, 3 * j + 3)) + list(range(3 * R, 3 * R + 12))
            AtA[np.ix_(sel, sel)] += w["AtA"]; Aty[sel] += w["Aty"]; yty += w["yty"]; N += w["n"]
    th = np.linalg.solve(AtA, Aty); S = th[3*R:3*R+9]; dg = th[3*R+9:]
    rss = yty - 2 * th @ Aty + th @ AtA @ th
    print(f"\n{ctrl}: pooled over {GOOD}: rms {np.sqrt(rss/N)*1000:.2f} mm")
    print(f"  S =\n{S.reshape(3,3).round(4)}  dg = {dg.round(4)}")
    Sdg = np.concatenate([S, dg])
    print("  per-recording b (m/s^2) with pooled S,dg  [order = recording time order]")
    bpr = {}
    for n in GOOD:
        idx = np.arange(len(W[n])); a = sum(W[n][i]["AtA"] for i in idx); y = sum(W[n][i]["Aty"] for i in idx)
        # b | S,dg fixed:  A_b^T A_b b = A_b^T y - A_b^T A_Sdg Sdg
        b = np.linalg.solve(a[:3, :3], y[:3] - a[:3, 3:] @ Sdg)
        bs = []
        for _ in range(200):
            pick = rng.choice(idx, len(idx)); a2 = sum(W[n][i]["AtA"] for i in pick); y2 = sum(W[n][i]["Aty"] for i in pick)
            bs.append(np.linalg.solve(a2[:3, :3], y2[:3] - a2[:3, 3:] @ Sdg))
        bs = np.array(bs); bpr[n] = b
        print(f"     {n:14s} b = {b.round(3)}  95%CI +/- {(1.96*bs.std(0)).round(3)}")
    # block-wise (20 s blocks) with S,dg fixed
    print("  20-s blocks: chi2/dof relative to per-recording constant, and ensemble mean vs time-in-recording")
    allB = []
    for n in GOOD:
        wins = W[n]; t = np.array([(w["t0"] - wins[0]["t0"]) / 1e9 for w in wins]); nb = int(t.max() // 20) + 1
        Bn = []; SEn = []; Tc = []
        for k in range(nb):
            idx = np.flatnonzero((t >= 20 * k) & (t < 20 * (k + 1)))
            if len(idx) < 6: continue
            a = sum(wins[i]["AtA"] for i in idx); y = sum(wins[i]["Aty"] for i in idx)
            b = np.linalg.solve(a[:3, :3], y[:3] - a[:3, 3:] @ Sdg)
            bs = []
            for _ in range(100):
                pick = rng.choice(idx, len(idx)); a2 = sum(wins[i]["AtA"] for i in pick); y2 = sum(wins[i]["Aty"] for i in pick)
                bs.append(np.linalg.solve(a2[:3, :3], y2[:3] - a2[:3, 3:] @ Sdg))
            Bn.append(b); SEn.append(np.std(bs, 0)); Tc.append(20 * k + 10)
            rows.append(dict(rec=n, ctrl=ctrl, t_center_s=20 * k + 10, **{f"b_{a_}": b[i] for i, a_ in enumerate("xyz")}, **{f"se_{a_}": SEn[-1][i] for i, a_ in enumerate("xyz")}))
        Bn = np.array(Bn); SEn = np.array(SEn); Tc = np.array(Tc)
        chi2 = (((Bn - Bn.mean(0)) / SEn) ** 2).sum(0) / (len(Bn) - 1)
        slope = np.array([np.polyfit(Tc, Bn[:, i], 1)[0] for i in range(3)]) * 60
        sumrows.append((n, chi2, slope, SEn.mean(0)))
        allB.append((Tc, Bn))
    print("     chi2/dof per recording (x,y,z):", [f"{n}:{np.round(c,1)}" for n, c, s, e in sumrows[-len(GOOD):]])
    print("     within-recording slope (m/s^2 per min) mean over recs:", np.mean([s for n, c, s, e in sumrows[-len(GOOD):]], 0).round(4), " mean block SE:", np.mean([e for n, c, s, e in sumrows[-len(GOOD):]], 0).round(4))
    # ensemble by time bin
    for a_, b_ in ((0, 30), (30, 60), (60, 90), (90, 130)):
        vals = np.array([B[i] for Tc, B in allB for i in range(len(Tc)) if a_ <= Tc[i] < b_])
        print(f"     t in [{a_:3d},{b_:3d}): n={len(vals)} mean {vals.mean(0).round(3)} sd {vals.std(0).round(3)}")
    # across-recording drift: b vs recording order (approx start times in minutes)
    tmin = np.array([0, 2, 8, 11, 21, 24], float)  # static_dark 17:31, walk_dark 17:33, static_easy 17:39, static_medium 17:42, walk_easy 17:52, walk_medium 17:55
    Bm = np.array([bpr[n] for n in GOOD])
    print("     across-recording trend (m/s^2 per min of session): ", np.array([np.polyfit(tmin, Bm[:, i], 1)[0] for i in range(3)]).round(4), " corr with time:", np.array([np.corrcoef(tmin, Bm[:, i])[0, 1] for i in range(3)]).round(2))
with open(OUT + "accel_bias_blocks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
