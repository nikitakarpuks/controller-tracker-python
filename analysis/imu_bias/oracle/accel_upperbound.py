"""Accel upper bound (position): start from MOCAP state (p0, v0 from a +-40 ms local linear fit), oracle mocap orientation, integrate
corrected accelerometer  a = R (f - b - S f) + g0 + dg   over T and compare with mocap position p(T).  Variants isolate bias vs gain vs gravity."""
import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from accel_oracle import *
GOOD = ["static_dark", "walk_dark", "static_easy", "static_medium", "walk_easy", "walk_medium"]
def loadw(n, c): return pickle.load(open(OUT + f"accel_windows_{n}_{c}.pkl", "rb"))
DUR = [0.05, 0.1, 0.3, 1.0]; NST = 200; rng = np.random.default_rng(21)
res = []
variants = ["A0_factory", "A1_bias_only", "A2_gain_only(S,dg)", "A3_bias+S+dg"]
for ctrl in CTRLS:
    W = {n: loadw(n, ctrl) for n in GOOD}; R = len(GOOD); P = 3 * R + 12
    AtA = np.zeros((P, P)); Aty = np.zeros(P)
    for j, n in enumerate(GOOD):
        for w in W[n]:
            sel = list(range(3 * j, 3 * j + 3)) + list(range(3 * R, 3 * R + 12))
            AtA[np.ix_(sel, sel)] += w["AtA"]; Aty[sel] += w["Aty"]
    th = np.linalg.solve(AtA, Aty); Sdg = th[3*R:]; S = Sdg[:9].reshape(3, 3); dg = Sdg[9:]
    for j, n in enumerate(GOOD):
        a = sum(w["AtA"] for w in W[n]); y = sum(w["Aty"] for w in W[n])
        b_full = th[3*j:3*j+3]
        b_only = np.linalg.solve(a[:3, :3], y[:3])                    # bias-only oracle (S=0, dg=0 refit)
        rdir = rec_dir(n); dev = load_mocap_device(rdir, ctrl); mo = MocapPose(dev)
        t, gb, ab = load_imu(rdir, ctrl); f = (DIAG_FLIP @ ab.T).T
        RI_all = None
        for T in DUR:
            dtn = int(T * 1e9)
            cand = np.arange(t[0] + 0.5e9, t[-1] - dtn - 0.5e9, 2e7).astype(np.int64)
            ok = mo.valid(cand) & mo.valid(cand + dtn) & mo.valid(cand - int(0.05e9)) & mo.valid(cand + int(0.05e9)); cand = cand[ok]
            starts = rng.choice(cand, min(NST, len(cand)), replace=False)
            errs = {v: [] for v in variants}
            for s0 in starts:
                # initial state from mocap: linear fit of position over +-40 ms
                tq = (s0 + np.linspace(-0.04e9, 0.04e9, 17)).astype(np.int64); pq = mo.pos_imu(tq); tt = (tq - s0) / 1e9
                v0 = np.array([np.polyfit(tt, pq[:, i], 1)[0] for i in range(3)]); p0 = np.array([np.polyval(np.polyfit(tt, pq[:, i], 1), 0) for i in range(3)])
                sel = np.flatnonzero((t >= s0) & (t <= s0 + dtn))
                tI = np.concatenate(([s0], t[sel], [s0 + dtn])).astype(np.int64); fI = np.stack([np.interp(tI, t, f[:, i]) for i in range(3)], 1)
                RI = mo.R_world_imu(tI).as_matrix(); ts = (tI - tI[0]) / 1e9
                p_true = mo.pos_imu(np.array([s0 + dtn]))[0]
                for v in variants:
                    if v == "A0_factory": fc = fI; g = G0
                    elif v == "A1_bias_only": fc = fI - b_only; g = G0
                    elif v == "A2_gain_only(S,dg)": fc = fI - fI @ S.T; g = G0 + dg
                    else: fc = fI - b_full - fI @ S.T; g = G0 + dg
                    a_w = np.einsum("nij,nj->ni", RI, fc) + g
                    p = p0 + v0 * ts[-1] + cum2(ts, a_w)[-1]
                    errs[v].append(np.linalg.norm(p - p_true) * 1000)
            for v in variants: res.append((n, ctrl, v, T, np.array(errs[v])))
        print(f"{ctrl[:5]} {n} done", flush=True)
rows = []
for v in variants:
    for T in DUR:
        e = np.concatenate([r[4] for r in res if r[2] == v and r[3] == T])
        rows.append(dict(variant=v, T_s=T, n=len(e), median_mm=np.median(e), mean_mm=e.mean(), p90_mm=np.percentile(e, 90)))
with open(OUT + "accel_upperbound.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nposition dead-reckoning error (mm), moderate recordings, both controllers -- median [mean]")
print(f"{'variant':24s}" + "".join(f"T={T:<14}" for T in DUR))
for v in variants:
    print(f"{v:24s}" + "".join(f"{[r for r in rows if r['variant']==v and r['T_s']==T][0]['median_mm']:6.2f} [{[r for r in rows if r['variant']==v and r['T_s']==T][0]['mean_mm']:6.2f}]  " for T in DUR))
