"""Accel: pooled (per-recording b(3), shared S(9), shared dg(3)) fits for each stream, analytic mapping to correction variants,
and the direct test of the Monado raw-unit conversion (1 g ~ 490000 counts read as /49000 => 10.0 m/s^2 per g)."""
import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *
np.set_printoptions(precision=4, suppress=True, linewidth=170)
I3 = np.eye(3)
SCALE = 9.80665 / 10.0         # correction for the driver's 10.0 m/s^2-per-1g conversion

def load_wins(ctrl, sname, names):
    return {n: pickle.load(open(FA + f"terms_accel_{n}_{ctrl}.pkl", "rb"))[sname] for n in names}

def normal_eq(W, names, sel_idx=None):
    R = len(names); P = 3 * R + 12
    AtA = np.zeros((P, P)); Aty = np.zeros(P); yty = 0.0; N = 0
    for j, n in enumerate(names):
        idx = range(len(W[n])) if sel_idx is None else sel_idx[n]
        for i in idx:
            w = W[n][i]; sel = list(range(3 * j, 3 * j + 3)) + list(range(3 * R, 3 * R + 12))
            AtA[np.ix_(sel, sel)] += w["AtA"]; Aty[sel] += w["Aty"]; yty += w["yty"]; N += w["n"]
    return AtA, Aty, yty, N

def pooled(W, names, sel_idx=None):
    R = len(names); AtA, Aty, yty, N = normal_eq(W, names, sel_idx)
    th = np.linalg.solve(AtA, Aty); rss = yty - 2 * th @ Aty + th @ AtA @ th
    return th[:3 * R].reshape(R, 3), th[3 * R:3 * R + 9].reshape(3, 3), th[3 * R + 9:], np.sqrt(rss / N), AtA

def boot(W, names, n=200, seed=11):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        sel = {nm: rng.integers(0, len(W[nm]), len(W[nm])) for nm in names}
        b, S, dg, _, _ = pooled(W, names, sel); out.append((b, S, dg))
    return out

def map_fit(fit_csv, A, beta):
    """from C = G f + h (G=(I-S_c)^-1, h=G b_c): variant s = A C + beta -> (S_X, b_X per rec)."""
    b, S, dg, _, _ = fit_csv
    G = np.linalg.inv(I3 - S)
    Gp = A @ G
    Gi = np.linalg.inv(Gp)
    hp = (A @ (G @ b.T)).T + beta
    return I3 - Gi, (Gi @ hp.T).T

if __name__ == "__main__":
    names = MODERATE
    res = {}
    for ctrl in CTRLS:
        print("=" * 110); print(ctrl, "pooled over", names)
        fits = {}
        for sname in ("csv", "base", "e0", "sens"):
            W = load_wins(ctrl, sname, names)
            f = pooled(W, names); bs = boot(W, names)
            fits[sname] = (f, bs)
            b, S, dg, rms, _ = f
            Sb = np.array([x[1] for x in bs]); bb = np.array([x[0] for x in bs])
            print(f"\n[{sname}] rms position misfit {rms*1000:.2f} mm")
            print(f"  S (shared, diag +/- SE) = {np.diag(S)} +/- {Sb.std(0).diagonal()}   offdiag max {np.abs(S - np.diag(np.diag(S))).max():.4f}")
            print(f"  dg = {dg}")
            for j, nm in enumerate(names): print(f"     {nm:14s} b = {b[j]}  +/- {bb[:, j].std(0)}")
            print(f"  mean b {b.mean(0)} sd across recs {b.std(0)}   |b| mean {np.linalg.norm(b, axis=1).mean():.3f}")
        res[ctrl] = {k: v[0] for k, v in fits.items()}
        # ---- hypothesis: driver's hard-coded 10.0 m/s^2 per g conversion  ->  S_csv (isotropic) ~ 1 - 0.980665 = 0.0193
        S = fits["csv"][0][1]; Sb = np.array([x[1] for x in fits["csv"][1]])
        print(f"\n  UNITS TEST  CSV-stream S diag {np.diag(S)} +/- {Sb.std(0).diagonal()}   mean {np.trace(S)/3:.4f}   prediction from 10.0 vs 9.80665: {1 - 9.80665/10.0:.4f}")
        # apply the scale correction s = 0.980665 C  -> what S/b remain?
        S2, b2 = map_fit(res[ctrl]["csv"], SCALE * I3, np.zeros(3))
        print(f"  after multiplying the CSV by 9.80665/10.0:  S diag {np.diag(S2)}  offdiag max {np.abs(S2 - np.diag(np.diag(S2))).max():.4f}   mean b {b2.mean(0)}  |b| mean {np.linalg.norm(b2, axis=1).mean():.3f}")
        # mapping validation against the directly built streams
        for sname, entry, kind in (("base", 1, "loader e1"), ("e0", 0, "loader e0"), ("sens", 1, "sensor-frame D(M1 D C + b1)")):
            _, _, Ma, ba = factory(ctrl, entry)
            A, beta = (D @ Ma @ D, D @ ba) if sname == "sens" else (Ma, ba)
            Sp, bp = map_fit(res[ctrl]["csv"], A, beta)
            b, Sd, _, _, _ = res[ctrl][sname]
            print(f"  mapping check {kind:28s} max|dS| {np.abs(Sp - Sd).max():.4f}  max|db| {np.abs(bp - b).max():.4f}")
    pickle.dump(res, open(FA + "accel_pooled_fits.pkl", "wb"))
