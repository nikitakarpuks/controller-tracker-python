"""Gyro: (b_rec, shared K) fits for each stream, analytic-mapping validation, and the variant table."""
import sys, pickle, csv, itertools
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *
np.set_printoptions(precision=4, suppress=True, linewidth=160)

def load_stream(ctrl, sname, names):
    return {n: pickle.load(open(FA + f"terms_gyro_{n}_{ctrl}.pkl", "rb"))[sname] for n in names}

def design(ds, masks, names):
    R = len(names); rows_A, rows_y, rec_of = [], [], []
    for j, n in enumerate(names):
        d = ds[n]; m = masks[n]
        Jb, JK, r0 = d["Jb"][m], d["JK"][m], d["r0"][m]
        A = np.zeros((len(r0), 3, 3 * R + 9)); A[:, :, 3 * j:3 * j + 3] = Jb; A[:, :, 3 * R:] = JK
        rows_A.append(A.reshape(-1, 3 * R + 9)); rows_y.append((-r0).reshape(-1)); rec_of += [j] * (3 * len(r0))
    return np.concatenate(rows_A), np.concatenate(rows_y), np.array(rec_of)

def pooled(ds, names, iters=3):
    """shared K (9), per-recording b (3); robust 4-MAD interval trimming (starts from per-rec K-free trims like the oracle)."""
    masks = {}
    for n in names:
        idx = np.arange(len(ds[n]["t0"])); _, _, k = GO.solve(ds[n], idx, with_K=True); masks[n] = k.copy()
    for _ in range(iters):
        A, y, _ = design(ds, masks, names); th, *_ = np.linalg.lstsq(A, y, rcond=None)
        R = len(names)
        for j, n in enumerate(names):
            d = ds[n]; pred = d["Jb"] @ th[3 * j:3 * j + 3] + d["JK"] @ th[3 * R:]
            res = d["r0"] + pred                              # residual after the fit for ALL intervals
            s = 1.4826 * np.median(np.abs(res[masks[n]])) + 1e-12
            masks[n] = (np.abs(res) < 4 * s).all(1)
    A, y, _ = design(ds, masks, names); th, *_ = np.linalg.lstsq(A, y, rcond=None)
    R = len(names)
    return th[:3 * R].reshape(R, 3), th[3 * R:].reshape(3, 3), masks

def boot(ds, names, n=200, block_s=10.0, seed=7):
    rng = np.random.default_rng(seed); R = len(names)
    # per-recording block resampling of intervals, refit pooled (masks from full fit reused to keep it fast)
    b0, K0, masks = pooled(ds, names)
    Ks, bs = [], []
    blocks = {}
    for nm in names:
        sec = ((ds[nm]["t0"] - ds[nm]["t0"][0]) / 1e9 / block_s).astype(int); blocks[nm] = (sec, np.unique(sec))
    for _ in range(n):
        dsb, mk = {}, {}
        for nm in names:
            sec, ub = blocks[nm]; pick = rng.choice(ub, len(ub)); idx = np.concatenate([np.where(sec == u)[0] for u in pick])
            dsb[nm] = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == sec.shape else v) for k, v in ds[nm].items()}
            mk[nm] = masks[nm][idx]
        A, y, _ = design(dsb, mk, names); th, *_ = np.linalg.lstsq(A, y, rcond=None)
        bs.append(th[:3 * R].reshape(R, 3)); Ks.append(th[3 * R:].reshape(3, 3))
    return np.array(bs), np.array(Ks)

if __name__ == "__main__":
    names = MODERATE
    out = {}
    for ctrl in CTRLS:
        print("=" * 100); print(ctrl, " pooled over", names)
        fits = {}
        for sname in ("csv", "base", "e0", "sens"):
            ds = load_stream(ctrl, sname, names)
            b, K, masks = pooled(ds, names)
            bb, KK = boot(ds, names)
            fits[sname] = (b, K, bb, KK)
            print(f"\n[{sname}]  shared K  (bootstrap SE in parens on diag):\n{K}\n  diag SE {KK.std(0).diagonal()}   offdiag SE max {np.abs(KK.std(0) - np.diag(np.diag(KK.std(0)))).max():.4f}")
            print("  per-recording b (rad/s) [x y z]  + bootstrap SE:")
            for j, nm in enumerate(names): print(f"     {nm:14s} {b[j]}  +/- {bb[:, j].std(0)}")
            print(f"  mean over recs {b.mean(0)}   sd across recs {b.std(0)}   |b| mean {np.linalg.norm(b, axis=1).mean():.4f}")
        out[ctrl] = fits
    pickle.dump(out, open(FA + "gyro_pooled_fits.pkl", "wb"))
