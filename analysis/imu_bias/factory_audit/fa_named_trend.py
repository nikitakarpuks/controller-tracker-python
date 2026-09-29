"""(1) the task's NAMED variants (both sensors), (2) per-recording generalization of the two proposed fixes, (3) drift of the residual bias with session time (temperature proxy)."""
import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *
from fa_accel import pooled as accel_pooled, boot as accel_boot, load_wins, map_fit as amap, SCALE
np.set_printoptions(precision=4, suppress=True, linewidth=170)
I3 = np.eye(3)
gv = pickle.load(open(FA + "gyro_variants.pkl", "rb")); gfits = pickle.load(open(FA + "gyro_pooled_fits.pkl", "rb")); afits = pickle.load(open(FA + "accel_pooled_fits.pkl", "rb"))

# ---------- 1. named variants ----------
names = [("none (CSV as recorded)", None, "I", "0"), ("mix only", None, "M", "0"), ("+bias only", None, "I", "+b"), ("-bias only", None, "I", "-b"),
         ("mix+bias  [= project loader]", None, "M", "+b"), ("mix + (-bias)", None, "M", "-b"), ("inverse mix only", None, "Minv", "0"), ("inverse mix + bias", None, "Minv", "+b")]
rows = []
print("== NAMED VARIANTS (frame = CSV axes, P=I) -- mean over 6 moderate recordings; gyro |b| rad/s, gyro mean gain err (trace(K)/3); accel |b| m/s^2, accel mean gain err (trace(S)/3)")
print(f"{'entry':5s} {'variant':30s} | {'gyro|b| L':>9s} {'R':>7s} {'trK/3 L':>8s} {'R':>7s} | {'accel|b| L':>10s} {'R':>7s} {'trS/3 L':>8s} {'R':>7s}")
for e in (0, 1):
    for label, _, an, bn in names:
        if an == "I" and bn == "0" and e == 0: continue
        line = f"e{e:<4d} {label:30s} |"
        rec = dict(entry=e, variant=label)
        for sensor in ("gyro", "accel"):
            for ctrl in CTRLS:
                Mg, bg, Ma, ba = factory(ctrl, e); M, b0 = (Mg, bg) if sensor == "gyro" else (Ma, ba)
                A = {"I": I3, "M": M, "Minv": np.linalg.inv(M)}[an]; beta = {"0": 0 * b0, "+b": b0, "-b": -b0}[bn]
                if sensor == "gyro":
                    b_, K_, _, _ = gfits[ctrl]["csv"]; Kx = A @ (I3 + K_) - I3; bx = (A @ b_.T).T + beta; g = np.trace(Kx) / 3
                else:
                    S_, bx = amap(afits[ctrl]["csv"], A, beta); g = np.trace(S_) / 3
                rec[f"{sensor}_absb_{ctrl[:1]}"] = float(np.linalg.norm(bx, axis=1).mean()); rec[f"{sensor}_gain_{ctrl[:1]}"] = float(g)
        line += f" {rec['gyro_absb_l']:9.4f} {rec['gyro_absb_r']:7.4f} {rec['gyro_gain_l']:8.4f} {rec['gyro_gain_r']:7.4f} | {rec['accel_absb_l']:10.3f} {rec['accel_absb_r']:7.3f} {rec['accel_gain_l']:8.4f} {rec['accel_gain_r']:7.4f}"
        print(line); rows.append(rec)
with open(FA + "named_variants.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------- 2. per-recording generalization of the two proposed fixes (fit-free where possible) ----------
print("\n== PER-RECORDING GENERALIZATION")
print("gyro: median |rotation error| over 0.5 s intervals (deg), loader vs CSV-as-is  (no fitting; straight from the interval residuals r0)")
better = 0; tot = 0
gen = []
for ctrl in CTRLS:
    for n in MODERATE:
        d = pickle.load(open(FA + f"terms_gyro_{n}_{ctrl}.pkl", "rb"))
        a = np.degrees(np.median(np.linalg.norm(d["base"]["r0"], axis=1))); c = np.degrees(np.median(np.linalg.norm(d["csv"]["r0"], axis=1)))
        better += c < a; tot += 1; gen.append(dict(sensor="gyro", ctrl=ctrl, rec=n, loader=a, fixed=c, rel=(c - a) / a))
        print(f"  {ctrl[:5]} {n:14s} loader {a:.3f}  csv {c:.3f}  ({100*(c-a)/a:+.1f}%)")
print(f"  CSV-as-is better in {better}/{tot} recording-controller pairs")
print("accel: zero-parameter position misfit (rms mm over 2 s windows, p0/v0 projected) -- loader vs CSV vs CSV*9.80665/10")
better = better2 = tot = 0
for ctrl in CTRLS:
    for n in MODERATE:
        st = pickle.load(open(FA + f"terms_accel_{n}_{ctrl}.pkl", "rb"))
        def rms(w): return np.sqrt(sum(x["yty"] for x in w) / sum(x["n"] for x in w)) * 1000
        rdir = rec_dir(n); t, _, ac = raw_csv(rdir, ctrl)
        wsc = accel_windows_stream(n, ctrl, t, SCALE * ac)
        a, c, s = rms(st["base"]), rms(st["csv"]), rms(wsc)
        better += c < a; better2 += s < c; tot += 1; gen.append(dict(sensor="accel", ctrl=ctrl, rec=n, loader=a, csv=c, csv_scaled=s))
        print(f"  {ctrl[:5]} {n:14s} loader {a:6.2f}  csv {c:6.2f}  csv*scale {s:6.2f}")
print(f"  CSV-as-is < loader in {better}/{tot};  CSV*scale < CSV in {better2}/{tot}")
with open(FA + "generalization.csv", "w", newline="") as f:
    keys = sorted({k for r in gen for k in r}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(gen)

# ---------- 3. drift of the residual bias (CSV stream) with session time ----------
T_MIN = {"static_dark": 0.0, "walk_dark": 2 + 47/60, "static_easy": 8 + 29/60, "static_medium": 11 + 10/60, "walk_easy": 21 + 23/60, "walk_medium": 24 + 7/60}   # minutes after 17:31:03 (folder names)
tm = np.array([T_MIN[n] for n in MODERATE])
print("\n== residual bias of the CSV stream vs session time (weighted linear fit; slope per minute; bootstrap SE per recording)")
def wfit(y, se):
    w = 1 / se**2; A = np.stack([np.ones_like(tm), tm - tm.mean()], 1); W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A); th = cov @ A.T @ W @ y; chi2 = ((y - A @ th)**2 * w).sum()
    return th[1], np.sqrt(cov[1, 1]), chi2
for ctrl in CTRLS:
    b, K, bb, KK = gfits[ctrl]["csv"]; se = bb.std(0)
    print(f"{ctrl} gyro (rad/s per min):", "  ".join(f"{a}: {s:+.5f} +/- {e:.5f} (chi2/dof {c/4:.1f})" for a, (s, e, c) in zip("xyz", [wfit(b[:, i], se[:, i]) for i in range(3)])))
for ctrl in CTRLS:
    W = load_wins(ctrl, "csv", MODERATE); f = accel_pooled(W, MODERATE); bs = accel_boot(W, MODERATE, n=200); se = np.array([x[0] for x in bs]).std(0)
    b = f[0]
    print(f"{ctrl} accel (m/s^2 per min):", "  ".join(f"{a}: {s:+.5f} +/- {e:.5f} (chi2/dof {c/4:.1f})" for a, (s, e, c) in zip("xyz", [wfit(b[:, i], se[:, i]) for i in range(3)])))
