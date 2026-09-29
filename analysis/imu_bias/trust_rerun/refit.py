"""Trust-duration crossings for the frame-correct, fully-new pipeline (CLED) at several tolerances and both median
and p90 aggregates, per fine gyro/accel bin, plus a proposed re-fit of the shipped budget formulas (numbers only).
Same crossing() as the thesis script (log-dt interpolation)."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tf", "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/TUM-THESIS/scripts/make_imu_trust_figure.py")
tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
variant = sys.argv[1] if len(sys.argv) > 1 else "CLED"
raw = pd.read_csv(HERE / "out" / variant / "imu_trust_all.csv")
dts = sorted(raw.dt_s.unique())
def curve(sub, err, q):
    out = []
    for dt in dts:
        s = sub[sub.dt_s == dt][err]
        if len(s) >= 20: out.append((dt, float(np.percentile(s, q))))
    return zip(*out)
def cross(sub, err, q, tol):
    d, v = curve(sub, err, q); return tf.crossing(list(d), list(v), tol)
def f(x): return "inf" if not np.isfinite(x) else f"{x*1e3:.0f}"
ROT = [(1.0, 50), (2.0, 50), (5.0, 50), (5.0, 90), (10.0, 90)]
POS = [(20, 50), (50, 50), (100, 50), (100, 90), (200, 90)]
rows = []
print(f"[{variant}] ROTATION trust duration (ms; inf = not exceeded within 1 s). columns: tol deg @ percentile")
print(f"{'bin(deg/s)':<14s}{'n':>7s}{'peak':>8s}" + "".join(f"{f'{t:g}@p{q}':>9s}" for t, q in ROT))
for lo, hi in zip(tf.FINE_GYRO[:-1], tf.FINE_GYRO[1:]):
    sub = raw[(raw.peak_gyro_dps >= lo) & (raw.peak_gyro_dps < hi)]
    if len(sub) < 200: continue
    pk = sub.peak_gyro_dps.median(); cs = [cross(sub, "rot_err_deg", q, t) for t, q in ROT]
    print(f"{lo:>5g}-{hi if hi<1e8 else 'inf':<8}{len(sub):>7d}{pk:>8.0f}" + "".join(f"{f(c):>9s}" for c in cs))
    rows.append(("gyro", pk, *cs))
print(f"\n[{variant}] POSITION trust duration (ms). columns: tol mm @ percentile")
print(f"{'bin(m/s2)':<14s}{'n':>7s}{'peak':>8s}" + "".join(f"{f'{t:g}@p{q}':>9s}" for t, q in POS))
for lo, hi in zip(tf.FINE_ACC[:-1], tf.FINE_ACC[1:]):
    sub = raw[(raw.peak_accel_mps2 >= lo) & (raw.peak_accel_mps2 < hi)]
    if len(sub) < 200: continue
    pk = sub.peak_accel_mps2.median(); cs = [cross(sub, "pos_err_mm", q, t) for t, q in POS]
    print(f"{lo:>5g}-{hi if hi<1e8 else 'inf':<8}{len(sub):>7d}{pk:>8.1f}" + "".join(f"{f(c):>9s}" for c in cs))
    rows.append(("accel", pk, *cs))
# BUG FIXED 2026-09-27: this used to name every column after ROT's tolerances (1/2/5/5/10 deg) regardless of
# `kind`, so an accel row's actual POS-tolerance crossings (20/50/100/100/200 mm) ended up stored under
# degree-labelled column names -- positionally matched, not value-matched. Two separate frames instead, each
# with its own correctly-labelled tolerance columns; concat with NaN for the other kind's columns so the CSV
# shape stays a drop-in replacement for any reader that expects one row per (kind, peak) pair.
_gyro_df = pd.DataFrame([r for r in rows if r[0] == "gyro"],
                        columns=["kind", "peak", *[f"c_{t:g}deg_p{q}" for t, q in ROT]])
_acc_df = pd.DataFrame([r for r in rows if r[0] == "accel"],
                       columns=["kind", "peak", *[f"c_{t:g}mm_p{q}" for t, q in POS]])
pd.concat([_gyro_df, _acc_df], ignore_index=True).to_csv(HERE / "out" / variant / "crossings_multi_tol.csv", index=False)

# budget re-fits: budget(x) = max(min, base - k*x), least squares on safety * crossing over bins where the crossing is finite
def fit(x, c, sf, floor_ms):
    ok = np.isfinite(c) & (c > 0)
    A = np.vstack([np.ones(ok.sum()), -x[ok]]).T
    base, k = np.linalg.lstsq(A, sf * c[ok], rcond=None)[0]
    return base, k
pos = [r for r in rows if r[0] == "accel"]; a = np.array([r[1] for r in pos])
col = {"20mm@p50": 2, "50mm@p50": 3, "100mm@p50": 4, "100mm@p90": 5}
print("\nPOSITION budget candidates: budget(a)=max(floor, base - k*a).  shipped: base 66 ms, k 0.001145 s/(m/s2), floor 35 ms -> 65/50/35/35 ms at a=0.5/14/49/77")
for name, ci in col.items():
    c = np.array([r[ci] for r in pos]) * 1e3
    for sf in (0.5, 0.7):
        base, k = fit(a, c, sf, 35.0)
        print(f"  target {name:>9s} x safety {sf}: base {base:6.1f} ms, k {k/1e3:.6f} s/(m/s2) -> budget at a=0.5/14/49/77: "
              + "/".join(f"{max(35.0, base - k * x):.0f}" for x in (0.5, 14, 49, 77)) + " ms")
gy = [r for r in rows if r[0] == "gyro"]; g = np.array([r[1] for r in gy])
gcol = {"1deg@p50": 2, "2deg@p50": 3, "5deg@p50": 4}
print("\nROTATION budget candidates (finite-crossing bins only; inf bins = the budget may be capped by base/extend): budget(g)=max(floor, base - k*g)")
print("  shipped: base 66 ms (+ calm extend to 150 ms below 100 deg/s), k 0.00018 s/(deg/s), floor 3 ms -> " + "/".join(f"{float(tf.rot_budget(x))*1e3:.0f}" for x in (26, 126, 297, 495, 862, 1190)) + " ms at g=26/126/297/495/862/1190")
for name, ci in gcol.items():
    c = np.array([r[ci] for r in gy]) * 1e3
    for sf in (0.5, 0.7):
        if np.isfinite(c).sum() < 3: continue
        base, k = fit(g, c, sf, 3.0)
        print(f"  target {name:>9s} x safety {sf}: base {base:6.1f} ms, k {k/1e3:.6f} s/(deg/s) -> budget at g=26/126/297/495/862/1190: "
              + "/".join(f"{max(3.0, base - k * x):.0f}" for x in (26, 126, 297, 495, 862, 1190)) + " ms")
