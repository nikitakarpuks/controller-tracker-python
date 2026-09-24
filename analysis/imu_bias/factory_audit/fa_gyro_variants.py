"""Gyro variant table.  From the pooled fit of the RAW CSV stream C = (I+K0) w + b0f (per-recording b0f, shared K0), every affine
correction s = A C + beta has K_X = A (I+K0) - I,  b_X = A b0f + beta  (exact under the linear model).  Validated against directly-built streams."""
import sys, pickle, itertools
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *
np.set_printoptions(precision=4, suppress=True, linewidth=170)
fits = pickle.load(open(FA + "gyro_pooled_fits.pkl", "rb"))
I3 = np.eye(3)

def Rrt(ctrl, entry):
    cfg = load_json_config(str(REPO / CONFIG["controllers"][ctrl]["config_path"]))
    S = [s for s in cfg["CalibrationInformation"]["InertialSensors"] if s["SensorType"].endswith("Gyro")][entry]
    return np.array(S["Rt"]["Rotation"]).reshape(3, 3)

def map_fit(ctrl, A, beta):
    b0, K0, _, _ = fits[ctrl]["csv"]
    return A @ (I3 + K0) - I3, (A @ b0.T).T + beta            # K_X (3,3), b_X (R,3)

print("== mapping validation: predicted from the CSV fit vs the directly-built stream fit (pooled, 6 moderate recs)")
for ctrl in CTRLS:
    for sname, entry, kind in (("base", 1, "loader e1"), ("e0", 0, "loader e0"), ("sens", 1, "sensor-frame D(M1 D C + b1)")):
        Mg, bg, _, _ = factory(ctrl, entry)
        if sname == "sens":
            A = D @ Mg @ D; beta = D @ bg
        else:
            A = Mg; beta = bg
        Kp, bp = map_fit(ctrl, A, beta)
        b, K, _, _ = fits[ctrl][sname]
        print(f"{ctrl[:5]} {kind:28s} max|dK| {np.abs(Kp - K).max():.4f}  max|db| {np.abs(bp - b).max():.4f}   (bootstrap SE: K 0.0006, b 0.001-0.003)")

# ---- variant family
def variants(ctrl):
    out = []
    Pset = {"I": I3, "D": D}
    for e in (0, 1):
        R = Rrt(ctrl, e)
        Pset.update({f"R{e}": R, f"Rt{e}": R.T, f"DR{e}": D @ R, f"DRt{e}": D @ R.T, f"RD{e}": R @ D, f"RtD{e}": R.T @ D})
    for e in (0, 1):
        Mg, bg, _, _ = factory(ctrl, e)
        for pname, P in Pset.items():
            if pname.endswith(str(1 - e)) and pname not in ("I", "D"):   # P built from the other entry's Rt: skip (keeps family physical)
                continue
            for aname, A0 in (("I", I3), ("M", Mg), ("Minv", np.linalg.inv(Mg))):
                for bname, sgn in (("0", 0.0), ("+b", 1.0), ("-b", -1.0)):
                    for order in ("mix_then_bias", "bias_then_mix"):
                        if (aname == "I" and order == "bias_then_mix") or (bname == "0" and order == "bias_then_mix"):
                            continue
                        A = P @ A0 @ P.T
                        beta_s = sgn * bg
                        beta = P @ beta_s if order == "mix_then_bias" else P @ (A0 @ beta_s)
                        out.append((f"e{e} P={pname} A={aname} b={bname} {order[:3]}", A, beta))
    return out

allres = {}
for ctrl in CTRLS:
    res = []
    for name, A, beta in variants(ctrl):
        K, b = map_fit(ctrl, A, beta)
        res.append((name, np.linalg.norm(b, axis=1).mean(), b.mean(0), b.std(0), np.diag(K), np.abs(K).max(), np.linalg.norm(K)))
    allres[ctrl] = {r[0]: r for r in res}
common = sorted(set(allres["left_controller"]) & set(allres["right_controller"]))
# a variant only counts if the SAME rule is applied to both controllers
score = sorted(common, key=lambda n: 0.5 * (allres["left_controller"][n][1] + allres["right_controller"][n][1]))
print(f"\n== {len(common)} variants (same rule on both controllers), ranked by mean |b_X| (rad/s), both controllers averaged. Reference: no correction (csv) and loader (e1 P=I A=M b=+b mix)")
print(f"{'variant':44s} {'|b| L':>7s} {'|b| R':>7s} {'b_mean L':>26s} {'b_mean R':>26s} {'diagK L':>24s} {'diagK R':>24s} {'|K|F L/R':>12s}")
def line(n):
    L, R = allres["left_controller"][n], allres["right_controller"][n]
    return f"{n:44s} {L[1]:7.4f} {R[1]:7.4f} {str(L[2]):>26s} {str(R[2]):>26s} {str(L[4]):>24s} {str(R[4]):>24s} {L[6]:5.4f}/{R[6]:5.4f}"
for n in score[:12]: print(line(n))
print("   ...")
for ref in ("e1 P=I A=I b=0 mix", "e1 P=I A=M b=+b mix", "e0 P=I A=M b=+b mix"):
    print(line(ref))
pickle.dump(allres, open(FA + "gyro_variants.pkl", "wb"))
import csv
with open(FA + "gyro_variants.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["variant", "absb_left", "absb_right", "bmean_left", "bmean_right", "diagK_left", "diagK_right", "Kfro_left", "Kfro_right"])
    for n in score:
        L, R = allres["left_controller"][n], allres["right_controller"][n]
        w.writerow([n, f"{L[1]:.5f}", f"{R[1]:.5f}", L[2].round(5).tolist(), R[2].round(5).tolist(), L[4].round(5).tolist(), R[4].round(5).tolist(), f"{L[6]:.5f}", f"{R[6]:.5f}"])
print("\nwrote gyro_variants.csv with", len(score), "variants")
