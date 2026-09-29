import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *
from fa_accel import map_fit, SCALE
np.set_printoptions(precision=3, suppress=True, linewidth=180)
I3 = np.eye(3)
fits = pickle.load(open(FA + "accel_pooled_fits.pkl", "rb"))

def Rrt(ctrl, entry):
    cfg = load_json_config(str(REPO / CONFIG["controllers"][ctrl]["config_path"]))
    S = [s for s in cfg["CalibrationInformation"]["InertialSensors"] if s["SensorType"].endswith("Accelerometer")][entry]
    return np.array(S["Rt"]["Rotation"]).reshape(3, 3)

def family(ctrl):
    out = []
    for e in (0, 1):
        _, _, Ma, ba = factory(ctrl, e); R = Rrt(ctrl, e)
        Pset = {"I": I3, "D": D, "R": R, "Rt": R.T, "DR": D @ R, "DRt": D @ R.T, "RD": R @ D, "RtD": R.T @ D}
        for pn, P in Pset.items():
            for an, A0 in (("I", I3), ("M", Ma), ("Minv", np.linalg.inv(Ma))):
                for sn, sc in (("s1", 1.0), ("s.98", SCALE)):
                    for bn, sg in (("0", 0.0), ("+b", 1.0), ("-b", -1.0)):
                        if pn != "I" and an == "I" and bn == "0": continue
                        A = sc * (P @ A0 @ P.T); beta = P @ (sg * ba)
                        out.append((f"e{e} P={pn} A={an} {sn} b={bn}", A, beta))
    return out

res = {}
for ctrl in CTRLS:
    r = {}
    for name, A, beta in family(ctrl):
        S, b = map_fit(fits[ctrl]["csv"], A, beta)
        r[name] = (np.linalg.norm(b, axis=1).mean(), b.mean(0), np.diag(S), np.linalg.norm(S))
    res[ctrl] = r
common = sorted(set(res["left_controller"]) & set(res["right_controller"]))
def sc(n):   # combined: mean |b|/0.1 m/s^2 + |S|_F/0.01
    return sum(res[c][n][0] / 0.1 + res[c][n][3] / 0.01 for c in CTRLS) / 2
rank = sorted(common, key=sc)
print(f"{len(common)} accel variants (same rule both controllers).  score = mean(|b|/0.1 + |S|_F/0.01).  |b| in m/s^2, S dimensionless")
print(f"{'variant':34s} {'|b| L':>7s} {'|b| R':>7s} {'diagS L':>22s} {'diagS R':>22s} {'score':>6s}")
def line(n): return f"{n:34s} {res['left_controller'][n][0]:7.3f} {res['right_controller'][n][0]:7.3f} {str(res['left_controller'][n][2]):>22s} {str(res['right_controller'][n][2]):>22s} {sc(n):6.2f}"
for n in rank[:10]: print(line(n))
print("  ... references:")
for n in ("e1 P=I A=I s1 b=0", "e1 P=I A=M s1 b=+b", "e0 P=I A=M s1 b=+b", "e1 P=I A=I s.98 b=0", "e1 P=I A=M s.98 b=+b"):
    print(line(n))
with open(FA + "accel_variants.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["variant", "absb_left", "absb_right", "bmean_left", "bmean_right", "diagS_left", "diagS_right", "SF_left", "SF_right", "score"])
    for n in rank:
        L, R = res["left_controller"][n], res["right_controller"][n]
        w.writerow([n, f"{L[0]:.4f}", f"{R[0]:.4f}", L[1].round(4).tolist(), R[1].round(4).tolist(), L[2].round(4).tolist(), R[2].round(4).tolist(), f"{L[3]:.4f}", f"{R[3]:.4f}", f"{sc(n):.3f}"])
print("wrote accel_variants.csv", len(rank))
# needed correction to zero the bias vs candidate factory vectors (both controllers): beta* = -h = -G b_csv
print("\nneeded additive correction beta* (CSV axes) to zero the bias  vs  the factory bias vectors:")
for ctrl in CTRLS:
    b, S, _, _, _ = fits[ctrl]["csv"]; G = np.linalg.inv(I3 - S); need = -(G @ b.mean(0))
    print(f"{ctrl}: beta* = {need.round(3)}   entry1 b = {factory(ctrl,1)[3].round(3)}   entry0 b = {factory(ctrl,0)[3].round(3)}")
