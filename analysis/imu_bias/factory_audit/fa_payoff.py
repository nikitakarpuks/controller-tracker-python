"""Payoff: score candidate correction streams against mocap with the oracle's own scoring loops (same start sampling, same metrics).
 gyro: rotation prediction error over gap T  (angle(Rm^T Rg));   accel: position dead-reckoning error over T from a MOCAP initial state + mocap orientation."""
import sys, pickle, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit")
from fa_common import *
from fa_accel import pooled as accel_pooled, load_wins, SCALE
from fa_gyro import pooled as gyro_pooled, load_stream
np.set_printoptions(precision=4, suppress=True, linewidth=170)
G0 = AO.G0
GT = [0.044, 0.1, 0.3, 1.0]; AT = [0.1, 0.3, 1.0]
NST = 200

def gyro_payoff():
    variants = ["G0_loader(e1 mix+bias)", "G1_csv_as_is", "G2_csv - const bias(per-rec oracle)", "G3_csv - ONE bias(all recs)", "G4_csv - bias - K (per-rec oracle)"]
    out = {v: {T: [] for T in GT} for v in variants}
    for ctrl in CTRLS:
        ds = load_stream(ctrl, "csv", MODERATE); b, K, _ = gyro_pooled(ds, MODERATE)
        b_one = np.median(b, 0)
        Mg, bg, _, _ = factory(ctrl, 1)
        for j, name in enumerate(MODERATE):
            rdir = rec_dir(name); mo = MocapOrientation(load_mocap_device(rdir, ctrl))
            t, gc, _ = raw_csv(rdir, ctrl)
            streams = {variants[0]: apply(gc, Mg, bg), variants[1]: gc, variants[2]: gc - b[j], variants[3]: gc - b_one, variants[4]: gc - gc @ K.T - b[j]}
            rng = np.random.default_rng(100 + j)
            for T in GT:
                dtn = int(T * 1e9)
                cand = np.arange(t[0] + 0.5e9, t[-1] - dtn - 0.5e9, 1e7).astype(np.int64)
                ok = mo.valid(cand) & mo.valid(cand + dtn); cand = cand[ok]
                starts = rng.choice(cand, min(NST, len(cand)), replace=False)
                R0 = mo.R_world_imu(starts); R1 = mo.R_world_imu(starts + dtn)
                for i, s0 in enumerate(starts):
                    Rm = (R0[i].inv() * R1[i]).as_matrix()
                    for v in variants:
                        Rg = integrate_gyro_segment(t, streams[v], int(s0), int(s0) + dtn)
                        out[v][T].append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rm.T @ Rg).as_rotvec())))
        print(f"gyro {ctrl} done", flush=True)
    return variants, out

def accel_payoff():
    variants = ["A0_loader(e1 mix+bias)", "A1_csv_as_is", "A2_csv * 9.80665/10", "A3_csv*scale - bias(per-rec oracle)", "A4_loader * 9.80665/10", "A5_csv oracle bias+S+dg (ceiling)"]
    out = {v: {T: [] for T in AT} for v in variants}
    for ctrl in CTRLS:
        W = load_wins(ctrl, "csv", MODERATE); b, S, dg, _, _ = accel_pooled(W, MODERATE)
        _, _, Ma, ba = factory(ctrl, 1)
        for j, name in enumerate(MODERATE):
            rdir = rec_dir(name); dev = load_mocap_device(rdir, ctrl); mo = AO.MocapPose(dev)
            t, _, ac = raw_csv(rdir, ctrl)
            f_load = apply(ac, Ma, ba)
            # bias defined by f_true = (I-S) C - b ; for A3 (scale-only + bias) re-express with A=scale: b_X = b (unchanged, see fa_accel.map_fit)
            streams = {variants[0]: (f_load, G0), variants[1]: (ac, G0), variants[2]: (SCALE * ac, G0), variants[3]: (SCALE * ac - b[j], G0),
                       variants[4]: (SCALE * f_load, G0), variants[5]: (ac - b[j] - ac @ S.T, G0 + dg)}
            rng = np.random.default_rng(200 + j)
            for T in AT:
                dtn = int(T * 1e9)
                cand = np.arange(t[0] + 0.5e9, t[-1] - dtn - 0.5e9, 2e7).astype(np.int64)
                ok = mo.valid(cand) & mo.valid(cand + dtn) & mo.valid(cand - int(0.05e9)) & mo.valid(cand + int(0.05e9)); cand = cand[ok]
                starts = rng.choice(cand, min(NST, len(cand)), replace=False)
                for s0 in starts:
                    tq = (s0 + np.linspace(-0.04e9, 0.04e9, 17)).astype(np.int64); pq = mo.pos_imu(tq); tt = (tq - s0) / 1e9
                    v0 = np.array([np.polyfit(tt, pq[:, i], 1)[0] for i in range(3)]); p0 = np.array([np.polyval(np.polyfit(tt, pq[:, i], 1), 0) for i in range(3)])
                    sel = np.flatnonzero((t >= s0) & (t <= s0 + dtn))
                    tI = np.concatenate(([s0], t[sel], [s0 + dtn])).astype(np.int64)
                    RI = mo.R_world_imu(tI).as_matrix(); ts = (tI - tI[0]) / 1e9
                    p_true = mo.pos_imu(np.array([s0 + dtn]))[0]
                    for v, (fs, g) in streams.items():
                        fI = np.stack([np.interp(tI, t, fs[:, i]) for i in range(3)], 1)
                        a_w = np.einsum("nij,nj->ni", RI, fI) + g
                        p = p0 + v0 * ts[-1] + AO.cum2(ts, a_w)[-1]
                        out[v][T].append(np.linalg.norm(p - p_true) * 1000)
        print(f"accel {ctrl} done", flush=True)
    return variants, out

def table(variants, out, Ts, unit, fname):
    rows = []
    print(f"\n{'variant':40s}" + "".join(f"T={T:<10}" for T in Ts) + f"   (median {unit}; mean in brackets)")
    for v in variants:
        line = f"{v:40s}"
        for T in Ts:
            e = np.array(out[v][T]); line += f"{np.median(e):6.3f} [{e.mean():6.3f}] "
            rows.append(dict(variant=v, T_s=T, n=len(e), median=np.median(e), mean=e.mean(), p90=np.percentile(e, 90)))
        print(line)
    with open(FA + fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gyro":
        v, o = gyro_payoff(); table(v, o, GT, "deg", "payoff_gyro.csv")
    else:
        v, o = accel_payoff(); table(v, o, AT, "mm", "payoff_accel.csv")
