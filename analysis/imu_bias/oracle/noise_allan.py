import sys, csv, pickle
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
import rest_gyro as rg
from common import *
OUT = rg.OUT
def allan(x, dt, ms):
    out = []
    for m in ms:
        n = len(x) // m
        if n < 3: out.append(np.nan); continue
        xm = x[:n*m].reshape(n, m, -1).mean(1)
        out.append(np.sqrt(0.5 * ((np.diff(xm, axis=0)) ** 2).mean(0)).mean())
    return np.array(out)
res = []; ad_g = {}; ad_a = {}
for name in REC_NAMES:
    rdir = rec_dir(name)
    for ctrl in CTRLS:
        dev = load_mocap_device(rdir, ctrl); mo = MocapOrientation(dev)
        t, gb, ab = load_imu(rdir, ctrl); gs = (DIAG_FLIP @ gb.T).T; as_ = (DIAG_FLIP @ ab.T).T
        lo = mo.lookup_times(np.array([t[0]]))[0]; hi = mo.lookup_times(np.array([t[-1]]))[0]
        for a, b in rg.find_rest_runs(dev, min_len_s=1.0, om_thr=0.3, v_thr=0.06):
            a2, b2 = max(a, lo + 0.1e9), min(b, hi - 0.1e9)
            if b2 - a2 < 1.0e9: continue
            shift = mo.lookup_times(np.array([a2]))[0] - a2
            sel = (t >= a2 - shift) & (t <= b2 - shift)
            if sel.sum() < 180: continue
            dt = np.median(np.diff(t[sel])) / 1e9
            g = gs[sel]; f = as_[sel]
            sg = np.sqrt(0.5 * np.var(np.diff(g, axis=0), axis=0)); sa = np.sqrt(0.5 * np.var(np.diff(f, axis=0), axis=0))
            res.append(dict(rec=name, ctrl=ctrl, n=int(sel.sum()), dur_s=(b2-a2)/1e9, dt_ms=dt*1e3, **{f"gyro_sigma_{k}": sg[i] for i, k in enumerate("xyz")}, **{f"accel_sigma_{k}": sa[i] for i, k in enumerate("xyz")}))
            ms = [1, 2, 4, 8, 16, 32, 64]
            ad_g[(name, ctrl, a2)] = (ms, dt, allan(g, dt, ms)); ad_a[(name, ctrl, a2)] = (ms, dt, allan(f, dt, ms))
with open(OUT + "noise_rest_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
G = np.array([[r[f"gyro_sigma_{k}"] for k in "xyz"] for r in res]); A = np.array([[r[f"accel_sigma_{k}"] for k in "xyz"] for r in res]); dts = np.array([r["dt_ms"] for r in res])
print(f"{len(res)} quiet runs (>=1 s inside IMU coverage); median sample interval {np.median(dts):.2f} ms (fs={1e3/np.median(dts):.0f} Hz)")
print("white-noise sigma per sample (lag-1 differences), median over runs: gyro [rad/s]", np.median(G, 0).round(5), "=", np.degrees(np.median(G, 0)).round(4), "deg/s ; accel [m/s^2]", np.median(A, 0).round(4))
fs = 1e3 / np.median(dts)
print(f"=> density assuming BW=fs/2={fs/2:.0f} Hz:  gyro {np.degrees(np.median(G,0)/np.sqrt(fs/2))*1e3} mdps/rtHz ; accel {np.median(A,0)/np.sqrt(fs/2)/9.80665*1e6} ug/rtHz")
print("datasheet class (ICM-20602): gyro ~4 mdps/rtHz, accel ~100 ug/rtHz  (InvenSense datasheet)")
# factory Noise field
for c in CTRLS:
    cfg = load_json_config(str(REPO / CONFIG["controllers"][c]["config_path"])); calib = create_imu_calib_from_config(cfg)
    print(f"factory 'Noise' field {c}: gyro {calib.gyro.noise_std} accel {calib.accel.noise_std}; BiasUncertainty gyro {calib.gyro.bias_uncertainty} accel {calib.accel.bias_uncertainty}; factory bias0 gyro {calib.gyro.bias0} accel {calib.accel.bias0}")
# Allan
print("Allan deviation median over runs (tau = m*dt):")
ms = [1, 2, 4, 8, 16, 32, 64]
for nm, ad in (("gyro [mrad/s]", ad_g), ("accel [mm/s^2]", ad_a)):
    M = np.array([v[2] for v in ad.values()]); sc = 1e3
    print(f"  {nm}: tau[s]={np.round(np.array(ms)*np.median(dts)/1e3,3)}  ADEV={np.nanmedian(M,0)*sc}")
