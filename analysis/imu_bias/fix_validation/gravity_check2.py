"""gravity_check2.py -- independent validation of the factory-audit long-window accel model (gain S, bias b, gravity correction dg) on
STRICTLY QUIET windows the audit never used to fit (only used as a held-out consistency check; the fit used moving windows):
    C = G (f_true + b),  G = (I - S)^-1,  f_true = -R^T (g_room + dg),  g_room = (0,-9.81,0)     (fa_accel.map_fit conventions)
For every quiet window (same definition as gravity_check.py) compare the measured window-mean CSV vector with the model prediction, and with
two simpler alternatives: (A) 'ideal' C = 10/9.80665 * 9.80665 u  (driver constant only, no bias) and (B) model with b = 0."""
import pickle
import numpy as np
import pandas as pd
import fv_common as C

FIT = pickle.load(open("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/factory_audit/accel_pooled_fits.pkl", "rb"))
G0v = np.array([0.0, -9.81, 0.0]); W = 0.25e9; G0 = 9.80665
rows = []
for ctrl in C.CTRLS:
    b_all, S, dg, _, _ = FIT[ctrl]["csv"]
    Gm = np.linalg.inv(np.eye(3) - S)
    for j, name in enumerate(C.MODERATE):
        rd = C.rec_dir(name); dev = C.load_mocap_device(rd, ctrl); mo = C.MocapPose(dev)
        t, g_csv, a_csv = C.raw_csv(rd, ctrl)
        Cm, Cp, Cp_nob, Ci, RTs = [], [], [], [], []
        for s in np.arange(t[0] + 0.5e9, t[-1] - W - 0.5e9, 0.25e9).astype(np.int64):
            q = np.array([s, s + int(W / 2), s + int(W)], dtype=np.int64)
            if not mo.valid(q).all(): continue
            tq = (s + np.linspace(0, W, 19)).astype(np.int64)
            if not mo.valid(tq).all(): continue
            pq = mo.pos_imu(tq); tt = (tq - s) / 1e9
            v1 = np.array([np.polyfit(tt[:10], pq[:10, i], 1)[0] for i in range(3)]); v2 = np.array([np.polyfit(tt[9:], pq[9:, i], 1)[0] for i in range(3)])
            om = np.linalg.norm((mo.R_world_imu(q[[0]])[0].inv() * mo.R_world_imu(q[[2]])[0]).as_rotvec()) / (W / 1e9)
            if np.linalg.norm(0.5 * (v1 + v2)) > 0.05 or np.linalg.norm(v2 - v1) > 0.02 or om > 0.3: continue
            sel = (t >= s) & (t <= s + W)
            if sel.sum() < 40: continue
            RT = mo.R_world_imu(np.array([s + int(W / 2)]))[0].as_matrix().T
            f_true = -RT @ (G0v + dg)
            Cm.append(a_csv[sel].mean(axis=0)); Cp.append(Gm @ (f_true + b_all[j])); Cp_nob.append(Gm @ f_true)
            Ci.append((10.0 / G0) * (RT @ np.array([0.0, G0, 0.0])))
        if not Cm: continue
        Cm, Cp, Cp_nob, Ci = map(np.array, (Cm, Cp, Cp_nob, Ci))
        e = lambda X: np.linalg.norm(Cm - X, axis=1)
        rows.append(dict(ctrl=ctrl, rec=name, n_win=len(Cm), meas_absC=np.linalg.norm(Cm, axis=1).mean(), model_absC=np.linalg.norm(Cp, axis=1).mean(),
                         model_nobias_absC=np.linalg.norm(Cp_nob, axis=1).mean(), ideal_absC=np.linalg.norm(Ci, axis=1).mean(),
                         err_model=e(Cp).mean(), err_model_nobias=e(Cp_nob).mean(), err_ideal_driver_only=e(Ci).mean(),
                         absC_L3=C.SCALE * np.linalg.norm(Cm, axis=1).mean()))
df = pd.DataFrame(rows); df.to_csv(C.HERE / "results" / "gravity_check_model_validation.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(df.to_string(index=False))
print("\nmean vector error vs measured quiet-window mean (m/s^2): full audit model %.3f | model with b=0 %.3f | driver-constant only %.3f  (n windows %d)" %
      (np.average(df.err_model, weights=df.n_win), np.average(df.err_model_nobias, weights=df.n_win), np.average(df.err_ideal_driver_only, weights=df.n_win), df.n_win.sum()))
