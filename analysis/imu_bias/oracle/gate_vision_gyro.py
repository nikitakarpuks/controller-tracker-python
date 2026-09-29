"""Sanity gate 2: gyro (zero extra bias) vs vision-derived relative rotation between consecutive strong vision frames
(README finding 9: ~0.34/0.51 deg median at the old lag). Inertial-frame vision orientation = R_world_headsetImu(t) (mocap) @ R_vision."""
import sys, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
rdir = rec_dir(name)
head = MocapOrientation(load_mocap_device(rdir, "headset"))
for ctrl in CTRLS:
    cfgf = CONFIG["fusion_heuristic"]; ov = cfgf.get("per_controller", {}).get(ctrl, {})
    si = float(ov.get("vision_weight_strong_inliers", cfgf.get("vision_weight_strong_inliers", 8)))
    se = float(ov.get("vision_weight_strong_error_px", cfgf.get("vision_weight_strong_error_px", 0.15)))
    ts, Rv = [], []
    with open(vision_csv(name)) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row[1] != ctrl or float(row[11]) < si or float(row[10]) > se: continue
            ts.append(int(row[0])); Rv.append(Rotation.from_quat([float(x) for x in row[2:6]]).as_matrix())
    ts = np.array(ts); o = np.argsort(ts); ts = ts[o]; Rv = np.array(Rv)[o]
    ok = head.valid(ts); ts = ts[ok]; Rv = Rv[ok]
    Rwh = head.R_world_imu(ts).as_matrix()
    Rw = Rwh @ Rv                       # inertial orientation of LED/body frame
    t, gb, ab = load_imu(rdir, ctrl)
    errs, mags, dts = [], [], []
    for i in range(len(ts) - 1):
        dt = (ts[i+1] - ts[i]) / 1e9
        if dt <= 0 or dt > 0.04: continue
        Rg = integrate_gyro_segment(t, gb, int(ts[i]), int(ts[i+1]))
        if Rg is None: continue
        Rvis = Rw[i].T @ Rw[i+1]
        errs.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rg.T @ Rvis).as_rotvec())))
        mags.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rvis).as_rotvec()))); dts.append(dt)
    errs = np.array(errs)
    print(f"{name}/{ctrl}: n={len(errs)} strong consecutive pairs; vision rotation/interval median {np.median(mags):.2f} deg; gyro-vs-vision median error {np.median(errs):.3f} deg (p90 {np.percentile(errs,90):.3f})")
