import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trust_sweep as ts
from src.imu_data import integrate_gyro_segment, _DIAG_FLIP as D
from src.transformations import Transform
from src.mocap_data import load_mocap_bridge

rec, ctrl = "euroc_recording_20260826173103_static_dark", "right_controller"
V = ts.VARIANTS["old"]
df = pd.read_csv(Path(__file__).resolve().parent / "out" / "old" / "imu_trust_all.csv")
df = df[(df.recording == rec) & (df.ctrl_name == ctrl)]
rec_dir = ts.RECORDINGS_ROOT / rec
cfg = ts.load_json_config(ts.CONFIG["controllers"][ctrl]["config_path"])
t_imu, gyro, accel = ts.load_and_calibrate_controller_imu(rec_dir/"mav0"/ts._IMU_REL_PATH[ctrl], cfg, lag_ns=ts._OLD_LAG_NS[ctrl], factory_corrected_input=False, accel_scale=1.0)
cm = ts.load_device_mocap(rec_dir, ctrl)
bridge = load_mocap_bridge(str(ts.REPO / ts.CONFIG["controllers"][ctrl]["mocap_bridge_path"].lstrip("./")))
print("bridge R (IMU->LED) =\n", np.round(bridge.R, 3), "\nangle(bridge.R, D) deg =", round(ts.rotation_angle_deg(bridge.R.T @ D), 2))
def Wimu(t):
    m = cm.pose_at(t); return Transform(*m).compose(cm.T_imu_marker.inverse())
out = []
for dt in (0.035, 0.1, 0.3, 1.0):
    s = df[np.isclose(df.dt_s, dt)]
    s = s.iloc[::max(1, len(s)//150)]
    e_imu, e_led, e_rawgyro = [], [], []
    for a in s.anchor_ts_ns:
        t0, t1 = int(a), int(a) + int(round(dt * 1e9))
        R_rel = integrate_gyro_segment(t_imu, gyro, t0, t1)             # gyro_body (D-flipped) integrated
        Ri = Wimu(t0).R.T @ Wimu(t1).R                                  # true relative rotation, mocap IMU (sensor) frame
        e_imu.append(ts.rotation_angle_deg(R_rel.T @ Ri))               # what the sweep effectively scores
        e_led.append(ts.rotation_angle_deg(R_rel.T @ (D @ Ri @ D)))     # truth expressed in the body/LED frame the gyro is in
    print(f"dt={dt:5.3f} n={len(e_imu)}  median err sweep-frame(IMU) {np.median(e_imu):7.2f} deg | body/LED frame {np.median(e_led):6.2f} deg")
