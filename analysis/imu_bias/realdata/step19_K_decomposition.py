"""Step 19: physical reading of the fitted gyro K: symmetric part (axis scale/shear) and antisymmetric part (small rotation misalignment,
angle in degrees = |vector| x 180/pi). Whole-recording fits with the GYRO regressor, vision and mocap references."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from step3f_scale_misalign import fit_body
for name in ("static_dark", "walk_medium"):
    run = Run(name)
    for c in CTRLS:
        prep = prepare(run, c, ego="mocap"); sv, _ = make_steps(run, c, prep); sm = mocap_steps(run, c)
        for lab, st in (("vision", sv), ("mocap", sm)):
            beta, se, _ = fit_body(st, key="omega_g"); K = beta[3:].reshape(3, 3)
            A = 0.5 * (K - K.T); S = 0.5 * (K + K.T)
            rotvec = np.array([A[2, 1], A[0, 2], A[1, 0]])       # axial vector of the antisymmetric part
            print(f"[{name}/{c}/{lab:6s}] diag(K)% = {np.round(np.diag(K)*100,2)}   symmetric offdiag% = {np.round([S[0,1]*100,S[0,2]*100,S[1,2]*100],2)}   misalignment axis-angle = {np.round(np.degrees(rotvec),2)} deg (|.|={np.degrees(np.linalg.norm(rotvec)):.2f})   bias b = {np.round(beta[:3],4)} rad/s")
