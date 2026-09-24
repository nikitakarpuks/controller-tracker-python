"""Controlled excitation study: CRB of b_a / b_g vs rotation amplitude (sinusoidal rotation about a tilted axis),
real strong-node cadence, vision 0.4deg/4mm, gravity known vs uncertain (1deg = 0.17 m/s^2)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from simlib import *
from simlib import _IMU
from crb import window_crb

class StubTruth:
    def __init__(self, amp_deg, f=0.3, axis=(0.3, 0.8, 0.5)):
        self.amp = np.radians(amp_deg); self.f = f; a = np.array(axis, float); self.ax = a / np.linalg.norm(a)
    def R(self, t):
        t = np.atleast_1d(t); return Rot.from_rotvec(self.ax[None] * (self.amp * np.sin(2 * np.pi * self.f * t))[:, None]).as_matrix()
    def w(self, t):
        t = np.atleast_1d(t); return self.ax[None] * (self.amp * 2 * np.pi * self.f * np.cos(2 * np.pi * self.f * t))[:, None]

class StubTruth3:
    """3-axis Lissajous rotation (axis diversity), rotvec(t) = amp*[sin(2 pi 0.30 t), sin(2 pi 0.41 t+1), sin(2 pi 0.23 t+2)]"""
    def __init__(self, amp_deg): self.amp = np.radians(amp_deg); self.f = np.array([0.30, 0.41, 0.23]); self.ph = np.array([0, 1.0, 2.0])
    def _rv(self, t): return self.amp * np.sin(2 * np.pi * self.f[None] * np.atleast_1d(t)[:, None] + self.ph[None])
    def R(self, t): return Rot.from_rotvec(self._rv(t)).as_matrix()
    def w(self, t):
        t = np.atleast_1d(t); h = 1e-4
        R0, R1 = self.R(t - h), self.R(t + h)
        return Rot.from_matrix(np.einsum("nji,njk->nik", R0, R1)).as_rotvec() / (2 * h)

root = REC_ROOT / RECORDINGS["static_dark"] / "mav0"
t_raw, _, _ = load_imu_csv(root / _IMU["right"] / "data.csv"); T0 = int(t_raw[0] + LAG_NS["right"])
tvs, strong = load_vision_rows("static_dark", "right"); tv = ((tvs[strong] - T0) / 1e9); tv = tv[(tv > 5) & (tv < 115)]
rows = []
for amp in (0, 5, 15, 30, 60, 120, 180):
    for W in (10, 30):
        for gp, gn in ((None, "gravity known"), (0.17, "gravity +-1deg")):
            res = window_crb(StubTruth(amp), tv, W, np.arange(6, 100 - W, 8.0), np.radians(0.4), 4e-3, grav_prior=gp)
            df = pd.DataFrame(res)
            rows.append(dict(rot_amp_deg=amp, W=W, gravity=gn, sd_ba=df.sd_ba.median(), sd_bg=df.sd_bg.median()))
for amp in (15, 30, 60, 90):
    for W in (10, 30):
        for gp, gn in ((None, "gravity known"), (0.17, "gravity +-1deg")):
            res = window_crb(StubTruth3(amp), tv, W, np.arange(6, 100 - W, 8.0), np.radians(0.4), 4e-3, grav_prior=gp)
            df = pd.DataFrame(res)
            rows.append(dict(rot_amp_deg=f"3axis-{amp}", W=W, gravity=gn, sd_ba=df.sd_ba.median(), sd_bg=df.sd_bg.median()))
r = pd.DataFrame(rows); r.to_csv(Path(__file__).parent / "exp1b_excitation.csv", index=False)
pd.set_option("display.width", 200)
r["rot_amp_deg"] = r.rot_amp_deg.astype(str)
print(r.pivot_table(index=["W", "gravity"], columns="rot_amp_deg", values="sd_ba").to_string(float_format=lambda v: f"{v:.2e}"))
print(r[r.gravity == "gravity known"].pivot_table(index="W", columns="rot_amp_deg", values="sd_bg").to_string(float_format=lambda v: f"{v:.2e}"))
