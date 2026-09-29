"""estimators.py -- causal gyro-bias estimators (3 params) fed by per-step rotation residuals.

Definitions (all rotations right-composed, body frame, this codebase's convention R_new = R_old @ R_rel):
  dR_v   : vision relative rotation between two consecutive frames, in the CONTROLLER BODY frame, with the
           headset ego-motion already removed:  dR_v = R_wc(i)^T R_wc(j),  R_wc = R_wh @ R_hc.
  dR_g(b): gyro-integrated relative rotation with bias b subtracted (integrate_gyro_segment(gyro - b)).
  e(b)   : Log(dR_g(b)^T dR_v)   -- body frame at the END frame j.  To first order e(b) = e0 + b*dt, where
           e0 = e(0) and dt = t_j - t_i (exact to ~|b dt||e0|/2 ~ 1e-6 rad; verified in test_estimators.py).
  Physical meaning: the gyro measures w_true + b_true, so dR_g(0) ~ dR_true Exp(+b_true dt) and
           e0 ~ -b_true * dt   =>   b_true ~ -e0 / dt.   (sign verified with exact synthetic tests.)

Every estimator exposes .update(step) -> current bias estimate AFTER absorbing the step, and .b (3,) rad/s.
A `step` is a dict: dt (s), e0 (3,), R_end (3,3 world orientation of the body at the END frame, used only by
the world-frame estimator), rate (rad/s, |body rate|, used only for gain scheduling). Estimators never look
at anything but the steps they were given => strictly causal by construction (test_estimators.py checks this).
"""
import numpy as np


class ZeroBias:
    name = "zero"

    def __init__(self):
        self.b = np.zeros(3)

    def update(self, s):
        return self.b


class ConstBias:
    name = "const"

    def __init__(self, b):
        self.b = np.asarray(b, dtype=float)

    def update(self, s):
        return self.b


class FeedbackEMA:
    """MENTOR-LITERAL scheme: predict with the current b, measure the disagreement, correct b, apply to the next
    frame.  measurement m = b - e(b)/dt  (= -e0/dt);  b <- (1-a) b + a m,  a = dt/tau  (gain per unit time 1/tau)
    or a = alpha (constant per-frame gain, `const_gain` mode -- NOT time-consistent when dt alternates 11/22 ms)."""

    def __init__(self, tau_s=10.0, gain_mode="dt", alpha=0.01, init=None):
        self.tau, self.mode, self.alpha = tau_s, gain_mode, alpha
        self.b = np.zeros(3) if init is None else np.asarray(init, float).copy()
        self.name = f"feedbackEMA(tau={tau_s}s,{gain_mode})" if gain_mode == "dt" else f"feedbackEMA(a={alpha})"

    def update(self, s):
        dt = s["dt"]
        e_b = s["e0"] + self.b * dt                    # residual seen by a predictor that already applies self.b
        a = min(1.0, dt / self.tau) if self.mode == "dt" else self.alpha
        m = self.b - e_b / dt
        self.b = (1 - a) * self.b + a * m
        return self.b


class RatioEMA:
    """dt-weighted ratio of exponentially-forgotten sums, BODY frame:  b = -sum(lam e0) / sum(lam dt)."""

    def __init__(self, tau_s=10.0, weight=None):
        self.tau, self.weight = tau_s, weight
        self.Se = np.zeros(3); self.Sd = 0.0; self.b = np.zeros(3)
        self.name = f"ratioEMA(tau={tau_s}s)"

    def update(self, s):
        lam = np.exp(-s["dt"] / self.tau)
        w = 1.0 if self.weight is None else self.weight(s)
        self.Se = lam * self.Se + w * s["e0"]
        self.Sd = lam * self.Sd + w * s["dt"]
        if self.Sd > 0:
            self.b = -self.Se / self.Sd
        return self.b


class WorldRLS:
    """Common-frame (world) accumulation:  the bias is constant in the BODY frame, so in the world frame each
    step contributes  -dt R_end b.  Accumulate A = sum lam dt R_end, S = sum lam (-R_end e0)  (S = A b + telescoped endpoint noise).
    Prior-regularised least squares (NOT the plain solve A b = S, which amplifies noise along near-singular directions of A --
    e.g. short tau on fast, large-angle motion):
        b = (A^T A + mu I)^-1 A^T S ,   mu = (sigma_s / sigma_b)^2
    sigma_s: endpoint-noise scale of S (rad, ~0.008 = vision/gyro endpoint noise), sigma_b: prior std of the bias (rad/s, ~0.02)."""

    def __init__(self, tau_s=10.0, sigma_b=0.02, sigma_s=0.008, weight=None):
        self.tau, self.mu, self.weight = tau_s, (sigma_s / sigma_b) ** 2, weight
        self.A = np.zeros((3, 3)); self.S = np.zeros(3); self.b = np.zeros(3)
        self.name = f"worldRLS(tau={tau_s}s)"

    def update(self, s):
        lam = np.exp(-s["dt"] / self.tau)
        w = 1.0 if self.weight is None else self.weight(s)
        Rj = s["R_end"]
        self.A = lam * self.A + w * s["dt"] * Rj
        self.S = lam * self.S + w * (-(Rj @ s["e0"]))
        self.b = np.linalg.solve(self.A.T @ self.A + self.mu * np.eye(3), self.A.T @ self.S)
        return self.b


class WindowMedian:
    """Windowed MEDIAN of per-step measurements m_k = -e0_k/dt_k over the last `window_s` seconds (robust to
    outliers but destroys the telescoping structure -- included as the literal 'windowed median' variant)."""

    def __init__(self, window_s=10.0):
        self.window = window_s
        self.buf = []      # (t, m)
        self.t = 0.0
        self.b = np.zeros(3)
        self.name = f"windowMedian({window_s}s)"

    def update(self, s):
        self.t += s["dt"]
        self.buf.append((self.t, -s["e0"] / s["dt"]))
        while self.buf and self.buf[0][0] < self.t - self.window:
            self.buf.pop(0)
        self.b = np.median(np.stack([m for _, m in self.buf]), axis=0)
        return self.b


def rate_weight(rate0=3.0):
    """Gain scheduling by excitation/dynamics: down-weight fast-rotation steps (timing/scale errors scale with the
    angular acceleration/rate) -- w = 1/(1+(rate/rate0)^2)."""
    return lambda s: 1.0 / (1.0 + (s["rate"] / rate0) ** 2)
