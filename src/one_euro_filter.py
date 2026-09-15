"""One Euro Filter (Casiez, Godbout, Jalon, Lecolinet, CHI 2012) -- an
adaptive low-pass filter purpose-built for exactly this project's jitter
complaint: a tracked position/orientation that trembles with small, sharp
changes when nearly still, but must not lag during real fast motion. A fixed-
alpha smoother can't do both (heavy enough to kill tremble is also heavy
enough to lag); this filter's cutoff frequency adapts to the signal's own
recent speed -- low speed -> low cutoff -> heavy smoothing, high speed ->
high cutoff -> the filter gets out of the way. Two parameters:

    min_cutoff (Hz): the cutoff used when the signal isn't moving at all.
        LOWER = more smoothing/more lag at rest.
    beta: how fast the cutoff opens up as speed increases. HIGHER = less lag
        during fast motion, at the cost of passing more noise through then.

Reference / interactive demo: https://cristal.univ-lille.fr/~casiez/1euro/

Used by HeuristicPoseFusionFilter to filter the REPORTED/displayed pose only
-- never the internal (R, p) tracking state IMU dead-reckoning anchors off
next frame, so smoothing adds lag to what's shown, never to the tracking
math itself. See that module for how these are wired in.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * np.pi * cutoff * t_e
    return r / (r + 1.0)


def _exp_smooth(a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """Filters an (N,) vector signal. One shared adaptive cutoff per call,
    derived from the signal's own Euclidean speed -- not N independent
    per-axis cutoffs, which would let a straight-line motion visibly distort
    off-axis as each component gets a different amount of smoothing.

    First call after construction/reset() only initializes state and returns
    the input unchanged (no history yet to smooth against)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(self, t_s: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._t_prev is None:
            self._x_prev, self._dx_prev, self._t_prev = x.copy(), np.zeros_like(x), t_s
            return x.copy()
        t_e = t_s - self._t_prev
        if t_e <= 0:
            # Degenerate/duplicate timestamp -- nothing to smooth against a
            # zero-width step; hold the last output rather than divide by ~0.
            return self._x_prev.copy()

        dx = (x - self._x_prev) / t_e
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx_hat = _exp_smooth(a_d, dx, self._dx_prev)

        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exp_smooth(a, x, self._x_prev)

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t_s
        return x_hat


class OneEuroRotationFilter:
    """One Euro filtering for a rotation signal (3,3), via tangent-space
    deltas against a persistent smoothed reference frame -- avoids filtering
    quaternion components directly (which needs manual renormalization and
    double-cover sign handling); reuses the exact rotvec convention already
    used throughout the pose-fusion code (Rotation.as_rotvec/from_rotvec).

    The reference frame IS the filter's own last smoothed output (an EMA
    chains off its own prior output, not the raw input), so consecutive
    small deltas keep composing correctly frame to frame."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self._filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._R_ref = None

    def reset(self) -> None:
        self._filter.reset()
        self._R_ref = None

    def __call__(self, t_s: float, R: np.ndarray) -> np.ndarray:
        if self._R_ref is None:
            self._R_ref = R.copy()
            self._filter(t_s, np.zeros(3))  # seed the inner filter's time/derivative baseline
            return R.copy()
        delta = Rotation.from_matrix(self._R_ref.T @ R).as_rotvec()
        delta_smoothed = self._filter(t_s, delta)
        R_smoothed = self._R_ref @ Rotation.from_rotvec(delta_smoothed).as_matrix()
        self._R_ref = R_smoothed
        return R_smoothed
