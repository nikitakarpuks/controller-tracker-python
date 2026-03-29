import numpy as np


class Transform:
    def __init__(self, R: np.ndarray, t: np.ndarray):
        self.R = R
        self.t = t

    def apply(self, pts):
        return (self.R @ pts.T).T + self.t

    def inverse(self):
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Transform(R_inv, t_inv)

    def compose(self, other: "Transform"):
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return Transform(R, t)
