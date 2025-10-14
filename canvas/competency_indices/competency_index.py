# competency_index.py
# CI mapping and ACI lower bound (Sec. IV, Eq.(5) and Eq.(8))
from __future__ import annotations
import numpy as np
from typing import Optional

Array = np.ndarray

class CompetencyIndex:
    def __init__(self, case: str = "traj", r_star: float = 0.5,
                 return_type: str = "final", clip: bool = True):
        self.case = case
        self.r_star = float(r_star)
        self.return_type = return_type
        self.clip = clip

    # ---- CI mapping: I = R* / (E + R*) ----
    def to_ci(self, err) -> float | Array:
        return self._to_ci(err, self.r_star)

    @staticmethod
    def _to_ci(err, r_star: float) -> float | Array:
        if np.isscalar(err):
            e = float(err)
            rs = float(r_star)
            val = rs / (e + rs) if np.isfinite(e) and rs > 0 else 0.0
            return float(np.clip(val, 0.0, 1.0))
        e = np.asarray(err, dtype=np.float64)
        rs = CompetencyIndex._broadcast_rstar(r_star, e.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = rs / (e + rs)
        ci[~np.isfinite(ci)] = 0.0
        return np.clip(ci, 0.0, 1.0)

    # ---- ACI lower bound: L = R* / (U + R*) ----
    @staticmethod
    def aci_lower_bound(U, r_star: float) -> float | Array:
        if np.isscalar(U):
            u = float(U)
            rs = float(r_star)
            val = rs / (u + rs) if np.isfinite(u) and rs > 0 else 0.0
            return float(np.clip(val, 0.0, 1.0))
        U = np.asarray(U, dtype=np.float64)
        rs = CompetencyIndex._broadcast_rstar(r_star, U.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = rs / (U + rs)
        lb[~np.isfinite(lb)] = 0.0
        return np.clip(lb, 0.0, 1.0)

    # ---- helper ----
    @staticmethod
    def _broadcast_rstar(r_star, shape):
        if np.isscalar(r_star):
            return np.full(shape, float(r_star), dtype=np.float64)
        rs = np.asarray(r_star, dtype=np.float64)
        if rs.shape == shape:
            return rs
        if rs.ndim == 1 and rs.size == shape[0]:
            out = np.tile(rs.reshape(-1, *([1] * (len(shape) - 1))), (1,) + shape[1:])
            return out.astype(np.float64)
        return np.full(shape, float(rs.ravel()[0]), dtype=np.float64)


