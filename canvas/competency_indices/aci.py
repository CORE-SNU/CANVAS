# canvas/competency_indices/aci.py
from __future__ import annotations
import math

class AdaptiveEnergyCI:
    """
    Online estimator for an upper energy bound U_t via Robbins–Monro quantile tracking.
    Tracks the (1 - alpha)-quantile of the energy stream E_t.

    Update rule:
      U <- U + eta * (q - I{E <= U}),  where q = 1 - alpha
    """

    def __init__(self, alpha: float = 0.1, step_size: float = 0.05, init_U: float = 0.0):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        self.q = 1.0 - float(alpha)
        self.eta = float(step_size)
        self.U = float(init_U)

    def update(self, score: float) -> float:
        # Indicator whether current score lies below the bound
        indicator = 1.0 if score <= self.U else 0.0
        self.U = max(1e-12, self.U + self.eta * (self.q - indicator))
        # Simple “cold start”: on first calls when U~0, pull it up quickly
        if self.U < score and self.U < 1e-6:
            self.U = score
        return self.U
