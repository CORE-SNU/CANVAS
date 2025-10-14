# score_function.py
# canvas/competency_indices/score_function.py
from __future__ import annotations
from typing import Dict, Optional, Sequence, Any
import numpy as np

class ScoreFunction:
    """
    s_{p,pi}(x, y) used to build per-frame energy E_t.

    Modes:
      - "pos"    : Position-error based (Eq. (2): worst agent at each step)
      - "action" : Controller-aware action difference (Eq. (3))
      - "regret" : Objective regret (Eq. (4))

    Aggregation across horizon:
      'max' | 'sum' | 'mean', optional step-wise weights.
    """

    def __init__(self, mode: str = "pos", horizon_agg: str = "max",
                 step_weights: Optional[Sequence[float]] = None):
        if mode not in {"pos", "action", "regret"}:
            raise ValueError(f"Unknown mode: {mode}")
        if horizon_agg not in {"max", "sum", "mean"}:
            raise ValueError(f"Unknown horizon_agg: {horizon_agg}")
        self.mode = mode
        self.horizon_agg = horizon_agg
        self.step_weights = step_weights

    def _agg(self, vals: Sequence[float]) -> float:
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return 0.0
        if self.step_weights is not None:
            w = np.asarray(self.step_weights[: arr.shape[0]], dtype=float)
            arr = arr * w
        if self.horizon_agg == "max":
            return float(np.max(arr))
        if self.horizon_agg == "mean":
            return float(np.mean(arr))
        return float(np.sum(arr))

    def __call__(self, x: Any,
                 y_future: Dict[int, np.ndarray],
                 yhat_future: Dict[int, np.ndarray],
                 controller: Optional[Any] = None) -> float:
        if self.mode == "pos":
            return self._score_pos(y_future, yhat_future)
        if self.mode == "action":
            if controller is None:
                raise ValueError("controller is required for mode='action'")
            return self._score_action(x, y_future, yhat_future, controller)
        if self.mode == "regret":
            if controller is None:
                raise ValueError("controller is required for mode='regret'")
            return self._score_regret(x, y_future, yhat_future, controller)
        raise RuntimeError("Invalid mode")

    def _score_pos(self, y_future: Dict[int, np.ndarray],
                   yhat_future: Dict[int, np.ndarray]) -> float:
        common = list(set(y_future) & set(yhat_future))
        if not common:
            return 0.0
        H = min(min(y_future[p].shape[0] for p in common),
                min(yhat_future[p].shape[0] for p in common))
        if H <= 0:
            return 0.0
        per_step_max = []
        for i in range(H):
            errs_i = []
            for pid in common:
                gt_i = np.asarray(y_future[pid][i], dtype=float)
                ph_i = np.asarray(yhat_future[pid][i], dtype=float)
                errs_i.append(float(np.linalg.norm(gt_i - ph_i)))
            per_step_max.append(max(errs_i) if errs_i else 0.0)
        return self._agg(per_step_max)

    def _score_action(self, x: Any,
                      y_future: Dict[int, np.ndarray],
                      yhat_future: Dict[int, np.ndarray],
                      controller: Any) -> float:
        u_gt = np.asarray(controller(x, predictions=y_future), dtype=float)
        u_ph = np.asarray(controller(x, predictions=yhat_future), dtype=float)
        if u_gt.shape != u_ph.shape:
            raise ValueError(f"Action shapes differ: {u_gt.shape} vs {u_ph.shape}")
        return float(np.linalg.norm(u_gt - u_ph))

    def _score_regret(self, x: Any,
                      y_future: Dict[int, np.ndarray],
                      yhat_future: Dict[int, np.ndarray],
                      controller: Any) -> float:
        u_star, J_star = controller.solve_with_cost(x, y_future)
        u_hat,  J_hat  = controller.solve_with_cost(x, yhat_future)
        regret = float(J_hat) - float(J_star)
        return float(regret if regret > 0.0 else 0.0)


