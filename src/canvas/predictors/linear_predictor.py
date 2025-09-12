# linear_predictor.py

from typing import Dict, Any
import numpy as np
from .wrapper_predictor import BasePredictors


class LinearPredictor(BasePredictors):
    def __init__(
        self,
        prediction_len: int = 12,
        history_len: int = 8,
        dt: float = 0.1,
        device: str = "cpu",
        *,
        smoothing_factor: float = 0.75,
    ):
        # Initialize common fields (history_len, prediction_len, dt, device)
        super().__init__(
            prediction_len=prediction_len,
            history_len=history_len,
            dt=dt,
            device=device,
        )

        self._history_len = self.history_len
        self._prediction_len = self.prediction_len

        assert self._prediction_len > 0, "prediction_len must be > 0"
        assert smoothing_factor >= 0.5, "smoothing_factor should be >= 0.5"
        self._smoothing_factor = float(smoothing_factor)

        # Precompute EMA filter (not currently used in the active path, but kept)
        self._filter = np.zeros((self._history_len, 1), dtype=float)
        self._filter[1:, 0] = np.flip(self._smoothing_factor ** np.arange(self._history_len - 1))
        self._filter[0, 0] = 1.0 - float(np.sum(self._filter[1:]))

    def __call__(self, tracking_result: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        """
        forecasting using accelerations (curvature via angular velocity over halves)
        tracking_result: Dict[obj_id, np.ndarray[T, 2]]
        returns: Dict[obj_id, np.ndarray[prediction_len, 2]]
        """
        dt = self._dt
        history_len = self._history_len
        prediction_len = self._prediction_len

        prediction_result: Dict[Any, np.ndarray] = {}

        for object_id, t in tracking_result.items():
            if not isinstance(t, np.ndarray):
                t = np.array(t)

            if t.shape[0] >= history_len + 2:
                pos0 = np.mean(t[-3:], axis=0)  # seed position (avg of last 3)
                v = (t[1:] - t[:-1]) / dt       # velocities

                vx, vy = v[-history_len:, 0], v[-history_len:, 1]

                half = history_len // 2
                vx1, vx2 = vx[:half], vx[half:2 * half]
                vy1, vy2 = vy[:half], vy[half:2 * half]

                vx1_mean = np.mean(vx1)
                vx2_mean = np.mean(vx2)
                vy1_mean = np.mean(vy1)
                vy2_mean = np.mean(vy2)

                vx_mean = np.mean(vx)
                vy_mean = np.mean(vy)

                v_mean = float(np.hypot(vx_mean, vy_mean))
                th1 = float(np.arctan2(vy1_mean, vx1_mean))
                th2 = float(np.arctan2(vy2_mean, vx2_mean))
                w_mean = (th2 - th1) / (max(1, half) * dt)  # angular vel
                th_mean = th2

                # Angular velocity propagation
                th_next = th_mean + dt * w_mean * np.arange(1, prediction_len + 1)

                # XY velocity and integrate to positions
                v_next = v_mean * np.vstack([np.cos(th_next), np.sin(th_next)]).T
                pos_next = pos0 + dt * np.cumsum(v_next, axis=0)

                prediction_result[object_id] = pos_next.astype(np.float32)

            else:
                # If too short, hold last position
                last = t[-1:, :]
                pos_next = np.repeat(last, prediction_len, axis=0)
                prediction_result[object_id] = pos_next.astype(np.float32)

        return prediction_result
