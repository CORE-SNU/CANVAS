import numpy as np


def fx(x, u, dt):
    return x + dt * np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])






class ExpectationMaximizationPredictor:
    def __init__(self, prediction_len, history_len, smoothing_factor, dt):

        self._history_len = history_len
        self._prediction_len = prediction_len

        assert self._prediction_len > 0

        # smoothing factor for computing the EMAs of the trajectories
        assert smoothing_factor >= .5
        self._smoothing_factor = smoothing_factor

        self._filter = np.zeros((history_len, 1))
        self._filter[1:, 0] = np.flip(smoothing_factor ** np.arange(history_len-1))
        self._filter[0, 0] = 1. - np.sum(self._filter[1:])
        self._dt = dt

        pass

    def __call__(self, tracking_result):
        """
        h: history of shape (# batch, history length * state dim.)
        h = (h[t-N+1], ..., h[t])

        forecasting using accelerations
        """

        dt = self._dt
        history_len = self._history_len
        prediction_len = self._prediction_len
        filter = self._filter

        prediction_result = {}

        for object_id, t in tracking_result.items():
            t = np.array(t)
            if t.shape[0] >= history_len + 2:
                # TODO: EKF or UT (or Invariant KF)
                x, y = t[:, 0], t[:, 1]

                x_filtered = np.convolve(x, filter, mode='valid')
                y_filtered = np.convolve(y, filter, model='valid')

                pos0 = np.array([x_filtered[-1], y_filtered[-1]])

                vx = x_filtered[1:] - x_filtered[:-1]
                vy = y_filtered[1:] - y_filtered[:-1]

                v = (vx ** 2 + vy ** 2) ** .5

                v_mean = np.mean(v)

                th = np.arctan2(vy, vx)

                w = th[1:] - th[:-1]
                w_mean = np.mean(w)

                # EMA of the position
                pos0 = np.sum(t[-history_len:] * filter, axis=0)       # p[k]
                pos1 = np.sum(t[-history_len-1:-1] * filter, axis=0)   # p[k-1]
                pos2 = np.sum(t[-history_len-2:-2] * filter, axis=0)   # p[k-2]

                v1 = pos0 - pos1                        # v[k-1]: linear velocity
                v2 = pos1 - pos2                        # v[k-2]

                th1 = np.arctan2(v1[1], v1[0])          # th[k-1]
                th2 = np.arctan2(v2[1], v2[0])          # th[k-2]

                w = th2 - th1                           # w[k-2]: angular velocity

                # angular velocity propagation
                th0 = th1 + w

                th_next = th0 + dt * w * np.arange(1, prediction_len+1)

                # xy-velocity
                v_next = .5 * (v1 + v2) * np.vstack([np.cos(th_next), np.sin(th_next)]).T
                pos_next = pos0 + dt * np.cumsum(v_next, axis=0)

                prediction_result[object_id] = pos_next

            else:
                # Don't generate prediction for the trajectories that are too short
                prediction_result[object_id] = None
                pass

        return prediction_result