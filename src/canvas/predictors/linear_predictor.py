import numpy as np


class LinearPredictor:
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

        prediction_result = {}

        for object_id, t in tracking_result.items():
            if not isinstance(t, np.ndarray):
                t = np.array(t)

            if t.shape[0] >= history_len + 2:
                pos0 = np.mean(t[-3:], axis=0)
                v = (t[1:] - t[:-1]) / dt

                vx, vy = v[-history_len:, 0], v[-history_len:, 1]

                half = history_len // 2
                vx1, vx2 = vx[:half], vx[half:2*half]
                vy1, vy2 = vy[:half], vy[half:2*half]

                vx1_mean = np.mean(vx1)
                vx2_mean = np.mean(vx2)

                vy1_mean = np.mean(vy1)
                vy2_mean = np.mean(vy2)

                vx_mean = np.mean(vx)
                vy_mean = np.mean(vy)

                v_mean = (vx_mean ** 2 + vy_mean ** 2) ** .5
                th1 = np.arctan2(vy1_mean, vx1_mean)
                th2 = np.arctan2(vy2_mean, vx2_mean)
                w_mean = (th2 - th1) / (half * dt)
                th_mean = th2

                # angular velocity propagation

                th_next = th_mean + dt * w_mean * np.arange(1, prediction_len + 1)

                # xy-velocity
                v_next = v_mean * np.vstack([np.cos(th_next), np.sin(th_next)]).T
                pos_next = pos0 + dt * np.cumsum(v_next, axis=0)

                prediction_result[object_id] = pos_next

                # TODO: EKF or UT (or Invariant KF)
                '''
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

                
                '''

            else:
                # Don't generate prediction for the trajectories that are too short
                prediction_result[object_id] = b = np.repeat(t[-1:, :], prediction_len, axis=0)
                pass
        return prediction_result