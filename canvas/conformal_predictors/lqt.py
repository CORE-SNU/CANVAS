import numpy as np
from typing import List


class LinearQuantileTracker:
    """
    Implementation of a linear quantile tracker updated via the online PGD introduced in
    Areces, F. et al. Online Conformal Prediction via Online Optimization. ICML. 2025.
    To consider delayed feedbacks, we implement a delayed version of the online GD following
    Quanrud, K. & Khashabi, D. Online learning with adversarial delays. NIPS. 2015.
    """

    def __init__(
            self,
            target_miscoverage_level,
            step_size,
            delay,
            sample_size
    ):
        self._alpha = target_miscoverage_level
        self._delay = delay

        # effective miscoverage level alpha_t
        # initialization: alpha_0 := alpha
        self._alpha_t = self._alpha

        self._past_scores = []
        self._past_quantiles = []

        # step size of the GD
        self._lr = step_size
        self._sample_size = sample_size

        self._step = 0

        # parameter vector theta in R^{p+1}
        self.params: np.ndarray = self._initialize_params()
        self._past_params: List[np.ndarray] = []

    def _initialize_params(self) -> np.ndarray:
        p = np.ones(self._sample_size + 1) / self._sample_size
        p[-1] = 0.
        return p

    def feature_vector(self):
        # feature vector [S_{t-N-p+1}, ..., S_{t-N}, 1]^T
        # If t < N + p - 1, then pad the vector with the oldest value.
        n = len(self._past_scores)
        assert n > 0
        s0 = self._past_scores[0]
        phi = max(self._sample_size - n, 0) * [s0] + self._past_scores[-self._sample_size:] + [1.]
        return np.array(phi)

    def update(self, score):
        # Given the latest score S_{t-N} (delayed N steps), perform online PGD to update param of the tracker
        self._past_scores.append(score)

        # true if the interval fails to cover the score
        # S_{t-N} <= q(t-N)?
        if len(self._past_quantiles) > self._delay:
            # online PGD: theta_{t-1} -> theta_t
            # sample gradient
            err: bool = score > self._past_quantiles[-self._delay]

            g = (self._alpha - err) * self.feature_vector()
            # GD step
            # step size satisfying Robbins-Monro conditions
            # decay = 1. / (1. + self._step) ** .51
            self.params -= self._lr * g
            # projection onto [0, 1]^{p+1}
            self.params = np.clip(self.params, 0., 1.)

            covered = 1. - err
        else:
            covered = None

        q = self.fit()  # compute q(t)
        self._past_quantiles.append(q)

        self._step += 1

        return covered

    def fit(self):
        # q(t) = theta_t^T phi_t
        return self.feature_vector() @ self.params
