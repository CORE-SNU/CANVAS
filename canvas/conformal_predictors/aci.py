import numpy as np



class DelayedACI:
    """
    Implementation of Adaptive Conformal Prediction (ACP) for pedestrian trajectory forecasting.
    For the details of ACP, refer to Gibbs & Candes, 2021, or Zaffran et al., 2022.
    Considering the lagged nature of prediction tasks, a lagged variant of ACP is implemented here; See Dixit et al., 2023.
    """

    def __init__(self,
                 target_miscoverage_level,
                 step_size,
                 delay,
                 max_score,
                 sample_size
                 ):
        self._alpha = target_miscoverage_level
        self._delay = delay

        # effective miscoverage level alpha_t
        # initialization: alpha_0 := alpha
        self._alpha_t = self._alpha

        self._past_scores = []
        self._past_quantiles = delay * [max_score]

        # step size parameter
        # gamma = 0 corresponds to the standard split conformal prediction
        self._lr = step_size
        self._max_score = max_score

        self._sample_size = sample_size

        self._step = 0

    def update(self, score):
        self._past_scores.append(score)
        err: bool = score <= self._past_quantiles[-self._delay]
        self._alpha_t += self._lr * (self._alpha - err)

    def fit(self):
        assert len(self._past_scores) > 0
        sz = min(self._sample_size, len(self._past_scores))
        if self._alpha_t <= 0.:
            q = self._max_score
        elif self._alpha_t >= 1.:
            q = 0.
        else:
            q = np.quantile(self._past_scores[-sz:], q=self._alpha_t)
        self._past_quantiles.append(q)
        return q
