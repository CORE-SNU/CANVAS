import math
from typing import Tuple
import numpy as np
from canvas.conformal_predictors.cp_utils import HistoryBuffer


class ScoreFunction:
    def __init__(self, prediction_len, clip: float = math.inf):
        assert clip > 0.
        self._prediction_len = prediction_len
        self._buffer = HistoryBuffer(history_length=prediction_len)
        self._past_snapshot = {}
        self.eps = 1e-6     # for preventing denominators from becoming 0
        self.clip = clip    # for preventing the error ratio from growing unbounded
        pass

    def _set_keys(self, keys):
        self._past_snapshot = {key: [] for key in keys}

    def __call__(self, *args, **kwargs) -> float:
        """
        post-processing of the computed errors to obtain the final rate
        """
        e, e_base = self.compute(*args, **kwargs)
        unclipped = (e + self.eps) / (e_base + self.eps)
        return min(unclipped, self.clip)

    def compute(self, *args, **kwargs) -> Tuple[float, float]:
        """
        compute the errors of a given model & baseline
        return (E^A_t, E^B_t)
        """
        raise NotImplementedError

    def save_snapshot(self, snapshot):
        for key, val in snapshot.items():
            if key in self._past_snapshot:
                self._past_snapshot[key].append(val)

        for key in self._past_snapshot:
            if key not in snapshot:
                raise KeyError(key)

    def update(self, obs):
        o = obs['non-ego']
        o_t = {key: val[-1] for key, val in o.items()}
        self._buffer.update(o=o_t)

    @property
    def delay(self):
        return self._prediction_len


class PositionalDisplacementScoreFunction(ScoreFunction):
    def __init__(self, prediction_len, step, clip: float = math.inf):
        super().__init__(prediction_len=prediction_len, clip=clip)
        self._set_keys(keys=['prediction', 'prediction_base'])
        self._step = step

    def compute(self, *args, **kwargs):
        # TODO: compared to other indices, this can exploit more recent observations, i.e., delay is smaller
        # retrieve the context at t - N & use them for computing the baseline

        prediction = self._past_snapshot['prediction'][-self._prediction_len]
        prediction_base = self._past_snapshot['prediction_base'][-self._prediction_len]
        # only data they are included in the prediction result
        ground_truth = self._buffer.query(keys=prediction.keys())

        i = self._step
        d_max = 0.
        for key, p in prediction.items():
            if key in ground_truth:
                g = ground_truth[key]
                d = np.sum((p[i] - g[i]) ** 2) ** .5 if g.shape[0] > i else 0.
                d_max = max(d, d_max)

        d_max_base = 0.
        for key, p in prediction_base.items():
            if key in ground_truth:
                g = ground_truth[key]
                d = np.sum((p[i] - g[i]) ** 2) ** .5 if g.shape[0] > i else 0.
                d_max_base = max(d, d_max_base)

        return d_max, d_max_base


class ActionDivergenceScoreFunction(ScoreFunction):
    def __init__(self, prediction_len, clip: float = math.inf):
        super().__init__(prediction_len=prediction_len, clip=clip)
        self._set_keys(keys=['controller', 'action', 'action_base', 'context', 'prediction', 'obs'])

    def compute(self, *args, **kwargs):
        # alias
        past = self._past_snapshot
        pl = self._prediction_len
        # retrieve the context at t - N & use them for computing the baseline
        controller = past['controller'][-pl]
        action = past['action'][-pl]
        action_base = past['action_base'][-pl]
        context = past['context'][-pl]
        prediction = past['prediction'][-pl]
        past_obs = past['obs'][-pl]

        # only data they are included in the prediction result
        ground_truth = self._buffer.query(keys=prediction.keys())
        action_gt, _ = controller(obs=past_obs, prediction_res=ground_truth, change_controller_state=False, **context)
        action_gt = action_gt

        ad = np.sum((action - action_gt) ** 2) ** .5
        ad_base = np.sum((action_base - action_gt) ** 2) ** .5
        return ad, ad_base


class PlanningRegretScoreFunction(ScoreFunction):
    def __init__(self, prediction_len, clip: float = math.inf):
        super().__init__(prediction_len=prediction_len, clip=clip)
        self._set_keys(keys=['controller', 'U', 'U_base', 'context', 'prediction', 'obs'])

    def compute(self, *args, **kwargs):

        past = self._past_snapshot
        pl = self._prediction_len

        # retrieve the context at t - N & use them for computing the baseline
        controller = past['controller'][-pl]
        U = past['U'][-pl]
        U_base = past['U_base'][-pl]
        context = past['context'][-pl]
        prediction = past['prediction'][-pl]
        past_obs = past['obs'][-pl]

        # only data they are included in the prediction result
        ground_truth = self._buffer.query(keys=prediction.keys())
        action_gt, info_gt = controller(obs=past_obs, prediction_res=ground_truth, change_controller_state=False, warm_start=U, **context)
        action_gt_base, info_gt_base = controller(obs=past_obs, prediction_res=ground_truth, change_controller_state=False, warm_start=U_base,     **context)
        cost_gt = info_gt['cost_to_go'].item()
        cost_gt_base = info_gt_base['cost_to_go'].item()
        cost = controller.cost_to_go(obs=past_obs, prediction_res=ground_truth, U=U).item()
        cost_base = controller.cost_to_go(obs=past_obs, prediction_res=ground_truth, U=U_base).item()

        pr = cost - cost_gt_base
        pr_base = cost_base - cost_gt_base

        return pr, pr_base
