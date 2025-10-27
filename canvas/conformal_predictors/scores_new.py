import numpy as np
from canvas.conformal_predictors.cp_utils import HistoryBuffer


class ScoreFunction:
    def __init__(self, prediction_len):
        self._prediction_len = prediction_len
        self._buffer = HistoryBuffer(history_length=prediction_len)
        self._past_snapshot = {}
        self.EPS = 1e-6     # for preventing denominators from becoming 0
        pass

    def _set_keys(self, keys):
        self._past_snapshot = {key: [] for key in keys}

    def __call__(self, *args, **kwargs):
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


class PositionalDisplacementScoreFunction(ScoreFunction):
    def __init__(self, prediction_len, step):
        super().__init__(prediction_len=prediction_len)
        self._set_keys(keys=['prediction', 'prediction_base'])
        self._step = step

    def __call__(self, *args, **kwargs):
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

        return (d_max + self.EPS) / (d_max_base + self.EPS)


class ActionDivergenceScoreFunction(ScoreFunction):
    def __init__(self, prediction_len):
        super().__init__(prediction_len=prediction_len)
        self._set_keys(keys=['controller', 'action', 'action_base', 'context', 'prediction', 'obs'])

    def __call__(self, *args, **kwargs):
        # retrieve the context at t - N & use them for computing the baseline
        controller = self._past_snapshot['controller'][-self._prediction_len]
        action = self._past_snapshot['action'][-self._prediction_len]
        action_base = self._past_snapshot['action_base'][-self._prediction_len]
        context = self._past_snapshot['context'][-self._prediction_len]
        prediction = self._past_snapshot['prediction'][-self._prediction_len]
        past_obs = self._past_snapshot['obs'][-self._prediction_len]

        # only data they are included in the prediction result
        ground_truth = self._buffer.query(keys=prediction.keys())
        action_gt, _ = controller(obs=past_obs, prediction_res=ground_truth, change_controller_state=False, **context)
        action_gt = action_gt

        return np.sum((action - action_gt) ** 2 + self.EPS) ** .5 / np.sum((action_base - action_gt) ** 2 + self.EPS) ** .5


class PlanningRegretScoreFunction(ScoreFunction):
    def __init__(self, prediction_len):
        super().__init__(prediction_len=prediction_len)
        self._set_keys(keys=['controller', 'U', 'U_base', 'context', 'prediction', 'obs'])

    def __call__(self, *args, **kwargs):
        # retrieve the context at t - N & use them for computing the baseline
        controller = self._past_snapshot['controller'][-self._prediction_len]
        U = self._past_snapshot['U'][-self._prediction_len]
        U_base = self._past_snapshot['U_base'][-self._prediction_len]
        context = self._past_snapshot['context'][-self._prediction_len]
        prediction = self._past_snapshot['prediction'][-self._prediction_len]
        past_obs = self._past_snapshot['obs'][-self._prediction_len]

        # only data they are included in the prediction result
        ground_truth = self._buffer.query(keys=prediction.keys())
        action_gt, info_gt = controller(obs=past_obs, prediction_res=ground_truth, change_controller_state=False, **context)
        cost_gt = info_gt['cost_to_go'].item()
        cost = controller.cost_to_go(obs=past_obs, prediction_res=ground_truth, U=U).item()
        cost_base = controller.cost_to_go(obs=past_obs, prediction_res=ground_truth, U=U_base).item()

        return (cost - cost_gt + self.EPS) / (cost_base - cost_gt + self.EPS)

