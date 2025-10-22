import numpy as np
from canvas.conformal_predictors.cp_utils import HistoryBuffer


class ScoreFunction:
    def __init__(self, prediction_len):
        self._prediction_len = prediction_len
        self._buffer = HistoryBuffer(history_length=prediction_len)
        pass

    def __call__(self, obs, *args, **kwargs):
        raise NotImplementedError


class ActionDivergenceScoreFunction(ScoreFunction):
    def __init__(self, prediction_len):
        super().__init__(prediction_len=prediction_len)
        self._past_controllers = []
        self._past_actions = []
        self._past_actions_base = []
        self._past_contexts = []
        self._past_predictions = []
        self._past_obs = []

    def __call__(self, obs, *args, **kwargs):
        # retrieve the context at t - N & use them for computing the baseline
        controller = self._past_controllers[-self._prediction_len]
        action = self._past_actions[-self._prediction_len]
        action_base = self._past_actions_base[-self._prediction_len]
        context = self._past_contexts[-self._prediction_len]
        prediction = self._past_predictions[-self._prediction_len]
        past_obs = self._past_obs[-self._prediction_len]

        # only data they are included in the prediction result
        ground_truth = self._buffer.query(keys=prediction.keys())

        action_gt, _ = controller(obs=past_obs, prediction_res=ground_truth, **context)

        action_gt = action_gt

        return np.sum((action - action_gt) ** 2) ** .5 / np.sum((action_base - action_gt) ** 2) ** .5

    def update(self, obs):
        o = obs['non-ego']
        o_t = {key: val[-1] for key, val in o.items()}
        self._buffer.update(o=o_t)
        print(self._buffer)

    def save_snapshot(self, obs, controller, action, action_base, prediction_res, context):
        self._past_controllers.append(controller)
        self._past_actions.append(action)
        self._past_actions_base.append(action_base)
        self._past_contexts.append(context)
        self._past_predictions.append(prediction_res)
        self._past_obs.append(obs)
