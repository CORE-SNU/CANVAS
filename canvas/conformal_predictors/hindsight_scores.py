import numpy as np


class HindsightScoreFunction:
    def __init__(self, prediction_len):
        self._prediction_len = prediction_len
        self.EPS = 1e-6

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class HindsightPositionalDisplacementScoreFunction(HindsightScoreFunction):
    def __init__(self, prediction_len, step):
        super().__init__(prediction_len)
        self._step = step

    def __call__(self, *args, **kwargs):
        future = kwargs['future']
        prediction = kwargs['prediction']
        prediction_base = kwargs['prediction_base']

        i = self._step
        d_max = 0.
        for key, p in prediction.items():
            if key in future:
                g = future[key]
                d = np.sum((p[i] - g[i]) ** 2) ** .5 if g.shape[0] > i else 0.
                d_max = max(d, d_max)

        d_max_base = 0.
        for key, p in prediction_base.items():
            if key in future:
                g = future[key]
                d = np.sum((p[i] - g[i]) ** 2) ** .5 if g.shape[0] > i else 0.
                d_max_base = max(d, d_max_base)

        return (d_max + self.EPS) / (d_max_base + self.EPS)


class HindsightActionDivergenceScoreFunction(HindsightScoreFunction):
    def __init__(self, prediction_len):
        super().__init__(prediction_len=prediction_len)

    def __call__(self, *args, **kwargs):
        # retrieve the context at t - N & use them for computing the baseline
        controller = kwargs['controller']
        action = kwargs['action']
        action_base = kwargs['action_base']
        context = kwargs['context']
        obs = kwargs['obs']

        # only data they are included in the prediction result
        future = kwargs['future']
        action_gt, _ = controller(obs=obs, prediction_res=future, change_controller_state=False, **context)
        action_gt = action_gt

        return np.sum((action - action_gt) ** 2 + self.EPS) ** .5 / np.sum((action_base - action_gt) ** 2 + self.EPS) ** .5


class HindsightPlanningRegretScoreFunction(HindsightScoreFunction):
    def __init__(self, prediction_len):
        super().__init__(prediction_len=prediction_len)

    def __call__(self, *args, **kwargs):
        # retrieve the context at t - N & use them for computing the baseline
        controller = kwargs['controller']
        U = kwargs['U']
        U_base = kwargs['U_base']
        context = kwargs['context']
        obs = kwargs['obs']

        # only data they are included in the prediction result
        future = kwargs['future']
        action_gt, info_gt = controller(obs=obs, prediction_res=future, change_controller_state=False, **context)
        cost_gt = info_gt['cost_to_go'].item()
        cost = controller.cost_to_go(obs=obs, prediction_res=future, U=U).item()
        cost_base = controller.cost_to_go(obs=obs, prediction_res=future, U=U_base).item()

        return (cost - cost_gt + self.EPS) / (cost_base - cost_gt + self.EPS)

