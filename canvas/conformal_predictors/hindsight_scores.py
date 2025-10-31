from typing import Tuple
import math
import numpy as np


class HindsightScoreFunction:
    def __init__(self, prediction_len, clip: float = math.inf):
        self._prediction_len = prediction_len
        self.clip = clip
        self.eps = 1e-6

    def compute(self, *args, **kwargs) -> Tuple[float, float]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> float:
        e, e_base = self.compute(*args, **kwargs)
        unclipped = (e + self.eps) / (e_base + self.eps)
        return min(unclipped, self.clip)


class HindsightPositionalDisplacementScoreFunction(HindsightScoreFunction):
    def __init__(self, prediction_len, step, clip: float = math.inf):
        super().__init__(prediction_len, clip)
        self._step = step

    def compute(self, *args, **kwargs):
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

        return d_max, d_max_base


class HindsightActionDivergenceScoreFunction(HindsightScoreFunction):
    def __init__(self, prediction_len, clip: float = math.inf):
        super().__init__(prediction_len=prediction_len, clip=clip)

    def compute(self, *args, **kwargs):
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
        ad = np.sum((action - action_gt) ** 2) ** .5
        ad_base = np.sum((action_base - action_gt) ** 2) ** .5
        return ad, ad_base


class HindsightPlanningRegretScoreFunction(HindsightScoreFunction):
    def __init__(self, prediction_len, clip: float = math.inf):
        super().__init__(prediction_len=prediction_len, clip=clip)

    def compute(self, *args, **kwargs):
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

        pr = cost - cost_gt
        pr_base = cost_base - cost_gt

        return pr, pr_base

