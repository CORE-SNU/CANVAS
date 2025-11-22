from typing import List
import numpy as np
from canvas.conformal_predictors.scores import ScoreFunction
from canvas.conformal_predictors.hindsight_scores import HindsightScoreFunction
from canvas.conformal_predictors.aci import DelayedACI


def get_average(x: list, window: int):
    assert x        # non-empty list
    n_items = len(x)
    n = min(n_items, window)
    return np.mean(x[-n:])


class MovingAverageCompetencyIndex:
    def __init__(self, prefix_len=0, window=1):

        self.prefix_len = prefix_len
        self._window = window
        self._score_functions = {}
        self._names: List[str] = []

        self._indices = {}      # for storing the competency indices across time
        self._scores = {}
        self._averages = {}
        self._step = 0

    def register(self, score_func: ScoreFunction, name: str):

        self._score_functions[name] = score_func
        self._names.append(name)
        self._indices[name] = self.prefix_len * [.5]
        self._scores[name] = []
        self._averages[name] = None

    def update(self, obs):
        for score in self._score_functions.values():
            score.update(obs)

    def save_snapshot(self, snapshot):
        for score in self._score_functions.values():
            score.save_snapshot(snapshot)

    def forward(self):
        res = {}
        for name in self._names:
            score = self._score_functions[name]

            s = score()
            self._scores[name].append(s)

            if self._averages[name] is None:
                val = s
            else:
                val = get_average(x=self._scores[name], window=self._window)
            self._averages[name] = val
            idx = 1. / (1. + val)
            res[name] = idx
            self._indices[name].append(idx)
        self._step += 1
        return res

    def pad(self, val=0.5):
        for i in self._indices.values():
            i.append(val)
        self._step += 1

    def get_history(self, name: str) -> np.ndarray:
        return np.array(self._indices[name])

    def get_average_values(self):
        return {name: np.mean(self._indices[name]) for name in self._names}


class ConformalizedCompetencyIndex:
    def __init__(self, prefix_len=0, momentum=0.5):

        self.prefix_len = prefix_len
        self._momentum = momentum
        self._score_functions = {}
        self._names: List[str] = []

        self._conformal_predictors = {}

        self._indices = {}      # for storing the competency indices across time
        self._covered = {}
        self._scores = {}
        self._averages = {}
        self._step = 0

    def register(self, score_func: ScoreFunction, conformal_predictor, name: str):

        self._score_functions[name] = score_func
        self._conformal_predictors[name] = conformal_predictor
        self._names.append(name)
        self._indices[name] = self.prefix_len * [.5]
        self._covered[name] = []
        self._scores[name] = []
        self._averages[name] = None

    def update(self, obs):
        for score in self._score_functions.values():
            score.update(obs)

    def save_snapshot(self, snapshot):
        for score in self._score_functions.values():
            score.save_snapshot(snapshot)

    def forward(self):
        res = {}
        for name in self._names:
            score = self._score_functions[name]
            cp = self._conformal_predictors[name]
            s = score()
            self._scores[name].append(s)

            if self._averages[name] is None:
                self._averages[name] = s
            else:
                self._averages[name] = self._momentum * self._averages[name] + (1. - self._momentum) * s

            covered = cp.update(self._averages[name])
            if covered is not None:
                self._covered[name].append(covered)
            ub = cp.fit()
            idx = 1. / (1. + ub)
            res[name] = idx
            self._indices[name].append(idx)
        self._step += 1
        return res

    def pad(self, val=0.5):
        for i in self._indices.values():
            i.append(val)
        self._step += 1
    def get_history(self, name: str) -> np.ndarray:
        return np.array(self._indices[name])

    def get_average_values(self):
        return {name: np.mean(self._indices[name]) for name in self._names}

    def get_coverage_rate(self):
        return {name: np.mean(self._covered[name]) for name in self._names}


class HindsightCompetencyIndex:
    def __init__(self, momentum=0.3):
        self._momentum = momentum
        self._score_functions = {}
        self._names: List[str] = []

        self._scores = {}
        self._indices = {}      # for storing the competency indices across time
        self._averages = {}

    def register(self, score_func: HindsightScoreFunction, name: str):
        self._score_functions[name] = score_func
        self._names.append(name)
        self._scores[name] = []
        self._indices[name] = []
        self._averages[name] = None

    def forward(self, snapshot):
        res = {}
        for name in self._names:
            score = self._score_functions[name]
            s = score(**snapshot)

            self._scores[name].append(s)

            # exponential moving average

            if self._averages[name] is None:
                self._averages[name] = s
            else:
                self._averages[name] = self._momentum * self._averages[name] + (1. - self._momentum) * s

            # e = np.mean(self._scores[name][-self._window:])

            idx = 1. / (1. + self._averages[name])
            res[name] = idx
            self._indices[name].append(idx)
        return res

    def get_history(self, name: str) -> np.ndarray:
        return np.array(self._indices[name])

    def get_score_history(self, name: str) -> np.ndarray:
        return np.array(self._scores[name])

    def get_average_values(self):
        return {name: np.mean(self._indices[name]) for name in self._names}

    def get_max_scores(self):
        return {name: np.max(self._scores[name]) for name in self._names}