from typing import List
import numpy as np
from canvas.conformal_predictors.scores_new import ScoreFunction
from canvas.conformal_predictors.hindsight_scores import HindsightScoreFunction
from canvas.conformal_predictors.aci import DelayedACI



class CompetencyIndex:
    def __init__(self, prefix_len=0):

        self.prefix_len = prefix_len

        self._scores = {}
        self._names: List[str] = []

        self._conformal_predictors = {}

        self._indices = {}      # for storing the competency indices across time
        self._covered = {}

    def register(self, score_func: ScoreFunction, conformal_predictor: DelayedACI, name: str):

        self._scores[name] = score_func
        self._conformal_predictors[name] = conformal_predictor
        self._names.append(name)
        self._indices[name] = self.prefix_len * [.5]
        self._covered[name] = []

    def update(self, obs):
        for score in self._scores.values():
            score.update(obs)

    def save_snapshot(self, snapshot):
        for score in self._scores.values():
            score.save_snapshot(snapshot)

    def forward(self):
        res = {}
        for name in self._names:
            score = self._scores[name]
            cp = self._conformal_predictors[name]
            s = score()
            covered = cp.update(s)
            self._covered[name].append(covered)
            ub = cp.fit()
            idx = 1. / (1. + ub)
            res[name] = idx
            self._indices[name].append(idx)
        return res

    def pad(self, val=0.5):
        for i in self._indices.values():
            i.append(val)

    def get_history(self, name: str) -> np.ndarray:
        return np.array(self._indices[name])

    def get_average_values(self):
        return {name: np.mean(self._indices[name]) for name in self._names}

    def get_coverage_rate(self):
        return {name: np.mean(self._covered[name]) for name in self._names}


class HindsightCompetencyIndex:
    def __init__(self):
        self._scores = {}
        self._names: List[str] = []
        self._indices = {}      # for storing the competency indices across time

    def register(self, score_func: HindsightScoreFunction, name: str):
        self._scores[name] = score_func
        self._names.append(name)
        self._indices[name] = []

    def forward(self, snapshot):
        res = {}
        for name in self._names:
            score = self._scores[name]
            e = score(**snapshot)
            idx = 1. / (1. + e)
            res[name] = idx
            self._indices[name].append(idx)
        return res

    def get_history(self, name: str) -> np.ndarray:
        return np.array(self._indices[name])

    def get_average_values(self):
        return {name: np.mean(self._indices[name]) for name in self._names}