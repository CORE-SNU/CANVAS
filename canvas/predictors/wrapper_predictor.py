import os
import numpy as np
import torch
from abc import ABC, abstractmethod

class BasePredictors(ABC):
    def __init__(self, 
                 prediction_len=12,history_len=8,
                 dt=0.1,
                 device='cpu'):
        """Predictor wrapper for different predictors and for simplicity of addition to this system.

        Args:
            prediction_len: Number of future steps to predict.
            history_len: Number of observed steps provided to the model.
            dt: Timestep used by some predictors.
            device: Torch device string, e.g. "cpu", "cuda:0".
        """
        self._dt = dt
        self.device = device
        self.history_len = history_len
        self.prediction_len = prediction_len
    @abstractmethod
    def __call__(self, tracking_results):
        return NotImplementedError("This is an abstract method.")
    

