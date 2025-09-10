import os
import numpy as np
from prediction.linear_predictor import LinearPredictor
from trajectron_predictor import TrajectronPredictor
from prediction.eigen.eigen_predictor_class import EigenTrajectoryPredictor


class Predictors:
    def __init__(self, chosen_predictor='linear',
                 prediction_len=12,history_len=8,
                 dt=0.1,smoothing_factor=0.75,
                 model_dir='/prediction/trajectron/models_11_Feb_2025_10_01_22eth_vel_ar3',
                 device='cpu',
                 cfg='/prediction/eigen/eigentrajectory-stgcnn-lobby_data.json'):

        self._dt = dt
        name = str(chosen_predictor).strip().lower()

        if name in ("linear", "lin"):
            # Linear predictor
            self.PredictorModel = LinearPredictor(
                prediction_len=prediction_len,
                history_len=history_len,
                smoothing_factor=0.75,
                dt=dt,
            )

        elif name in ("trajectron", "traj", "tpp"):
            # Trajectron++ predictor
            self.PredictorModel = TrajectronPredictor(
                prediction_len=prediction_len,
                model_dir=model_dir,
                device=device,
            )

        elif name in ("eigen", "eigentrajectory", "eigen_traj"):
            # Eigen predictor)
            self.PredictorModel = EigenTrajectoryPredictor(
                prediction_len=prediction_len,
                history_len=history_len,
                model_path=model_dir,
                cfg=cfg,)

        else:
            raise ValueError(
                f"Unknown predictor '{chosen_predictor}'. "
                "Expected one of: Linear, Trajectron, Eigen."
            )
    def __call__(self,dyanmic_obs):
        return self.PredictorModel(dynamic_obs)
    def predictor(self):
        return self.PredictorModel