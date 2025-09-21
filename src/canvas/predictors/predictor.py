import os
import numpy as np
from .linear_predictor import LinearPredictor
from .trajectron_predictor import TrajectronPredictor
from .eigen.eigen_predictor_class import EigenTrajectoryPredictor
from .gp_predictor import GaussianProcessPredictor
from .koopcast_predictor import Koopcast_predictor
import torch


class Predictors:
    def __init__(self, chosen_predictor='linear',
                 prediction_len=12,history_len=8,
                 dt=0.1,smoothing_factor=0.75,
                 model_dir='src/canvas/predictors/eigen/models/lobby_data/model_best.pth', #'src/canvas/predictors/trajectron/models_11_Feb_2025_10_01_22eth_vel_ar3'
                 device='cpu',
                 dataset='Lobby',
                 cfg='src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-lobby_data.json'):
        """Simple access class for different predictors.

        Args:
            chosen_predictor: One of {"linear","lin","trajectron","traj","tpp",
                "eigen","eigentrajectory","eigen_traj","pytorch","torch"}.
            prediction_len: Number of future steps to predict.
            history_len: Number of observed steps provided to the model.
            dt: Timestep used by some predictors.
            smoothing_factor: Smoothing factor for the Linear predictor.
            model_dir: For Trajectron++ (dir), Eigen (model_path), or PyTorch (full model file).
            device: Torch device string, e.g. "cpu", "cuda:0".
            cfg: JSON config path (For Eigen only in the current setup).
        """
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
            if(dataset=='ETH'):
                model_dir="src/canvas/predictors/trajectron/models_11_Feb_2025_10_01_22eth_vel_ar3"
            elif(dataset=='Hotel'):
                model_dir="src/canvas/predictors/trajectron/models_10_Feb_2025_21_00_50hotel_vel_ar3"
            elif(dataset=='Univ'):
                model_dir="src/canvas/predictors/trajectron/models_10_Feb_2025_15_01_26_univ_vel_ar3"
            elif(dataset=='Zara01'):
                model_dir="src/canvas/predictors/trajectron/models_10_Feb_2025_11_19_14_zara01_vel_ar3"
            elif(dataset=='Zara02'):
                model_dir="src/canvas/predictors/trajectron/models_03_Feb_2025_14_11_39_zara02_vel_ar3"
            elif(dataset=='Lobby'):
                model_dir="src/canvas/predictors/trajectron/models_17_Mar_2025_22_52_52lobby_data_ar3"
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: ETH, Hotel, Univ, Zara01, Zara02, Lobby."
                )

            self.PredictorModel = TrajectronPredictor(
                prediction_len=prediction_len,
                model_dir=model_dir,
                device=device,
            )

        elif name in ("eigen", "eigentrajectory", "eigen_traj"):
            # EigenTrajectory predictor
            if(dataset=='ETH'):
                model_dir="src/canvas/predictors/eigen/models/eth/model_best.pth"
                cfg="src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-eth.json"
            elif(dataset=='Hotel'):
                model_dir="src/canvas/predictors/eigen/models/hotel/model_best.pth"
                cfg="src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-hotel.json"
            elif(dataset=='Univ'):
                model_dir="src/canvas/predictors/eigen/models/uni/model_best.pth"
                cfg="src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-uni.json"
            elif(dataset=='Zara01'):
                model_dir="src/canvas/predictors/eigen/models/zara01/model_best.pth"
                cfg="src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-zara01.json"
            elif(dataset=='Zara02'):
                model_dir="src/canvas/predictors/eigen/models/zara02/model_best.pth"
                cfg="src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-zara02.json"
            elif(dataset=='Lobby'):
                model_dir="src/canvas/predictors/eigen/models/lobby_data/model_best.pth"
                cfg="src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-lobby_data.json"
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: ETH, Hotel, Univ, Zara01, Zara02, Lobby."
                )

            self.PredictorModel = EigenTrajectoryPredictor(
                prediction_len=prediction_len,
                history_len=history_len,
                cfg=cfg,
                model_path=model_dir,
            )
            
        elif name in ("gp","gpy", "gaussianprocess", "gaussian_process"):
            # GaussianProcess predictor
            self.PredictorModel=GaussianProcessPredictor(
                prediction_len=prediction_len,
                history_len=history_len,
                dt=dt,
                device=device,
            )
        elif name in ("koopcast","mdnkoopman","mdn_koopman"):
            # KoopCast predictor
            if(dataset=='ETH'):
                k_path='src/canvas/predictors/koopcast/data/biwi_eth_koopman_K_1.npy'
                cfg='src/canvas/predictors/koopcast/data/biwi_eth_cfg.json'
                mdn_pt='src/canvas/predictors/koopcast/data/biwi_eth_mdn.pt'
            elif(dataset=='Hotel'):
                k_path='src/canvas/predictors/koopcast/data/biwi_hotel_koopman_K_1.npy'
                cfg='src/canvas/predictors/koopcast/data/biwi_hotel_cfg.json'
                mdn_pt='src/canvas/predictors/koopcast/data/biwi_hotel_mdn.pt'
            elif(dataset=='Univ'):
                k_path='src/canvas/predictors/koopcast/data/univ_koopman_K_1.npy'
                cfg='src/canvas/predictors/koopcast/data/univ_cfg.json'
                mdn_pt='src/canvas/predictors/koopcast/data/univ_mdn.pt'
            elif(dataset=='Zara01'):
                k_path='src/canvas/predictors/koopcast/data/crowds_zara01_koopman_K_1.npy'
                cfg='src/canvas/predictors/koopcast/data/crowds_zara01_cfg.json'
                mdn_pt='src/canvas/predictors/koopcast/data/crowds_zara01_mdn.pt'
            elif(dataset=='Zara02'):
                k_path='src/canvas/predictors/koopcast/data/crowds_zara02_koopman_K_1.npy'
                cfg='src/canvas/predictors/koopcast/data/crowds_zara02_cfg.json'
                mdn_pt='src/canvas/predictors/koopcast/data/crowds_zara02_mdn.pt'
            elif(dataset=='Lobby'):
                k_path='src/canvas/predictors/koopcast/data/0_koopman_K_1.npy'
                cfg='src/canvas/predictors/koopcast/data/0_cfg.json'
                mdn_pt='src/canvas/predictors/koopcast/data/0_mdn.pt'
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: ETH, Hotel, Univ, Zara01, Zara02, Lobby."
                    )

            self.PredictorModel=Koopcast_predictor(
                prediction_len=prediction_len,
                history_len=history_len,
                dt=dt,
                K_path=k_path,
                cfg_path=cfg,
                mdn_pt_path=mdn_pt,
                device=device,
            )
        elif name in ("pytorch", "torch"):
            self.PredictorModel=torch.load(model_dir,map_location=device)
            self.PredictorModel.eval()
        else:
            raise ValueError(
                f"Unknown predictor '{chosen_predictor}'. "
                "Expected one of: Linear, Trajectron, Eigen."
            )
    def __call__(self,dynamic_obs):
        """Run the selected predictor.

        Args:
            dynamic_obs: Observation/input structure required by the underlying
                predictor (often a tensor or dict-like). This wrapper forwards
                the input unchanged.

        Returns:
            Whatever the underlying predictor returns (commonly a dict of shape
            ``{B:[prediction_len, D]}``).

        Example:
            >>> preds = Predictors("linear", prediction_len=12, dt=0.1)
            >>> y = preds(dynamic_obs)  # forwards to LinearPredictor(dynamic_obs)
        """
        return self.PredictorModel(dynamic_obs)
    def predictor(self):
        """Return the underlying predictor instance.

        Use this when you need direct access (e.g., to call ``to(device)``,
        set modes, or inspect parameters).

        Example:
            >>> p = Predictors("eigen").predictor()
            >>> p.eval()  # switch to eval mode
        """
        return self.PredictorModel