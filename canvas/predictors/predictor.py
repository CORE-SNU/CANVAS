import os
import pathlib
import numpy as np
from .linear_predictor import LinearPredictor
from .trajectron_predictor import TrajectronPredictor
from .eigen.eigen_predictor_class import EigenTrajectoryPredictor
from .gp_predictor import GaussianProcessPredictor
from .koopcast_predictor import Koopcast_predictor
import torch


ASSET_DIR = pathlib.Path(__file__).parent.parent.parent / 'assets'


class Predictors:
    def __init__(self, chosen_predictor='linear',
                 prediction_len=12,history_len=8,
                 dt=0.1,smoothing_factor=0.75,
                 model_dir='canvas/predictors/eigen/models/lobby_data/model_best.pth',
                 device='cpu',
                 dataset='Lobby',
                 cfg='canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-lobby_data.json'):
        """Simple access class for different predictors.

        Args:
            chosen_predictor: One of {"linear","lin","trajectron","traj","tpp",
                "eigen","eigentrajectory","eigen_traj","pytorch","torch", "stgcnn","socialvae"}.
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
        dataset=dataset.strip().lower()

        if name in ("linear", "lin"):
            # Linear predictor
            self.PredictorModel = LinearPredictor(
                prediction_len=prediction_len,
                history_len=history_len,
                smoothing_factor=0.75,
                dt=dt,
            )

        elif name in ("trajectron", "traj", "tpp"):

            trajectron_dirnames = {
                'eth': 'eth_vel_ar3',
                'hotel': 'hotel_vel_ar3',
                'univ': 'univ_vel_ar3',
                'zara1': 'zara01_vel_ar3',
                'zara2': 'zara02_vel_ar3',
                'snu-asri': 'snu-asri_ar3',
                'snu-asri-ood': 'snu-asri_ar3'
            }

            if dataset in trajectron_dirnames:
                model_dir = ASSET_DIR / 'models' / 'trajectron' / trajectron_dirnames[dataset]
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: eth, hotel, univ, zara1, zara2, snu-asri, snu-asri-ood"
                )

            self.PredictorModel = TrajectronPredictor(
                prediction_len=prediction_len,
                model_dir=model_dir,
                device=device,
            )

        elif name in ("eigen", "eigentrajectory", "eigen_traj"):
            # EigenTrajectory predictor
            if(dataset=='eth'):
                model_dir="canvas/predictors/eigen/models/eth/model_best.pth"
                cfg="canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-eth.json"
            elif(dataset=='hotel'):
                model_dir="canvas/predictors/eigen/models/hotel/model_best.pth"
                cfg="canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-hotel.json"
            elif(dataset=='univ'):
                model_dir="canvas/predictors/eigen/models/uni/model_best.pth"
                cfg="canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-uni.json"
            elif(dataset=='zara1' or dataset=='zara01'):
                model_dir="canvas/predictors/eigen/models/zara01/model_best.pth"
                cfg="canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-zara01.json"
            elif(dataset=='zara02' or dataset=='zara2'):
                model_dir="canvas/predictors/eigen/models/zara02/model_best.pth"
                cfg="canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-zara02.json"
            elif(dataset=='snu-asri' or dataset=='snu-asri-ood'):
                model_dir="canvas/predictors/eigen/models/lobby_data/model_best.pth"
                cfg="canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-lobby_data.json"
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: eth, hotel, univ, zara1, zara2, snu-asri, snu-asri-ood."
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
            if(dataset=='eth'):
                k_path='canvas/predictors/koopcast/data/biwi_eth_koopman_K_1.npy'
                cfg='canvas/predictors/koopcast/data/biwi_eth_cfg.json'
                mdn_pt='canvas/predictors/koopcast/data/biwi_eth_mdn.pt'
            elif(dataset=='hotel'):
                k_path='canvas/predictors/koopcast/data/biwi_hotel_koopman_K_1.npy'
                cfg='canvas/predictors/koopcast/data/biwi_hotel_cfg.json'
                mdn_pt='canvas/predictors/koopcast/data/biwi_hotel_mdn.pt'
            elif(dataset=='univ'):
                k_path='canvas/predictors/koopcast/data/univ_koopman_K_1.npy'
                cfg='canvas/predictors/koopcast/data/univ_cfg.json'
                mdn_pt='canvas/predictors/koopcast/data/univ_mdn.pt'
            elif(dataset=='zara01' or dataset=='zara1'):
                k_path='canvas/predictors/koopcast/data/crowds_zara01_koopman_K_1.npy'
                cfg='canvas/predictors/koopcast/data/crowds_zara01_cfg.json'
                mdn_pt='canvas/predictors/koopcast/data/crowds_zara01_mdn.pt'
            elif(dataset=='zara02' or dataset=='zara2'):
                k_path='canvas/predictors/koopcast/data/crowds_zara02_koopman_K_1.npy'
                cfg='canvas/predictors/koopcast/data/crowds_zara02_cfg.json'
                mdn_pt='canvas/predictors/koopcast/data/crowds_zara02_mdn.pt'
            elif(dataset=='snu-asri' or dataset=='snu-asri-ood'):
                k_path='canvas/predictors/koopcast/data/0_koopman_K_1.npy'
                cfg='canvas/predictors/koopcast/data/0_cfg.json'
                mdn_pt='canvas/predictors/koopcast/data/0_mdn.pt'
            else:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: eth, hotel, univ, zara1, zara2, snu-asri, snu-asri-ood."
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
        elif name in ("SocialVAE","socialvae","social_vae"):
            from .SocialVAE.social_vae_runner import Social_VAE_Predictor
            # dataset is a string like 'ETH', 'Hotel', 'Univ', 'Zara01', 'Zara02', 'Lobby', etc.
            key = dataset.strip().lower()

            cfg_model_map = {
                # ETH/Hotel/Univ
                'eth':        ("/config/eth.py",        "/models/eth"),
                'hotel':      ("/config/hotel.py",      "/models/hotel"),
                'univ':       ("/config/univ.py",       "/models/univ"),

                # Zara
                'zara01':     ("/config/zara01.py",     "/models/zara01"),
                'zara1':      ("/config/zara01.py",     "/models/zara01"),
                'zara02':     ("/config/zara02.py",     "/models/zara02"),
                'zara2':      ("/config/zara02.py",     "/models/zara02"),

                # Lobby
                'snu-asri':      ("/config/lobby_data.py", "/models/lobby"),
                'snu-asri-ood': ("/config/lobby_data.py", "/models/lobby"),
            }

            if key not in cfg_model_map:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: eth, hotel, univ, zara1, zara2, snu-asri, snu-asri-ood."
                )

            cfg, model_path = cfg_model_map[key]
            self.PredictorModel=Social_VAE_Predictor(
                prediction_len=prediction_len,
                history_len=history_len,
                dt=dt,
                device=device,
                cfg= cfg,
                model_path=model_path,
            )
        elif name in ("STGCNN","socialstgcnn","social_stgcnn","stgcnn","social-stgcnn"):
            from .Social_STGCNN.STGCNN_live_test import STGCNN_Predictor
            key = dataset.strip().lower()

            folder_map = {
                'eth':       'social-stgcnn-eth',
                'hotel':     'social-stgcnn-hotel',
                'univ':      'social-stgcnn-univ',
                'zara01':    'social-stgcnn-zara1',   # handle 01 -> 1
                'zara1':     'social-stgcnn-zara1',
                'zara02':    'social-stgcnn-zara2',   # handle 02 -> 2
                'zara2':     'social-stgcnn-zara2',
                'snu-asri':     'social-stgcnn-lobby_data',
                'snu-asri-ood':'social-stgcnn-lobby_data',
            }

            if key not in folder_map:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    "Expected one of: eth, hotel, univ, zara1, zara2, snu-asri, snu-asri-ood."
                )

            model_dir = f"/checkpoint/{folder_map[key]}"
            self.PredictorModel=STGCNN_Predictor(
                prediction_len=prediction_len,
                history_len=history_len,
                dt=dt,
                device=device,
                model_dir=model_dir,
            )
        elif name in ("pytorch", "torch"):
            self.PredictorModel=torch.load(model_dir,map_location=device)
            self.PredictorModel.eval()
        else:
            raise ValueError(
                f"Unknown predictor '{chosen_predictor}'. "
                "Expected one of: Linear, Trajectron, EigenTrajectory, GaussianProcess, Koopcast, SocialVAE, Social-STGCNN, PyTorch."
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