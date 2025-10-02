# eigen/eigen_predictor_class.py

import os
import sys
from types import SimpleNamespace
from typing import Dict, Any

import numpy as np
import torch

# Local module path so we can import 'baseline'
_DATA_DIR = os.path.dirname(__file__)
sys.path.append(_DATA_DIR)
import importlib
baseline = importlib.import_module("baseline")

from EigenTrajectory import EigenTrajectory
from .utils.utils import get_exp_config, DotDict
from .utils import trainer as trainer_mod

from ..wrapper_predictor import BasePredictors


class EigenTrajectoryPredictor(BasePredictors):
    def __init__(
        self,
        prediction_len: int = 12,
        history_len: int = 8,
        dt: float = 0.1,
        device: str = "cpu",
        *,
        cfg: str = "src/canvas/predictors/eigen/json_files/eigentrajectory-stgcnn-lobby_data.json",
        model_path: str = "src/canvas/predictors/eigen/lobby_data/model_best.pth",
        tag: str = "EigenTrajectory-TEMP",
        gpu_id: str = "cpu",
        test: bool = True,
    ):
        # Initialize common fields (history_len, prediction_len, dt, device)
        super().__init__(
            prediction_len=prediction_len,
            history_len=history_len,
            dt=dt,
            device=device,
        )

        self.cfg = cfg
        self.model_path = model_path

        # Keep this name so the existing __call__ body can stay the same
        self.target_prediction_length = prediction_len

        # (Optional) Preserve existing behavior for this baseline
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # Build args expected by trainer
        args = SimpleNamespace(cfg=cfg, tag=tag, gpu_id=gpu_id, test=test)

        # Load config + baseline-dependent components
        hyper_params = get_exp_config(cfg)
        BasePredictor = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
        hook_func = DotDict({
            "model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
            "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
            "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook
        })
        ModelTrainer = getattr(
            trainer_mod,
            *[name for name in trainer_mod.__dict__.keys() if hyper_params.baseline in name.lower()]
        )

        # Instantiate trainer and load weights
        trainer = ModelTrainer(
            base_model=BasePredictor,
            model=EigenTrajectory,
            hook_func=hook_func,
            args=args,
            hyper_params=hyper_params,
        )
        trainer.load_model(model_path)

        self.trainer = trainer
        self.hyper_params = hyper_params


    @staticmethod
    def interpolate_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
        L = traj.shape[0]
        if L == target_length:
            return traj
        new_traj = np.empty((target_length, traj.shape[1]))
        original_indices = np.linspace(0, 1, num=L)
        new_indices = np.linspace(0, 1, num=target_length)
        for i in range(traj.shape[1]):
            new_traj[:, i] = np.interp(new_indices, original_indices, traj[:, i])
        return new_traj


    def __call__(self, dynamic_obs: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        data = dynamic_obs
        eligible_tensors = []
        eligible_keys = []  # 추적 가능한 객체의 key 목록

        for key in sorted(data.keys()):
            arr = data[key]
            # 배열이 2차원이고 열이 2개이며, 최소 history_len행 이상의 데이터를 가지고 있는지 확인
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= self.history_len:
                truncated = arr[-self.history_len:]
                eligible_tensors.append(truncated)
                eligible_keys.append(key)
            elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] < self.history_len:
                truncated = arr[-self.history_len:]
                pad_count = self.history_len - truncated.shape[0]
                pad_array = np.tile(arr[0:1], (pad_count, 1))
                truncated = np.concatenate((pad_array, truncated), axis=0)
                eligible_tensors.append(truncated)
                eligible_keys.append(key)

        target_prediction_length = self.target_prediction_length  # keep your constant

        if eligible_tensors:
            final_tensor = torch.tensor(np.stack(eligible_tensors), dtype=torch.float32)
            with torch.no_grad():
                results = self.trainer.predict(final_tensor)
            results = results['recon_traj'].mean(axis=0)  # keep your mean over samples
            if isinstance(results, torch.Tensor):
                results = results.detach().cpu().numpy()

            final_dict: Dict[Any, np.ndarray] = {}
            for i, key in enumerate(eligible_keys):
                pred = results[i]
                # 예측 결과 길이가 target_prediction_length가 아니면 보간하여 맞춤
                if pred.shape[0] != target_prediction_length:
                    pred = self.interpolate_trajectory(pred, target_prediction_length)
                final_dict[key] = pred.astype(np.float32)
            return final_dict
        else:
            # No eligible trajectories
            return {}
