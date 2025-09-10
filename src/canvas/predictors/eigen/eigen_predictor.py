import torch
import argparse
import sys
sys.path.append('/home/snowhan1021/tools_paper/CANavi/prediction/eigen')
import importlib
baseline = importlib.import_module("baseline")
from EigenTrajectory import *
from .utils.utils import *
import os
import numpy as np
from .utils import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default="/home/snowhan1021/tools_paper/CANavi/prediction/eigen/eigentrajectory-stgcnn-lobby_data.json", type=str, help="config file path")
parser.add_argument('--tag', default="EigenTrajectory-TEMP", type=str, help="personal tag for the model")
parser.add_argument('--gpu_id', default="cpu", type=str, help="gpu id for the model")
parser.add_argument('--test', default=True, type=str, help="gpu id for the model")

args = parser.parse_args()

model_path = "/home/snowhan1021/tools_paper/CANavi/prediction/eigen/lobby_data/model_best.pth"

hyper_params = get_exp_config(args.cfg)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
hook_func = DotDict({
    "model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
    "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
    "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook
})
ModelTrainer = getattr(trainer, *[s for s in trainer.__dict__.keys() if hyper_params.baseline in s.lower()])
trainer = ModelTrainer(base_model=PredictorModel, model=EigenTrajectory, hook_func=hook_func,
                       args=args, hyper_params=hyper_params)
trainer.load_model(model_path)

def interpolate_trajectory(traj, target_length):
    # traj: (L, 2), target_length: 목표 길이
    L = traj.shape[0]
    if L == target_length:
        return traj
    new_traj = np.empty((target_length, traj.shape[1]))
    original_indices = np.linspace(0, 1, num=L)
    new_indices = np.linspace(0, 1, num=target_length)
    for i in range(traj.shape[1]):
        new_traj[:, i] = np.interp(new_indices, original_indices, traj[:, i])
    return new_traj

def eigen_predictor(dynamic_obs):
    data = dynamic_obs
    eligible_tensors = []
    eligible_keys = []  # 추적 가능한 객체의 key 목록
    for key in sorted(data.keys()):
        arr = data[key]
        # 배열이 2차원이고 열이 2개이며, 최소 8행 이상의 데이터를 가지고 있는지 확인
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 8:
            truncated = arr[-8:]
            eligible_tensors.append(truncated)
            eligible_keys.append(key)
        elif arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] < 8:
            truncated = arr[-8:]
            pad_count = 8 - truncated.shape[0]
            pad_array = np.tile(arr[0:1], (pad_count, 1))
            truncated = np.concatenate((pad_array, truncated), axis=0)
            eligible_tensors.append(truncated)
            eligible_keys.append(key)

    # GridMPC에서 기대하는 예측 길이 (예: 16)
    target_prediction_length = 16

    if eligible_tensors:
        final_tensor = torch.tensor(np.stack(eligible_tensors), dtype=torch.float32)
        results = trainer.predict(final_tensor)
        results = results['recon_traj'].mean(axis=0)
        if isinstance(results, torch.Tensor):
            results = results.detach().cpu().numpy()
        final_dict = {}
        for i, key in enumerate(eligible_keys):
            pred = results[i]
            # 예측 결과 길이가 target_prediction_length가 아니면 보간하여 맞춤
            if pred.shape[0] != target_prediction_length:
                pred = interpolate_trajectory(pred, target_prediction_length)
            final_dict[key] = pred
        return final_dict
    else:
        #print("No eligible trajectories found in eigen_predictor. Returning empty dictionary.")
        return {}

