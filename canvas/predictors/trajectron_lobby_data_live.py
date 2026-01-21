import os
import json
import torch
import sys

sys.path.append("/home/core/Documents/MPC20250312/prediction/trajectron")

import numpy as np
import pandas as pd
from .trajectron.model.model_registrar import ModelRegistrar
from .trajectron.model.trajectron import Trajectron
from .trajectron.environment import Environment, Scene, Node, derivative_of

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}

# Define the multi-index columns for the data
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


def process_tracking_results(tracking_results, dt=0.4):
    """
    Processes a dictionary of tracking results into a Scene.

    Parameters:
      tracking_results : dict
         Keys are object ids (혹은 Node 객체) and values are histories (array-like with shape (T, 2)
         where the two columns represent x and y positions).
      dt : float
         Time difference between timesteps.

    Returns:
      scene : Scene
         A scene with one Node per object id, where each node's trajectory is padded to the same length.
    """
    # tracking_results가 비어있으면 빈 Scene 반환
    if not tracking_results:
        return Scene(timesteps=0, dt=dt, name="tracking_results")

    # 각 객체의 trajectory를 numpy array로 변환 및 reshape 처리 (1차원인 경우)
    valid_histories = {}
    for key, history in tracking_results.items():
        arr = np.array(history)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            if arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            else:
                continue  # 형식이 올바르지 않은 경우 건너뜁니다.
        if arr.shape[0] < 2:
            continue
        valid_histories[key] = arr

    if not valid_histories:
        return Scene(timesteps=0, dt=dt, name="tracking_results")

    # 모든 객체의 trajectory 길이 중 최대 길이를 구함
    max_timesteps = max(arr.shape[0] for arr in valid_histories.values())

    # 모든 객체에 대해 max_timesteps에 맞춰 padding (마지막 값 반복)
    padded_histories = {}
    all_x, all_y = [], []
    for key, arr in valid_histories.items():
        if arr.shape[0] < max_timesteps:
            pad_length = max_timesteps - arr.shape[0]
            arr = np.concatenate([arr, np.repeat(arr[-1:, :], pad_length, axis=0)], axis=0)
        padded_histories[key] = arr
        all_x.extend(arr[:, 0])
        all_y.extend(arr[:, 1])

    global_mean_x = np.mean(all_x) if all_x else 0
    global_mean_y = np.mean(all_y) if all_y else 0

    scene = Scene(timesteps=max_timesteps, dt=dt, name="tracking_results")
    scene.pos_x_mean = global_mean_x
    scene.pos_y_mean = global_mean_y

    for key, arr in padded_histories.items():
        # key가 Node 객체라면 id를 추출하고, 아니면 문자열로 통일
        if hasattr(key, 'id'):
            obj_id = str(key.id)
        else:
            obj_id = str(key)
        # global mean 보정
        arr_centered = arr - np.array([global_mean_x, global_mean_y])
        x = arr_centered[:, 0]
        y = arr_centered[:, 1]
        vx = derivative_of(x, dt)
        vy = derivative_of(y, dt)
        ax = derivative_of(vx, dt)
        ay = derivative_of(vy, dt)

        data_dict = {
            ('position', 'x'): x,
            ('position', 'y'): y,
            ('velocity', 'x'): vx,
            ('velocity', 'y'): vy,
            ('acceleration', 'x'): ax,
            ('acceleration', 'y'): ay
        }
        node_data = pd.DataFrame(data_dict, columns=data_columns)
        node = Node(node_type='PEDESTRIAN', node_id=obj_id, data=node_data, first_timestep=0)
        scene.nodes.append(node)

    return scene


#python evaluate_trajectron_lobby_data.py --data ../processed/lobby_data_test.pkl  --model ./models/models_17_Mar_2025_22_52_52lobby_data_ar3

def predict(tracking_result, prediction_len=None,model_dir="./", env=None, eval_stg=None, hyperparams=None):
    """
    Predict trajectories using the Trajectron model.
    :param tracking_result: dict, keys are object ids and values are histories.
    :param prediction_len: 예측하고자 하는 길이(타임스텝 수). 제공되면 hyperparameter를 오버라이드함.
    :return: predictions dictionary.
    """

    scene = process_tracking_results(tracking_result, dt=0.1)
    env.scenes = [scene]

    eval_stg.set_environment(env)
    eval_stg.set_annealing_params()
    # 여기서 원하는 예측 길이가 있다면 hyperparameter를 오버라이드합니다.
    if prediction_len is not None:
        hyperparams['prediction_horizon'] = prediction_len

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius_val = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius_val)

    scenes = env.scenes

    # get_timesteps_data에 전달되는 max_ft도 prediction_len으로 설정합니다.
    ph = hyperparams['prediction_horizon']
    with torch.no_grad():
        for scene in scenes:
            t = scene.timesteps - 1  # 최신 시점에서 예측 수행
            timesteps = np.array([t])
            # min_future_timesteps은 그대로 0, max_ft를 ph로 전달합니다.
            predictions = eval_stg.predict(
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_history_timesteps=7,
                min_future_timesteps=0,
                z_mode=True,
                gmm_mode=True,
                full_dist=False
            )
            if not predictions:
                continue
            pos_x_mean = scene.pos_x_mean
            pos_y_mean = scene.pos_y_mean
            for primary_idx, pedestrians in predictions.items():
                for ped_key, ped_data in pedestrians.items():
                    ped_data[..., 0] = ped_data[..., 0] + pos_x_mean
                    ped_data[..., 1] = ped_data[..., 1] + pos_y_mean
    return predictions
