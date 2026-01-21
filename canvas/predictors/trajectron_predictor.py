# trajectron_predictor.py

import numpy as np
import os
import json
from .trajectron_lobby_data_live import predict
from .trajectron.environment.node import Node
from .trajectron.environment.node_type import NodeType
from .wrapper_predictor import BasePredictors
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

def linear_extrapolation(traj, prediction_len, dt=0.1, noise_floor=0.01):
    """
    간단한 선형 외삽 함수.
    마지막 3 프레임의 평균 위치와 최근 3 프레임의 평균 속도를 사용하여
    prediction_len 길이의 미래 trajectory를 생성합니다.
    """
    if traj.shape[0] >= 3:
        pos0 = np.mean(traj[-3:], axis=0)
        v = (traj[1:] - traj[:-1]) / dt
        v_mean = np.mean(v[-min(3, v.shape[0]):], axis=0)
        if np.linalg.norm(v_mean) < 1e-3:
            v_mean = v_mean + noise_floor  # 최소 노이즈 플로어
        pos_next = pos0 + dt * np.outer(np.arange(1, prediction_len + 1), v_mean)
        return pos_next
    else:
        return np.tile(traj[-1, :], (prediction_len, 1))


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    return trajectron, hyperparams

class TrajectronPredictor(BasePredictors):
    def __init__(
        self,
        prediction_len: int = 12,
        history_len: int = 8,
        dt: float = 0.1,
        device: str = "cpu",
        *,
        model_dir: str = "./",
    ):
        # Common interface fields: prediction_len, history_len, dt, device
        super().__init__(
            prediction_len=prediction_len,
            history_len=history_len,
            dt=dt,
            device=device,
        )
        self.env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN)] = 3.0
        self.env.attention_radius = attention_radius
        self.eval_stg, self.hyperparams = load_model(model_dir, self.env, ts=100)
        self.model_dir = model_dir

    def __call__(self, tracking_results):
        if not tracking_results:
            return {}

        converted_results = {}
        for obj_id, traj in tracking_results.items():
            dynamic_type = NodeType("PEDESTRIAN", 1)
            node = Node(node_type=dynamic_type, node_id=str(obj_id), data=traj)
            converted_results[node] = traj

        # Trajectron 모델을 통해 예측 수행
        predictions_dict = predict(
            converted_results, self.prediction_len, model_dir=self.model_dir,env=self.env, eval_stg=self.eval_stg, hyperparams=self.hyperparams
        )

        return predictions_dict
