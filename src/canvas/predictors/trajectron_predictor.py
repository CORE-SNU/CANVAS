import numpy as np
from trajectron_lobby_data_live import predict
from .trajectron.environment.node import Node
from .trajectron.environment.node_type import NodeType

def linear_extrapolation(traj, prediction_len, dt=0.1, noise_floor=0.01):
    """
    간단한 선형 외삽 함수.
    입력 trajectory가 충분히 길다면 마지막 3 프레임의 평균 위치와 최근 3 프레임의 평균 속도를 사용하여
    prediction_len 길이의 미래 trajectory를 생성합니다.
    만약 계산된 속도가 너무 작으면 noise_floor 값을 추가하여 최소한의 변화가 있도록 합니다.
    """
    if traj.shape[0] >= 3:
        pos0 = np.mean(traj[-3:], axis=0)
        v = (traj[1:] - traj[:-1]) / dt
        v_mean = np.mean(v[-min(3, v.shape[0]):], axis=0)
        if np.linalg.norm(v_mean) < 1e-3:
            v_mean = v_mean + noise_floor  # 최소 노이즈 플로어 추가
        pos_next = pos0 + dt * np.outer(np.arange(1, prediction_len + 1), v_mean)
        return pos_next
    else:
        return np.tile(traj[-1, :], (prediction_len, 1))

class TrajectronPredictor:
    def __init__(self, prediction_len, model_dir="./", device='cpu'):
        self.prediction_len = prediction_len
        self.model_dir = model_dir
        self.device = device

    def __call__(self, tracking_results):
        # 동적 객체가 없으면 빈 dict 반환
        if not tracking_results:
            return {}

        converted_results = {}
        # linear predictor와 달리, dummy_data 대신 실제 trajectory 데이터를 함께 저장합니다.
        for obj_id, traj in tracking_results.items():
            dynamic_type = NodeType('PEDESTRIAN', 1)
            node = Node(node_type=dynamic_type, node_id=str(obj_id), data=traj)
            converted_results[node] = traj

        # Trajectron 모델을 통해 예측을 수행합니다.
        predictions_dict = predict(converted_results, self.prediction_len,model_dir=self.model_dir)

        if isinstance(predictions_dict, dict) and len(predictions_dict) == 1:
            frame, predictions_inner = next(iter(predictions_dict.items()))
        else:
            predictions_inner = predictions_dict

        final_predictions = {}
        for key, traj_pred in predictions_inner.items():
            key_id = int(str(key).split('/')[-1])  # 문자열 대신 정수로 변환
            traj_pred = np.squeeze(traj_pred)
            if traj_pred.ndim == 1:
                traj_pred = traj_pred.reshape(-1, 2)
            desired_length = self.prediction_len
            current_length = traj_pred.shape[0]
            if current_length > 1 and np.std(traj_pred) < 1e-3:
                orig_traj = None
                for node_obj, orig in converted_results.items():
                    if int(node_obj.id) == key_id:  # 비교도 int로
                        orig_traj = np.array(orig)
                        break
                if orig_traj is not None:
                    traj_pred = linear_extrapolation(orig_traj, desired_length, dt=0.1)
                    noise = np.random.randn(*traj_pred.shape) * 0.01
                    traj_pred = traj_pred + noise
                else:
                    pad_length = desired_length - current_length
                    pad_values = np.repeat(traj_pred[-1:, :], pad_length, axis=0)
                    traj_pred = np.concatenate([traj_pred, pad_values], axis=0)
            else:
                if current_length < desired_length:
                    pad_length = desired_length - current_length
                    pad_values = np.repeat(traj_pred[-1:, :], pad_length, axis=0)
                    traj_pred = np.concatenate([traj_pred, pad_values], axis=0)
                elif current_length > desired_length:
                    traj_pred = traj_pred[:desired_length, :]
            final_predictions[key_id] = traj_pred  # key를 int로 저장

        # 입력 tracking_results에 있지만 예측 결과에 포함되지 않은 객체 처리
        for node_obj, orig_traj in converted_results.items():
            key_id = int(node_obj.id)  # 여기도 int로
            if key_id not in final_predictions:
                orig_traj = np.array(orig_traj)
                if orig_traj.ndim == 1:
                    orig_traj = orig_traj.reshape(-1, 2)
                traj_pred = linear_extrapolation(orig_traj, self.prediction_len, dt=0.1)
                noise = np.random.randn(*traj_pred.shape) * 0.01
                traj_pred = traj_pred + noise
                final_predictions[key_id] = traj_pred

        return final_predictions
