import os
import re
import math
import pickle
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import torch
import torch.nn as nn

###########################################
# 1. 기본 장애물 정보 (lobby/train)
###########################################
default_obstacles_meter = {
    'pillar_left': [(0.12, -0.62), (-0.16, -0.62), (-0.38, -0.88), (-0.3, -1.38), (0.16, -1.44), (0.42, -1.16), (0.36, -0.76)],
    'pillar_right': [(5.62, -0.96), (5.48, -1.36), (5.62, -1.64), (6.12, -1.80), (6.36, -1.42), (6.26, -1.04)],
    'wall_north': [(0.68, 3.98), (0.08, 1.34), (11.80, 0.58), (11.76, 2.52)],
    'wall_east': [(8.24, -1.24), (7.66, -10.02), (11.58, -10.30), (11.70, -1.54)],
    'wall_south': [(-2.70, -8.38), (-2.78, -9.44), (7.60, -9.92), (7.72, -8.86)],
    'wall_west': [(-4.12, 0.02), (-4.42, -8.64), (-2.66, -8.80), (-2.24, -0.22)],
    'wall_northwest': [(-3.34, 1.46), (-1.88, 1.30), (-1.62, 4.48), (-3.08, 4.38)],
    'fanuc': [(2.44, -1.86), (2.26, -4.38), (3.20, -4.44), (3.42, -1.96)],
    'entrance_wall_left': [(0.38, -6.36), (0.26, -8.82), (1.68, -8.92), (1.84, -6.48)],
    'entrance_wall_right': [(3.66, -6.44), (3.42, -8.82), (4.86, -9.00), (5.04, -6.54)]
}
default_obstacle_polygons = [Polygon(coords) for coords in default_obstacles_meter.values()]

###########################################
# 2. lobby2 파일별 장애물 정보
###########################################
obstacles_meter_zara01 = {
    'obstacle1': [(-0.02104651, 8.0), (-0.02104651, 12.3864436), (9.5, 12.3864436), (9.5, 8.0), (-0.02104651, 8.0)],
    'obstacle2': [(10.0, 11.2), (10.0, 13.3864436), (12.0, 13.3864436), (12.0, 11.2), (10.0, 11.2)],
    'obstacle3': [(10.0, 6.5), (11.9, 6.5), (11.9, 11.3), (10.0, 11.3), (10.0, 6.5)],
    'obstacle4': [(-0.02104651, 0.76134018), (15.13244069, 0.76134018), (15.13244069, 2.9), (-0.02104651, 2.9), (-0.02104651, 0.76134018)]
}
obstacles_meter_zara02 = {
    'obstacle1': [(-0.35779069, 9.0), (-0.35779069, 14.94274416), (9.5, 14.94274416), (9.5, 9.0), (-0.35779069, 9.0)],
    'obstacle2': [(10.0, 11.2), (10.0, 14.94274416), (12.0, 14.94274416), (12.0, 11.2), (10.0, 11.2)],
    'obstacle3': [(10.0, 6.5), (11.9, 6.5), (11.9, 11.3), (10.0, 11.3), (10.0, 6.5)],
    'obstacle4': [(-0.35779069, 0.72625721), (15.55842276, 0.72625721), (15.55842276, 2.9), (-0.35779069, 2.9), (-0.35779069, 0.72625721)]
}
obstacles_meter_uni = {
    'obstacle1': [(3.8, 13.85420137), (6.0, 11.5), (10.0, 13.85420137), (3.8, 13.85420137)],
    'obstacle2': [(-0.17468604, 9.8), (-0.17468604, 11.5), (4.0, 11.5), (3.5, 9.8), (-0.17468604, 9.8)]
}

###########################################
# 2. lobby3 장애물 정보
###########################################
obstacles_meter_lobby3 = {
    'left_top': [(-2.28, 1.19), (-10, 1.19), (-10, 5), (-2.28, 5)],
    'left_bottom': [(-2.59, -8.95), (-10, -8.95), (-10, -15), (-2.59, -15)],
    'right_top': [(-0.23, 1.17), (10, 1.17), (10, 5), (-0.23, 5)],
    'right_bottom': [(8.13, -8.95), (10, -8.95), (10, -15), (8.13, -15)],
    'bottom': [(-2.59, -8.95), (-2.59, -15), (8.13, -15), (8.13, -8.95)],  # 연결된 하단 벽
    'left': [(-10, -0.6), (-2.59, -0.6), (-2.59, -8.95), (-10, -8.95)],  # left_bottom과 연결
    'right': [(8.13, -8.95), (8.13, -0.6), (10, -0.6), (10, -8.95)],
    'elevator': [(-2.30, 3.92), (-0.26, 3.92), (-0.26, 5), (-2.30, 5)],
    'entrance': [(0.63, -8.86), (0.63, -6.54), (5.03, -6.54), (5.03, -8.86)],
    'middle_obstacle': [(2.27, -2.00), (3.26, -2.00), (3.29, -4.29), (2.21, -4.34)],
    'pillar_left': [(-0.26, -0.55), (-0.54, -0.73), (-0.67, -1.11), (-0.52, -1.44), (-0.23, -1.59),
                    (0.06, -1.37), (0.20, -1.07), (0.03, -0.75)],
    'pillar_right': [(5.80, -0.68), (5.56, -0.77), (5.38, -1.09), (5.48, -1.41),
                     (5.79, -1.55), (6.06, -1.39), (6.18, -1.16), (6.04, -0.86)]
}
default_obstacle_polygons_lobby3 = [Polygon(coords) for coords in obstacles_meter_lobby3.values()]
def get_12dim_obstacle_vector(x0, y0, radius=1.0, polygons=None):
    if polygons is None:
        polygons = default_obstacle_polygons

    angles = range(0, 360, 30)  # 12방향
    vector = []
    origin = Point(x0, y0)

    for deg in angles:
        rad = math.radians(deg - 90)  # '위쪽(북쪽)' = deg=90
        x_end = x0 + radius * math.cos(rad)
        y_end = y0 + radius * math.sin(rad)
        ray = LineString([(x0, y0), (x_end, y_end)])

        min_distance = None
        for poly in polygons:
            if ray.intersects(poly):
                inter = ray.intersection(poly)
                # 교차점이 여러 개일 수 있으니 origin~가장 가까운 교차점 거리만 확인
                if inter.geom_type == 'Point':
                    dist = origin.distance(inter)
                else:
                    # LineString, MultiPoint 등
                    dist = min(origin.distance(Point(p)) for p in inter.coords)
                if min_distance is None or dist < min_distance:
                    min_distance = dist

        if min_distance is not None:
            vector.append(radius - min_distance)  
        else:
            # 교차점 없다면 radius 거리만큼 확보
            vector.append(radius)
    return vector


##########################################################
# PyTorch 기반 간단한 인코더
##########################################################
class GeoEncoder(nn.Module):
    def __init__(self, input_dim=12, encoded_dim=4):
        super(GeoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoded_dim)
        )

    def forward(self, x):
        return self.encoder(x)


##########################################################
# KoopmanPredictor (Geometry + Encoder)
##########################################################
class KoopmanPredictor:
    def __init__(self,
                 prediction_len=12,
                 data_dir='lobby2/biwi_eth',
                 min_samples=20,
                 dt=0.1,
                 pattern=r'^.*\d{2}\.npy$',
                 model_file='/home/core/Documents/MPC20250312/koopman/koopman_model_encoder_geo.pkl'):

        self._prediction_len = prediction_len
        self._min_samples = min_samples
        self._dt = dt
        self.pattern = pattern

        if os.path.exists(model_file):
            print(f"[INFO] Loading Koopman model from {model_file}")
            self._load_model(model_file)
        else:
            print("[INFO] No saved model found. Building Koopman model from training data...")

            # 기본 obstacle polygons
            self.obstacle_polygons = default_obstacle_polygons
            self.encoder = GeoEncoder()
            # 훈련 데이터 로드 → 학습
            all_past, all_future = self._load_training_data(data_dir)
            self._train_koopman(all_past, all_future)

            # 모델 저장
            self._save_model(model_file)
            print(f"[INFO] Koopman model saved to {model_file}")

    def _train_koopman(self, all_past, all_future):
        """
        all_past: (16, N), all_future: (16, N)
        - 이 데이터를 이용해 증강 상태(geometry encoder 포함) 후 Koopman operator K 학습
        """
        N = all_past.shape[1]
        augmented_past_list = []
        augmented_future_list = []

        for i in range(N):
            past_state = all_past[:, i]     # shape (16,)
            future_state = all_future[:, i] # shape (16,)
            augmented_past_list.append(self._augment_state(past_state))
            augmented_future_list.append(self._augment_state(future_state))

        augmented_past = np.column_stack(augmented_past_list)   # (20, N)
        augmented_future = np.column_stack(augmented_future_list)  # (20, N)

        # observables
        psi_past_list = []
        psi_future_list = []

        for k in range(N):
            psi_past_list.append(self._compute_observables(augmented_past[:, k]))
            psi_future_list.append(self._compute_observables(augmented_future[:, k]))

        psi_past = np.column_stack(psi_past_list)    # (20, N)
        psi_future = np.column_stack(psi_future_list)  # (20, N)

        # Koopman operator K 구하기
        try:
            pseudo_inv = np.linalg.pinv(psi_past)
            self.K = psi_future @ pseudo_inv  # (20, 20)
        except np.linalg.LinAlgError:
            self.K = np.eye(self._observable_dim())

    def _augment_state(self, state):
        """
        state(16차원) -> 앞 2차원(최신 frame x,y)을 사용해 12차원 geometry vector 계산
        → GeoEncoder로 4차원 축소 → 최종 16 + 4 = 20차원으로 augment
        """
        # 최신 프레임이 state[:2] (코드 구조상 [0]이 x, [1]이 y)
        x, y = state[0], state[1]
        # 12차원 raw obstacle vector
        geo_raw = np.array(get_12dim_obstacle_vector(x, y, radius=3.0, polygons=self.obstacle_polygons))
        # GeoEncoder 사용
        geo_tensor = torch.tensor(geo_raw, dtype=torch.float32).unsqueeze(0)  # (1,12)
        with torch.no_grad():
            encoded_geo = self.encoder(geo_tensor).numpy().squeeze()  # (4,)

        return np.concatenate([state, encoded_geo])

    # =============== Prediction / Forward ===============
    def __call__(self, tracking_result):
        """
        tracking_result: {object_id: numpy array of shape (T, 2)}
        """
        prediction_result = {}
        required_history = 8

        for object_id, history in tracking_result.items():
            if not isinstance(history, np.ndarray):
                history = np.array(history)

            if len(history) < required_history:
                # 관측 길이 부족 → 마지막 위치를 계속 유지
                xy0 = history[-1] if len(history) > 0 else np.array([0.0, 0.0])
                p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                prediction_result[object_id] = p
                continue

            past_frames = history[-required_history:]
            base_state = past_frames[::-1].flatten()  # (16,) 최신 프레임이 앞쪽
            # 20차원 증강
            state = self._augment_state(base_state)

            ps = []
            for _ in range(self._prediction_len):
                state = self._forward(state)
                ps.append(state[:2])  # 2차원만 기록

            prediction_result[object_id] = np.array(ps)

        return prediction_result

    def _forward(self, state):
        """
        - state: (20,)
        - state[:16]: 원래 16차원, state[16:]는 인코딩된 4차원
        - K @ state -> next_state
        - 다시 geometry 인코딩으로 20차원으로 갱신
        """
        o = self._compute_observables(state)
        o_next = self.K @ o
        # next_state[:16]만 떼서 장애물 벡터 구한 뒤, 다시 인코딩
        next_state = self._to_state(o_next)
        x, y = next_state[0], next_state[1]

        geo_raw = np.array(get_12dim_obstacle_vector(x, y, radius=2.0, polygons=self.obstacle_polygons))
        geo_tensor = torch.tensor(geo_raw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            encoded_geo = self.encoder(geo_tensor).numpy().squeeze()

        new_state = np.concatenate([next_state[:16], encoded_geo])  # 최종 20차원
        return new_state

    def _compute_observables(self, state):
        # 여기서는 상태 자체(20차원)를 그대로 observable로 사용
        return state

    def _observable_dim(self):
        return 20

    def _to_state(self, observables):
        return observables[:20]

    # =============== Loading Training Data ===============
    def _load_training_data(self, data_dir):
        """
        data_dir/train 경로의 npy 파일을 훑어 (8프레임 past, 8프레임 future) 형태로 학습용 데이터 구성
        """
        from shapely.geometry import Polygon

        train_dir = os.path.join(data_dir, 'train')
        pattern = re.compile(r'^.*\.npy$')
        all_past = []
        all_future = []
        required_history = 8

        # 예시: lobby2인 경우 특정 파일만, lobby3인 경우 다른 polygon 설정
        valid_files = None
        if "lobby2" in data_dir:
            valid_files = {"crowds_zara01_train.npy", "crowds_zara02_train.npy", "uni_examples_train.npy"}
        elif "lobby3" in data_dir:
            self.obstacle_polygons = default_obstacle_polygons_lobby3

        for fname in os.listdir(train_dir):
            if pattern.match(fname):
                # lobby2인 경우 file whitelist
                if valid_files is not None and fname not in valid_files:
                    continue
                filepath = os.path.join(train_dir, fname)
                # 파일별로 polygon 교체 (lobby2 예시)
                if valid_files is not None:
                    if fname == "crowds_zara01_train.npy":
                        obs_dict = obstacles_meter_zara01
                    elif fname == "crowds_zara02_train.npy":
                        obs_dict = obstacles_meter_zara02
                    elif fname == "uni_examples_train.npy":
                        obs_dict = obstacles_meter_uni
                    else:
                        continue
                    self.obstacle_polygons = [Polygon(coords) for coords in obs_dict.values()]

                data = np.load(filepath)  # (T, num_agents, 2)
                T, num_agents, _ = data.shape

                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < (required_history + 1):
                        continue
                    for i in range(len(valid_idx) - required_history):
                        past_indices = valid_idx[i : i + required_history]
                        future_indices = valid_idx[i + 1 : i + 1 + required_history]
                        past_window = agent_xy[past_indices]   # (8,2)
                        future_window = agent_xy[future_indices] # (8,2)

                        past_state = past_window[::-1].flatten()    # (16,), 최신이 앞쪽
                        future_state = future_window[::-1].flatten()

                        all_past.append(past_state)
                        all_future.append(future_state)

        if len(all_past) == 0:
            raise ValueError(f"No valid training data found in {train_dir}.")

        all_past = np.array(all_past).T   # (16, N)
        all_future = np.array(all_future).T  # (16, N)
        return all_past, all_future

    # =======================
    # 테스트 / 평가 함수들
    # =======================
    def testtraj(self, test_dir):
        """
        기존 방식의 testtraj
        """
        pattern = re.compile(self.pattern)
        goal = []
        future = []
        prediction_len = 12
        history_len = 8

        for fname in os.listdir(test_dir):
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                T, num_agents, _ = data.shape

                for agent_id in range(num_agents):
                    all_ade = []
                    all_fde = []
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]

                    if len(valid_idx) < 20:
                        for _ in range(len(agent_xy)):
                            gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    else:
                        for _ in range(valid_idx[0] + history_len - 1):
                            gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                        for start_idx in range(len(valid_idx) - history_len):
                            history_idx = valid_idx[start_idx : start_idx + history_len]
                            future_idx = valid_idx[start_idx + history_len : start_idx + history_len + prediction_len]

                            if len(future_idx) < prediction_len:
                                history = agent_xy[history_idx]
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]
                                pad_size = prediction_len - len(future_idx)
                                pad = [[np.nan, np.nan] for _ in range(pad_size)]
                                gt_future = np.concatenate([agent_xy[future_idx], pad])
                                all_ade.append(gt_future)
                                all_fde.append(pred_future)
                            else:
                                history = agent_xy[history_idx]
                                gt_future = agent_xy[future_idx]
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]
                                all_ade.append(gt_future)
                                all_fde.append(pred_future)
                                if pred_future.shape[0] != prediction_len:
                                    print(f"[Warning] 예측 길이 불일치: {pred_future.shape[0]}개")
                                    continue
                        for _ in range(valid_idx[-1], len(agent_xy)):
                            gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)

                    if len(goal):
                        all_fde = np.reshape(all_fde, (len(agent_xy), prediction_len, 1, 2))
                        all_ade = np.reshape(all_ade, (len(agent_xy), prediction_len, 1, 2))
                        future = np.concatenate([future, all_fde], axis=2)
                        goal = np.concatenate([goal, all_ade], axis=2)
                    else:
                        future = np.reshape(all_fde, (len(agent_xy), prediction_len, 1, 2))
                        goal = np.reshape(all_ade, (len(agent_xy), prediction_len, 1, 2))

                print(len(agent_xy))
                print(goal.shape)
                print(future.shape)
        return goal, future

    def testtraj2(self, test_dir):
        """
        기존 방식의 testtraj2
        """
        pattern = re.compile(self.pattern)
        goal = None
        future = None
        prediction_len = 12
        history_len = 8

        for fname in os.listdir(test_dir):
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                T, num_agents, _ = data.shape

                all_ade_list = []
                all_fde_list = []

                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    all_ade = []
                    all_fde = []

                    if len(valid_idx) < 20 or not np.all(np.diff(valid_idx) == 1):
                        for _ in range(len(agent_xy)):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    else:
                        for _ in range(valid_idx[0] + history_len - 1):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)

                        for start_idx in range(len(valid_idx) - history_len):
                            history_idx = valid_idx[start_idx : start_idx + history_len]
                            future_idx = valid_idx[start_idx + history_len : start_idx + history_len + prediction_len]

                            history = agent_xy[history_idx]
                            if len(future_idx) < prediction_len:
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]
                                pad_size = prediction_len - len(future_idx)
                                pad = np.full((pad_size, 2), np.nan)
                                gt_future = np.vstack([agent_xy[future_idx], pad])
                            else:
                                gt_future = agent_xy[future_idx]
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]
                                if pred_future.shape[0] != prediction_len:
                                    print(f"[Warning] 예측 길이 불일치: {pred_future.shape[0]}개")
                                    continue
                            all_ade.append(gt_future)
                            all_fde.append(pred_future)

                        for _ in range(valid_idx[-1], len(agent_xy)):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)

                    all_ade_list.append(np.array(all_ade))
                    all_fde_list.append(np.array(all_fde))

                all_ade_np = np.stack(all_ade_list, axis=2)
                all_fde_np = np.stack(all_fde_list, axis=2)

                if goal is None:
                    goal = all_ade_np
                    future = all_fde_np
                else:
                    goal = np.concatenate([goal, all_ade_np], axis=2)
                    future = np.concatenate([future, all_fde_np], axis=2)

                print(len(agent_xy))
                print(goal.shape)
                print(future.shape)

        return goal, future

    @staticmethod
    def pad_to_length(arr, target_length):
        """
        arr의 axis=0 길이를 target_length로 맞춰 NaN 패딩
        """
        current_length = arr.shape[0]
        if current_length < target_length:
            pad_width = ((0, target_length - current_length), (0, 0), (0, 0), (0, 0))
            return np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
        return arr

    def evaluate_test(self, test_dir):
        """
        간단한 ADE / FDE 측정
        """
        pattern = re.compile(r'^.*\.npy$')
        all_ade = []
        all_fde = []

        history_len = 8
        prediction_len = 12
        required_length = history_len + prediction_len

        for fname in os.listdir(test_dir):
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                T, num_agents, _ = data.shape

                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]

                    if len(valid_idx) < required_length:
                        continue

                    for start_idx in range(len(valid_idx) - required_length + 1):
                        history_idx = valid_idx[start_idx : start_idx + history_len]
                        future_idx = valid_idx[start_idx + history_len : start_idx + required_length]
                        if len(history_idx) < history_len or len(future_idx) < prediction_len:
                            continue
                        history = agent_xy[history_idx]
                        gt_future = agent_xy[future_idx]
                        pred_dict = self({agent_id: history})
                        pred_future = pred_dict[agent_id][:prediction_len]
                        if pred_future.shape[0] != prediction_len:
                            print(f"[Warning] 예측 길이 불일치: {pred_future.shape[0]}개")
                            continue

                        errors = np.linalg.norm(pred_future - gt_future, axis=1)
                        all_ade.append(errors.mean())
                        all_fde.append(errors[-1])

        if len(all_ade) > 0:
            ade = np.mean(all_ade)
            fde = np.mean(all_fde)
            print(f"✅ Average Distance Error (ADE): {ade:.4f}")
            print(f"✅ Final Distance Error (FDE): {fde:.4f}")
        else:
            print("❌ No valid test data found.")

    # ======================================
    # 모델 저장 / 불러오기
    # ======================================
    def _save_model(self, filename):
        """
        - self.K (numpy array)
        - self.encoder (PyTorch state_dict)
        - self.obstacle_polygons
        등을 pickle로 저장
        """
        model_dict = {
            'K': self.K,
            'encoder_state_dict': self.encoder.state_dict(),
            'obstacle_polygons': self.obstacle_polygons,
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)

    def _load_model(self, filename):
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
        # K
        self.K = model_dict['K']
        # encoder
        self.encoder = GeoEncoder()
        self.encoder.load_state_dict(model_dict['encoder_state_dict'])
        # obstacle_polygons
        self.obstacle_polygons = model_dict['obstacle_polygons']
