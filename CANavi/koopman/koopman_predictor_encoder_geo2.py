import os
import re
import math
import pickle
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. 장애물 정의 (모두 그대로 사용)
# =============================================================================
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
obstacles_meter_eth = {
    'obstacle1': [(14.41, -6.17), (14.41, -1.6), (-0.5, -1.0), (-0.7, -6.17), (14.41, -6.17)],
    'obstacle2': [(-1.6, 12.8), (-1.8, 16.4), (14.41, 16.4), (14.41, 15.8), (-1.6, 12.8)]
}
obstacles_meter_hotel = {
    'obstacle1': [(5.4, -10.31), (4.4, -10.31), (4.1, 4.31), (5.4, 4.31), (5.4, -10.31)],
    'obstacle2': [(-2.5, -10.31), (-0.8, -10.31), (-0.8, 4.31), (-2.5, 4.31), (-2.5, -10.31)]
}
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
obstacles_meter_lobby3 = {
    'left_top': [(-2.28, 1.19), (-10, 1.19), (-10, 5), (-2.28, 5)],
    'left_bottom': [(-2.59, -8.95), (-10, -8.95), (-10, -15), (-2.59, -15)],
    'right_top': [(-0.23, 1.17), (10, 1.17), (10, 5), (-0.23, 5)],
    'right_bottom': [(8.13, -8.95), (10, -8.95), (10, -15), (8.13, -15)],
    'bottom': [(-2.59, -8.95), (-2.59, -15), (8.13, -15), (8.13, -8.95)],
    'left': [(-10, -0.6), (-2.59, -0.6), (-2.59, -8.95), (-10, -8.95)],
    'right': [(8.13, -8.95), (8.13, -0.6), (10, -0.6), (10, -8.95)],
    'elevator': [(-2.30, 3.92), (-0.26, 3.92), (-0.26, 5), (-2.30, 5)],
    'entrance': [(0.63, -8.86), (0.63, -6.54), (5.03, -6.54), (5.03, -8.86)],
    'middle_obstacle': [(2.27, -2.00), (3.26, -2.00), (3.29, -4.29), (2.21, -4.34)],
    'pillar_left': [(-0.26, -0.55), (-0.54, -0.73), (-0.67, -1.11), (-0.52, -1.44), (-0.23, -1.59),
                    (0.06, -1.37), (0.20, -1.07), (0.03, -0.75)],
    'pillar_right': [(5.80, -0.68), (5.56, -0.77), (5.38, -1.09), (5.48, -1.41),
                     (5.79, -1.55), (6.06, -1.39), (6.18, -1.16), (6.04, -0.86)]
}

# =============================================================================
# 2. 12방향 레이캐스트 -> 거리 벡터 (길이 12) 계산
# =============================================================================
def get_12dim_obstacle_vector(x0, y0, radius=1.0, polygons=None):
    """
    x0, y0에서 30도 간격으로 선분을 뻗어 장애물과 교차하는 최소 거리를 찾습니다.
    교차점이 없으면 radius 그대로, 있으면 min_distance를 벡터에 저장합니다.
    """
    if polygons is None:
        polygons = [Polygon(coords) for coords in default_obstacles_meter.values()]

    angles = range(0, 360, 30)
    vector = []
    origin = Point(x0, y0)

    for deg in angles:
        rad = math.radians(deg - 90)
        x_end = x0 + radius * math.cos(rad)
        y_end = y0 + radius * math.sin(rad)
        ray = LineString([(x0, y0), (x_end, y_end)])

        min_distance = None
        for poly in polygons:
            if ray.intersects(poly):
                inter = ray.intersection(poly)
                if inter.geom_type == 'Point':
                    dist = origin.distance(inter)
                else:
                    dist = min(origin.distance(Point(p)) for p in inter.coords)
                if min_distance is None or dist < min_distance:
                    min_distance = dist

        # 교차점 존재하면 min_distance, 없으면 radius
        if min_distance is not None:
            vector.append(min_distance)
        else:
            vector.append(radius)
    return vector

# =============================================================================
# 3. PyTorch 기반 12->4 인코더
# =============================================================================
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

# =============================================================================
# 4. KoopmanPredictor: train(...)에서 GeoEncoder+K 동시학습
# =============================================================================
class KoopmanPredictor:
    def __init__(self,
                 prediction_len=12,
                 data_dir='lobby2/biwi_eth',
                 min_samples=20,
                 dt=0.1,
                 pattern=r'^.*\d{2}\.npy$',
                 model_file='/home/core/Documents/MPC20250312/koopman/koopman_model_encoder_geo2.pkl'):
        self._prediction_len = prediction_len
        self._min_samples = min_samples
        self._dt = dt
        self.pattern = pattern

        # 기본 폴리곤: lobby3
        self.obstacle_polygons = [Polygon(coords) for coords in obstacles_meter_lobby3.values()]

        # Encoder 및 K 초기화
        self.encoder = GeoEncoder()
        self.K = None  # shape (20, 20) 예정

        # model_file이 존재하면 로드
        if os.path.exists(model_file):
            print(f"[INFO] Loading Koopman model from {model_file}")
            self._load_model(model_file)
        else:
            print("[INFO] No saved model found. You can call train(...) to build a new model.")

    def _compute_observables(self, state):
        return state.flatten()

    def _observable_dim(self):
        return 20  # 16 + 4

    def _augment_state(self, state):
        """
        예측 시 사용: numpy 기반
        state: (16,)
        """
        x, y = state[0], state[1]
        # 12차원 장애물 벡터
        geo_raw = np.array(get_12dim_obstacle_vector(x, y, radius=3.0, polygons=self.obstacle_polygons))
        # Encoder
        geo_tensor = torch.tensor(geo_raw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            encoded_geo = self.encoder(geo_tensor).numpy().squeeze()
        return np.concatenate([state, encoded_geo])  # (20,)

    # --------------------------------------------------------------------------
    # 예측 관련
    # --------------------------------------------------------------------------
    def __call__(self, tracking_result):
        prediction_result = {}
        required_history = 8
        for object_id, history in tracking_result.items():
            if not isinstance(history, np.ndarray):
                history = np.array(history)
            if len(history) < required_history:
                # 관측 부족 -> 마지막 위치 유지
                xy0 = history[-1] if len(history) > 0 else np.array([0., 0.])
                p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                prediction_result[object_id] = p
                continue

            past_frames = history[-required_history:]
            base_state = past_frames[::-1].flatten()  # (16,)
            state = self._augment_state(base_state)

            ps = []
            for _ in range(self._prediction_len):
                state = self._forward(state)
                ps.append(state[:2])
            prediction_result[object_id] = np.array(ps)
        return prediction_result

    def _forward(self, state):
        # Koopman으로 next state 추정
        o = self._compute_observables(state)
        o_next = self.K @ o
        next_state = self._to_state(o_next)

        # 다시 GeoEncoder로 20차원 증강
        x, y = next_state[0], next_state[1]
        geo_raw = np.array(get_12dim_obstacle_vector(x, y, radius=2.0, polygons=self.obstacle_polygons))
        geo_tensor = torch.tensor(geo_raw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            encoded_geo = self.encoder(geo_tensor).numpy().squeeze()

        new_state = np.concatenate([next_state[:16], encoded_geo])
        return new_state

    def _to_state(self, observables):
        return observables[:20]

    # --------------------------------------------------------------------------
    # 학습: train(...)
    # --------------------------------------------------------------------------
    def train(self, num_epochs=30, lr=1e-3, data_dir="lobby2/biwi_eth"):
        """
        1) (past, future) 로딩
        2) encoder 학습 + K 업데이트
        3) 최종 K를 self.K에 반영
        """
        all_past, all_future = self._load_training_data(data_dir)
        past_tensor = torch.tensor(all_past.T, dtype=torch.float32)    # (N,16)
        future_tensor = torch.tensor(all_future.T, dtype=torch.float32) # (N,16)
        num_samples = past_tensor.shape[0]

        optimizer = optim.Adam(self.encoder.parameters(), lr=lr)

        # 미리 장애물 벡터 계산
        print("[TRAIN] Pre-calculating obstacle vectors...")
        cached_past_geo = []
        cached_future_geo = []
        for i in range(num_samples):
            px, py = past_tensor[i][0].item(), past_tensor[i][1].item()
            fx, fy = future_tensor[i][0].item(), future_tensor[i][1].item()

            cached_past_geo.append(get_12dim_obstacle_vector(px, py, radius=3.0, polygons=self.obstacle_polygons))
            cached_future_geo.append(get_12dim_obstacle_vector(fx, fy, radius=3.0, polygons=self.obstacle_polygons))
        print("[TRAIN] Done. Start training...")

        for epoch in range(num_epochs):
            psi_past_list = []
            psi_future_list = []

            for i in range(num_samples):
                base_past = past_tensor[i]
                base_future = future_tensor[i]

                # 주의: 학습 시 encoder에 gradient가 흘러야 하므로 no_grad 사용 안 함
                gp = torch.tensor(cached_past_geo[i], dtype=torch.float32)
                gf = torch.tensor(cached_future_geo[i], dtype=torch.float32)
                aug_past = torch.cat([base_past, self.encoder(gp.unsqueeze(0)).squeeze(0)], dim=0)
                aug_future = torch.cat([base_future, self.encoder(gf.unsqueeze(0)).squeeze(0)], dim=0)

                psi_past_list.append(aug_past)
                psi_future_list.append(aug_future)

            psi_past = torch.stack(psi_past_list, dim=0)   # (N,20)
            psi_future = torch.stack(psi_future_list, dim=0) # (N,20)

            # Koopman operator K 추정
            psi_past_pin = torch.pinverse(psi_past)
            K_est = (psi_past_pin @ psi_future).T  # (20,20)

            # 예측된 future
            psi_future_pred = psi_past @ K_est.T
            loss = ((psi_future - psi_future_pred)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}")

        # 최종 K 추정(업데이트된 encoder로 다시 계산)
        with torch.no_grad():
            psi_past_list = []
            psi_future_list = []
            for i in range(num_samples):
                base_past = past_tensor[i]
                base_future = future_tensor[i]
                gp = torch.tensor(cached_past_geo[i], dtype=torch.float32)
                gf = torch.tensor(cached_future_geo[i], dtype=torch.float32)
                aug_past = torch.cat([base_past, self.encoder(gp.unsqueeze(0)).squeeze(0)], dim=0)
                aug_future = torch.cat([base_future, self.encoder(gf.unsqueeze(0)).squeeze(0)], dim=0)
                psi_past_list.append(aug_past)
                psi_future_list.append(aug_future)
            psi_past = torch.stack(psi_past_list, dim=0)
            psi_future = torch.stack(psi_future_list, dim=0)
            psi_past_pin = torch.pinverse(psi_past)
            K_final = (psi_past_pin @ psi_future).T

        self.K = K_final.cpu().numpy()
        print("[TRAIN] Final K updated.")

    # --------------------------------------------------------------------------
    # 학습 데이터 로딩
    # --------------------------------------------------------------------------
    def _load_training_data(self, data_dir):
        train_dir = os.path.join(data_dir, 'train')
        pattern = re.compile(r'^.*\.npy$')
        all_past = []
        all_future = []
        required_history = 8

        valid_files = None
        if "lobby2" in data_dir:
            valid_files = {
                "crowds_eth_train.npy", "crowds_hotel_train.npy",
                "crowds_zara01_train.npy", "crowds_zara02_train.npy", "uni_examples_train.npy"
            }
        elif "lobby3" in data_dir:
            self.obstacle_polygons = [Polygon(coords) for coords in obstacles_meter_lobby3.values()]

        for fname in os.listdir(train_dir):
            if pattern.match(fname):
                if valid_files is not None and fname not in valid_files:
                    continue
                filepath = os.path.join(train_dir, fname)
                # 파일별 장애물
                if valid_files is not None:
                    if fname == "crowds_eth_train.npy":
                        obstacles_meter = obstacles_meter_eth
                    elif fname == "crowds_hotel_train.npy":
                        obstacles_meter = obstacles_meter_hotel
                    elif fname == "crowds_zara01_train.npy":
                        obstacles_meter = obstacles_meter_zara01
                    elif fname == "crowds_zara02_train.npy":
                        obstacles_meter = obstacles_meter_zara02
                    elif fname == "uni_examples_train.npy":
                        obstacles_meter = obstacles_meter_uni
                    else:
                        continue
                    self.obstacle_polygons = [Polygon(coords) for coords in obstacles_meter.values()]

                data = np.load(filepath)
                T, num_agents, _ = data.shape
                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < (required_history + 1):
                        continue
                    for i in range(len(valid_idx) - required_history):
                        past_indices = valid_idx[i : i + required_history]
                        future_indices = valid_idx[i + 1 : i + 1 + required_history]
                        past_window = agent_xy[past_indices]
                        future_window = agent_xy[future_indices]
                        past_state = past_window[::-1].flatten()
                        future_state = future_window[::-1].flatten()
                        all_past.append(past_state)
                        all_future.append(future_state)
        if len(all_past) == 0:
            raise ValueError(f"No valid training data found in {train_dir}.")
        all_past = np.array(all_past).T
        all_future = np.array(all_future).T
        return all_past, all_future

    # --------------------------------------------------------------------------
    # 모델 저장 / 로드
    # --------------------------------------------------------------------------
    def save_model(self, filename):
        self._save_model(filename)

    def _save_model(self, filename):
        model_dict = {
            'K': self.K,
            'encoder_state': self.encoder.state_dict(),
            'obstacle_polygons': self.obstacle_polygons
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"[INFO] Model saved to {filename}.")

    def _load_model(self, filename):
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
        self.K = model_dict['K']
        self.encoder.load_state_dict(model_dict['encoder_state'])
        self.obstacle_polygons = model_dict['obstacle_polygons']
        print(f"[INFO] Model loaded from {filename}.")

    # --------------------------------------------------------------------------
    # 테스트 / 평가 함수들
    # --------------------------------------------------------------------------
    def testtraj(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = []
        future = []
        prediction_len = 12
        history_len = 8
        for fname in os.listdir(test_dir):
            goal = []
            future = []
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                name, _ = os.path.splitext(filepath)
                T, num_agents, _ = data.shape
                filepath_to_save_predictions = name + '_koopman_predictions.npy'

                for agent_id in range(num_agents):
                    all_ade = []
                    all_fde = []
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]

                    if len(valid_idx) < 20:
                        for start_idx in range(len(agent_xy)):
                            gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    else:
                        for start_idx in range(valid_idx[0] + history_len - 1):
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
                                all_fde.append(pred_future)
                                pad_size = prediction_len - len(future_idx)
                                pad = [[np.nan, np.nan] for _ in range(pad_size)]
                                gt_future = np.concatenate([agent_xy[future_idx], pad])
                                all_ade.append(gt_future)
                            else:
                                history = agent_xy[history_idx]
                                gt_future = agent_xy[future_idx]
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]
                                all_ade.append(gt_future)
                                all_fde.append(pred_future)
                                if pred_future.shape[0] != prediction_len:
                                    print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
                                    continue

                        for start_idx in range(valid_idx[-1], len(agent_xy)):
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
                np.save(filepath_to_save_predictions, future)

        return goal, future

    def testtraj2(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = None
        future = None
        prediction_len = 12
        history_len = 8
        for fname in os.listdir(test_dir):
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                name, _ = os.path.splitext(filepath)
                T, num_agents, _ = data.shape

                all_ade_list = []
                all_fde_list = []

                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]

                    all_ade = []
                    all_fde = []

                    if len(valid_idx) < 20 or not np.all(np.diff(valid_idx) == 1):
                        for start_idx in range(len(agent_xy)):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    else:
                        for start_idx in range(valid_idx[0] + history_len - 1):
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
                                    print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
                                    continue
                            all_ade.append(gt_future)
                            all_fde.append(pred_future)

                        for start_idx in range(valid_idx[-1], len(agent_xy)):
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

    def evaluate_test(self, test_dir):
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
                            print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
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

    @staticmethod
    def pad_to_length(arr, target_length):
        current_length = arr.shape[0]
        if current_length < target_length:
            pad_width = ((0, target_length - current_length), (0, 0), (0, 0), (0, 0))
            return np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
        return arr
