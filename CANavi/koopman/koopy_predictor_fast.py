import os
import re
import math
import pickle
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, LineString, Point
import torch
import time
from scipy.spatial import ConvexHull
import torch.nn as nn

def vertices2halfspaces(vertices):
    """
    Convert a 2D convex polytope from V-representation (vertices) to H-representation (inequalities).
    """
    verts = np.array(vertices)
    hull = ConvexHull(verts)
    hull_verts = verts[hull.vertices]
    inequalities = []
    for i in range(len(hull_verts)):
        p1 = hull_verts[i]
        p2 = hull_verts[(i + 1) % len(hull_verts)]
        edge = p1 - p2
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        c = normal.dot(p1)
        inequalities.append([normal[0], normal[1], c])
    return np.array(inequalities)


def cast_rays(pose3d, obstacles, max_distance=5., n_rays=100):
    """
    pose3d: array (x, y, theta)
    obstacles: list of halfspace arrays (m,3) for Ax <= b
    returns: distances per ray
    """
    ori = pose3d[-1]
    th = np.radians(np.arange(0, 360, 360 / n_rays)) - np.pi/2
    ray_dirs = np.stack([np.cos(th), np.sin(th)], axis=0)
    pos = pose3d[:2]
    if pos.ndim == 1:
        pos = pos[:, None]
    distances = np.zeros((len(obstacles), n_rays))
    for i, hs in enumerate(obstacles):
        A = hs[:, :-1]
        b = hs[:, -1][:, None]
        r = b - A @ pos
        dot = A @ ray_dirs
        positive = dot > 0
        negative = dot < 0
        outside = r < 0
        inside = ~outside
        ub = np.where(positive & inside, r / dot, np.inf)
        lb = np.where(negative, r / dot, 0.)
        lb = np.clip(lb, 0., np.inf)
        max_lb = lb.max(axis=0)
        min_ub = ub.min(axis=0)
        feasible = max_lb <= min_ub
        distances[i, feasible] = np.clip(max_lb[feasible], -np.inf, max_distance)
        distances[i, ~feasible] = max_distance
        empty = np.any(outside & (~negative), axis=0)
        distances[i, empty] = max_distance
    return distances.min(axis=0)

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

default_obstacle_polygons_lobby2 = [Polygon(coords) for coords in obstacles_meter_zara02.values()]

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
default_obstacle_halfspaces_lobby3 = [vertices2halfspaces(coords) for coords in obstacles_meter_lobby3.values()]

def get_12dim_obstacle_vector(x0, y0, radius=2.0, halfspaces=None):
    """
    Fast 12-direction occupancy vector using ray casting on halfspaces.
    """
    if halfspaces is None:
        halfspaces = default_obstacle_halfspaces_lobby3
    pose = np.array([x0, y0, -90.0], dtype=float)
    dists = cast_rays(pose, halfspaces, max_distance=radius, n_rays=12)
    return (dists < radius).astype(int)


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

class KoopmanPredictor:
    def __init__(self, prediction_len=12, data_dir='lobby2/biwi_eth', min_samples=20, dt=0.4, 
                 pattern=r'^.*\d{2}\.npy$',model_file='/home/core/Documents/MPC20250312/koopman/koopy.pkl', train_on_init=True):
        self._prediction_len = prediction_len
        self._min_samples = min_samples
        self._dt = dt
        self.pattern = pattern
        self.encoder = GeoEncoder()
        self.K = None

        # cache obstacle halfspaces for inference
        self.obstacle_halfspaces = default_obstacle_halfspaces_lobby3.copy()

        if os.path.exists(model_file):
            self._load_model(model_file)
        else:
            print("[INFO] No saved model found.")

    def _set_polygons_for_file(self, fname):
        if "eth"    in fname: meter = obstacles_meter_eth
        elif "hotel" in fname: meter = obstacles_meter_hotel
        elif "zara01" in fname: meter = obstacles_meter_zara01
        elif "zara02" in fname: meter = obstacles_meter_zara02
        elif "uni"   in fname: meter = obstacles_meter_uni
        else:                meter = obstacles_meter_lobby3
        self.obstacle_halfspaces = [vertices2halfspaces(coords) for coords in meter.values()]

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
    def _compute_observables(self, state):
        state = state.flatten()
        return state

    def _observable_dim(self):
        return 20

    def _augment_state(self, state):
        base = state[:2]
        geo_raw = get_12dim_obstacle_vector(
            base[0], base[1], radius=2.0,
            halfspaces=self.obstacle_halfspaces  # MODIFIED
        )
        geo_t = torch.tensor(geo_raw, dtype=torch.float32).unsqueeze(0)
        enc = self.encoder(geo_t).detach().cpu().numpy().squeeze()
        return np.concatenate([state, enc])

    def _augment_state_train(self, state_tensor, cached_geo=None):
        base_state = state_tensor[:2]
        if cached_geo is None:
            base_state_np = base_state.detach().cpu().numpy()
            geo_raw = get_12dim_obstacle_vector(base_state_np[0], base_state_np[1], radius=2.0, polygons=self.obstacle_polygons)
        else:
            geo_raw = cached_geo
        geo_tensor = torch.tensor(geo_raw, dtype=torch.float32)
        encoded_geo = self.encoder(geo_tensor.unsqueeze(0)).squeeze(0)
        return torch.cat([state_tensor, encoded_geo], dim=0)

    def __call__(self, tracking_result):
        prediction_result = {}
        required_history = 8
        for object_id, history in tracking_result.items():
            if not isinstance(history, np.ndarray):
                history = np.array(history)
            if len(history) < required_history:
                xy0 = history[-1] if len(history) > 0 else np.array([0., 0.])
                p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                prediction_result[object_id] = p
                continue
            past_frames = history[-required_history:]
            base_state = past_frames[::-1].flatten()
            state = self._augment_state(base_state)
            ps = []
            for _ in range(self._prediction_len):
                state = self._forward(state)
                ps.append(state[:2])
            prediction_result[object_id] = np.array(ps)
        return prediction_result

    def _forward(self, state):
        o = self._compute_observables(state)
        o_next = self.K @ o
        next_state = self._to_state(o_next)
        base = next_state[:2]
        geo_raw = get_12dim_obstacle_vector(
            base[0], base[1], radius=2.0,
            halfspaces=self.obstacle_halfspaces  # MODIFIED
        )
        geo_t = torch.tensor(geo_raw, dtype=torch.float32).unsqueeze(0)
        enc = self.encoder(geo_t).detach().cpu().numpy().squeeze()
        return np.concatenate([next_state[:16], enc])

    def _to_state(self, observables):
        return observables[:20]

    def _load_training_data(self, data_dir):
        # train_dir = os.path.join(data_dir, 'train')
        train_dir = data_dir
        pattern = re.compile(r'^.*\.npy$')

        all_past, all_future = [], []
        past_polygons_list, future_polygons_list = [], []
        required_history = 8

        def pick_obstacles(fname):
            if "eth" in fname:          return obstacles_meter_eth
            if "hotel" in fname:        return obstacles_meter_hotel
            if "zara01" in fname:       return obstacles_meter_zara01
            if "zara02" in fname:       return obstacles_meter_zara02
            if "uni_examples" in fname: return obstacles_meter_uni
            return obstacles_meter_lobby3

        for fname in os.listdir(train_dir):
            if not pattern.match(fname):
                continue
            obs_dict = pick_obstacles(fname)
            polygons = [Polygon(coords) for coords in obs_dict.values()]

            data = np.load(os.path.join(train_dir, fname))  # (T, n_agent, 2)
            T, num_agents, _ = data.shape

            for agent_id in range(num_agents):
                traj = data[:, agent_id, :]
                valid = np.where(~np.isnan(traj).any(axis=1))[0]
                if len(valid) < required_history + 1:
                    continue

                for i in range(len(valid) - required_history):
                    past_idx   = valid[i : i + required_history]
                    future_idx = valid[i + 1 : i + 1 + required_history]

                    past_state   = traj[past_idx][::-1].flatten()   # shape (16,)
                    future_state = traj[future_idx][::-1].flatten()

                    all_past.append(past_state)
                    all_future.append(future_state)

                    # 이 샘플에 대응하는 장애물 리스트도 같이 저장
                    past_polygons_list.append(polygons)
                    future_polygons_list.append(polygons)

        if not all_past:
            raise ValueError(f"No valid training data in {train_dir}")

        # (16, N) 형태로 변환
        return (
            np.array(all_past).T,
            np.array(all_future).T,
            past_polygons_list,
            future_polygons_list
        )


        
    def train(self, num_epochs=30, lr=1e-3, data_dir =None):
        all_past, all_future, past_polys, future_polys = self._load_training_data(data_dir)
        past_tensor   = torch.tensor(all_past.T,   dtype=torch.float32)
        future_tensor = torch.tensor(all_future.T, dtype=torch.float32)
        N = past_tensor.shape[0]

        cached_past_geo = [
            get_12dim_obstacle_vector(
                past_tensor[i,0].item(),
                past_tensor[i,1].item(),
                polygons=past_polys[i]
            )
            for i in range(N)
        ]
        cached_fut_geo = [
            get_12dim_obstacle_vector(
                future_tensor[i,0].item(),
                future_tensor[i,1].item(),
                polygons=future_polys[i]
            )
            for i in range(N)
        ]

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        for epoch in range(num_epochs):
            psi_past_list   = []
            psi_future_list = []
            for i in range(N):
                psi_past_list.append(self._augment_state_train(past_tensor[i],   cached_geo=cached_past_geo[i]))
                psi_future_list.append(self._augment_state_train(future_tensor[i], cached_geo=cached_fut_geo[i]))
            psi_past   = torch.stack(psi_past_list,   dim=0)
            psi_future = torch.stack(psi_future_list, dim=0)

            K_tensor    = (torch.pinverse(psi_past) @ psi_future).T
            psi_pred    = psi_past @ K_tensor.T
            loss        = torch.mean((psi_future - psi_pred)**2)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if epoch==0 or (epoch+1)%10==0:
                print(f"Epoch {epoch+1}: Loss={loss.item():.6f}")

        with torch.no_grad():
            psi_past   = torch.stack([self._augment_state_train(past_tensor[i],   cached_geo=cached_past_geo[i])   for i in range(N)], dim=0)
            psi_future = torch.stack([self._augment_state_train(future_tensor[i], cached_geo=cached_fut_geo[i]) for i in range(N)], dim=0)
            K_tensor   = (torch.pinverse(psi_past) @ psi_future).T
            self.K     = K_tensor.cpu().numpy()
        print("최종 K 계산 완료.")


    def testtraj2(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = None
        future = None
        prediction_len = 12
        history_len = 8
        for fname in os.listdir(test_dir):
            self._set_polygons_for_file(fname)
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
                            history_idx = valid_idx[start_idx:start_idx + history_len]
                            future_idx = valid_idx[start_idx + history_len:start_idx + history_len + prediction_len]
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


    def _set_polygons_for_file(self, fname):
        if "eth"    in fname: meter = obstacles_meter_eth
        elif "hotel" in fname: meter = obstacles_meter_hotel
        elif "zara01" in fname: meter = obstacles_meter_zara01
        elif "zara02" in fname: meter = obstacles_meter_zara02
        elif "uni"   in fname: meter = obstacles_meter_uni
        else:                meter = obstacles_meter_lobby3
        self.obstacle_polygons = [Polygon(coords) for coords in meter.values()]

