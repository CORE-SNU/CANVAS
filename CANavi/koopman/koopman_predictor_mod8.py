import os
import re
import pickle
import numpy as np
from itertools import combinations

class KoopmanPredictor:
    def __init__(self,
                 prediction_len=12,
                 data_dir='lobby2/biwi_eth',
                 grid_width=1,
                 grid_height=1,
                 radius=2,
                 min_samples=100,
                 dt=0.1,
                 pattern=r'^.*\d{2}\.npy$',
                 model_file='koopman_model_mod8.pkl'):
        self._prediction_len = prediction_len
        self._GRID_WIDTH = grid_width
        self._GRID_HEIGHT = grid_height
        self._radius = radius
        self._min_samples = min_samples
        self._dt = dt
        self.pattern = pattern

        if os.path.exists(model_file):
            print(f"[INFO] Loading Koopman model from {model_file}")
            loaded_model = self.load_model(model_file)

            self._xmin = loaded_model._xmin
            self._xmax = loaded_model._xmax
            self._ymin = loaded_model._ymin
            self._ymax = loaded_model._ymax
            self._grid_x = loaded_model._grid_x
            self._grid_y = loaded_model._grid_y
            self._nx = loaded_model._nx
            self._ny = loaded_model._ny
            self._Ks = loaded_model._Ks
        else:
            print(f"[INFO] No saved model found. Building Koopman model from training data...")
            # 훈련 데이터 로딩 및 local Koopman operator 구축
            all_past, all_future = self._load_training_data(data_dir)

            xs = all_past[0]
            ys = all_past[1]
            self._xmin, self._xmax = xs.min(), xs.max()
            self._ymin, self._ymax = ys.min(), ys.max()

            self._grid_x = np.arange(self._xmin, self._xmax + grid_width, grid_width)
            self._grid_y = np.arange(self._ymin, self._ymax + grid_height, grid_height)
            self._nx = len(self._grid_x)
            self._ny = len(self._grid_y)
            self._Ks = np.empty((self._nx, self._ny), dtype=object)

            self._build_local_koopman(all_past, all_future)

            # 모델 저장
            self.save_model(model_file)
            print(f"[INFO] Koopman model saved to {model_file}")

    def __call__(self, tracking_result):
        # 입력 tracking_result: {object_id: trajectory (numpy array)}
        prediction_result = {}
        required_history = 8  # history 길이 8 사용
        for object_id, history in tracking_result.items():
            if not isinstance(history, np.ndarray):
                history = np.array(history)
            if len(history) < required_history:
                if len(history) == 0:
                    xy0 = np.array([0., 0.])
                    p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                elif len(history) == 1:
                    xy0 = history[-1]
                    p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                else:
                    dt_ = self._dt
                    xy_last = history[-1]
                    xy_second_last = history[-2]
                    vel = (xy_last - xy_second_last) / dt_
                    p = []
                    for k in range(self._prediction_len):
                        xy_pred = xy_last + vel * ((k+1)*dt_)
                        p.append(xy_pred)
                    p = np.array(p)
                prediction_result[object_id] = p
                continue

            # 최근 8 프레임을 사용하여 16차원 state 구성 (최신 프레임부터)
            past_frames = history[-required_history:]
            state = past_frames[::-1].flatten()  # [x_t, y_t, x_{t-1}, y_{t-1}, …]
            ps = []
            for _ in range(self._prediction_len):
                state = self._forward(state)
                ps.append(state[:2])  # 예측 결과의 (x,y)
            prediction_result[object_id] = np.array(ps)
        return prediction_result

    # 훈련 데이터 로딩 (history 8 기반)
    def _load_training_data(self, data_dir):
        train_dir = os.path.join(data_dir, 'train')
        pattern = re.compile(r'^.*\.npy$')
        all_past = []
        all_future = []
        required_history = 8
        for fname in os.listdir(train_dir):
            if pattern.match(fname):
                filepath = os.path.join(train_dir, fname)
                data = np.load(filepath)  # shape: (T, num_agents, 2)
                T, num_agents, _ = data.shape
                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < (required_history + 1):
                        continue
                    for i in range(len(valid_idx) - required_history):
                        past_indices = valid_idx[i:i+required_history]
                        future_indices = valid_idx[i+1:i+required_history+1]
                        past_window = agent_xy[past_indices]
                        future_window = agent_xy[future_indices]
                        past_state = past_window[::-1].flatten()
                        future_state = future_window[::-1].flatten()
                        all_past.append(past_state)
                        all_future.append(future_state)
        if len(all_past) == 0:
            raise ValueError(f"No valid training data found in {train_dir}.")
        all_past = np.array(all_past).T  # shape: (16, N)
        all_future = np.array(all_future).T  # shape: (16, N)
        return all_past, all_future

    # 각 그리드 셀마다 local Koopman operator 구성
    def _build_local_koopman(self, all_past, all_future):
        px_all = all_past[::2, :]  # 모든 x 좌표
        py_all = all_past[1::2, :]  # 모든 y 좌표

        for i in range(self._nx):
            for j in range(self._ny):
                gx = self._grid_x[i]
                gy = self._grid_y[j]
                dist_past_sq = (px_all[:4, :] - gx)**2 + (py_all[:4, :] - gy)**2
                idx = np.where(np.all(dist_past_sq <= self._radius**2, axis=0))[0]
                if len(idx) < self._min_samples:
                    self._Ks[i, j] = np.eye(self._observable_dim())
                    continue
                local_past = all_past[:, idx]
                local_future = all_future[:, idx]
                psi_past_list = []
                psi_future_list = []
                n_local = local_past.shape[1]
                for k in range(n_local):
                    psi_past_list.append(self._compute_observables(local_past[:, k]))
                    psi_future_list.append(self._compute_observables(local_future[:, k]))
                psi_past = np.column_stack(psi_past_list)
                psi_future = np.column_stack(psi_future_list)
                try:
                    pseudo_inv = np.linalg.pinv(psi_past)
                    K_local = psi_future @ pseudo_inv
                except np.linalg.LinAlgError:
                    K_local = np.eye(self._observable_dim())
                self._Ks[i, j] = K_local

    def _observable_dim(self):
        return 16  # 여기서는 16차원 state 사용

    @staticmethod
    def _compute_observables(state):
        state = state.flatten()
        dt = 0.1  # 시간 간격
        dx = (state[::2][1:] - state[::2][:-1]) / dt
        dy = (state[1::2][1:] - state[1::2][:-1]) / dt

        # 다항식 항 (예: state^2, state^3 등 추가 가능)
        poly_terms = np.concatenate([state**2, state**3])

        # 여기서는 단순히 원본 state를 반환 (필요시 다항항 등을 포함하도록 수정 가능)
        observables = np.concatenate([state])
        return observables

    def _to_state(self, observables):
        return observables[:16]

    def _forward(self, state):
        x, y = state[0], state[1]
        K = self._get_K(x, y)
        o = self._compute_observables(state)
        o_next = K @ o
        next_state = self._to_state(o_next)
        return next_state

    def _to_idx(self, x, y):
        i = int((x - self._xmin) // self._GRID_WIDTH)
        j = int((y - self._ymin) // self._GRID_HEIGHT)
        i = np.clip(i, 0, self._nx - 1)
        j = np.clip(j, 0, self._ny - 1)
        return i, j

    def _get_K(self, x, y):
        i, j = self._to_idx(x, y)
        K = self._Ks[i, j]
        if K.shape[0] != self._observable_dim():
            K = np.eye(self._observable_dim())
            self._Ks[i, j] = K
        return K

    # 테스트 데이터를 평가하는 함수 (여기서는 testtraj2 사용)
    def testtraj2(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = None
        future = None
        prediction_len = 12
        history_len = 8
        diff = 8 - history_len
        
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
                        # 관측 길이가 부족한 경우 모두 NaN 처리
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

                print("Agent 길이:", len(agent_xy))
                print("goal shape:", goal.shape)
                print("future shape:", future.shape)

        return goal, future

    # ========================
    # 모델 저장 및 불러오기 기능
    # ========================
    def save_model(self, filename):
        model_dict = {
            'prediction_len': self._prediction_len,
            'GRID_WIDTH': self._GRID_WIDTH,
            'GRID_HEIGHT': self._GRID_HEIGHT,
            'radius': self._radius,
            'min_samples': self._min_samples,
            'dt': self._dt,
            'pattern': self.pattern,
            'xmin': self._xmin,
            'xmax': self._xmax,
            'ymin': self._ymin,
            'ymax': self._ymax,
            'grid_x': self._grid_x,
            'grid_y': self._grid_y,
            'nx': self._nx,
            'ny': self._ny,
            'Ks': self._Ks,
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
        instance = cls.__new__(cls)
        instance._prediction_len = model_dict['prediction_len']
        instance._GRID_WIDTH = model_dict['GRID_WIDTH']
        instance._GRID_HEIGHT = model_dict['GRID_HEIGHT']
        instance._radius = model_dict['radius']
        instance._min_samples = model_dict['min_samples']
        instance._dt = model_dict['dt']
        instance.pattern = model_dict['pattern']
        instance._xmin = model_dict['xmin']
        instance._xmax = model_dict['xmax']
        instance._ymin = model_dict['ymin']
        instance._ymax = model_dict['ymax']
        instance._grid_x = model_dict['grid_x']
        instance._grid_y = model_dict['grid_y']
        instance._nx = model_dict['nx']
        instance._ny = model_dict['ny']
        instance._Ks = model_dict['Ks']
        return instance
