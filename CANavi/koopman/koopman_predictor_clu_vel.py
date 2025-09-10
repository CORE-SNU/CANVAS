import os
import re
import pickle
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
import scipy.linalg

class KoopmanPredictor:
    def __init__(self,
                 prediction_len=12,
                 data_dir='lobby2/biwi_eth',
                 min_samples=1000,
                 dt=0.4,
                 pattern=r'^.*\d{2}\.npy$',
                 n_clusters=15,
                 model_file='/home/core/Documents/MPC20250312/koopman/koopman_model_clu_vel.pkl'):

        self._prediction_len = prediction_len
        self._min_samples = min_samples
        self._dt = dt
        self.pattern = pattern
        self.n_clusters = n_clusters

        if os.path.exists(model_file):
            print(f"[INFO] Loading Koopman model from {model_file}")
            loaded_model = self.load_model(model_file)

            self.kmeans = loaded_model.kmeans
            self.local_Ks = loaded_model.local_Ks

        else:
            print(f"[INFO] No saved model found. Building Koopman model from training data...")
            # 훈련 데이터 로딩 및 local Koopman operator 구축
            all_past, all_future = self._load_training_data(data_dir)

            samples = all_past.T
            features = []
            for s in samples:
                xs = s[0::2]
                ys = s[1::2]
                vx = np.diff(xs) / self._dt
                vy = np.diff(ys) / self._dt
                avg_vx = np.mean(vx)
                avg_vy = np.mean(vy)
                angles = np.arctan2(vy, vx)
                angle_diffs = np.diff(angles)
                angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
                avg_ang_vel = np.mean(angle_diffs) / self._dt
                features.append([avg_vx, avg_vy, avg_ang_vel])
            features = np.array(features)

            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = self.kmeans.fit_predict(features)

            self.local_Ks = []
            for c in range(n_clusters):
                idx = np.where(cluster_labels == c)[0]
                if len(idx) < self._min_samples:
                    K_local = np.eye(self._observable_dim())
                else:
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
                self.local_Ks.append(K_local)

            self.save_model(model_file)
            print(f"[INFO] Koopman model saved to {model_file}")

    def __call__(self, tracking_result):
        prediction_result = {}
        required_history = 8  # history 길이 8 사용
        for object_id, history in tracking_result.items():
            if not isinstance(history, np.ndarray):
                history = np.array(history)
            if len(history) < required_history:
                xy0 = history[-1] if len(history) > 0 else np.array([0., 0.])
                p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                prediction_result[object_id] = p
                continue
            past_frames = history[-required_history:]
            state = past_frames[::-1].flatten()  # 최신 프레임이 앞쪽에 오도록 (16차원)
            ps = []
            for _ in range(self._prediction_len):
                state = self._forward(state)  # 다음 state 예측
                ps.append(state[:2])          # 예측된 위치 (x, y)
            prediction_result[object_id] = np.array(ps)
        return prediction_result

    def _load_training_data(self, data_dir):
        train_dir = os.path.join(data_dir, 'train')
        pattern = re.compile(r'^.*\.npy$')
        all_past = []
        all_future = []
        required_history = 8  # history 길이 8
        for fname in os.listdir(train_dir):
            if pattern.match(fname):
                filepath = os.path.join(train_dir, fname)
                data = np.load(filepath)  # shape (T, num_agents, 2)
                T, num_agents, _ = data.shape
                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < (required_history + 1):
                        continue
                    for i in range(len(valid_idx) - required_history):
                        past_indices = valid_idx[i:i+required_history]
                        future_indices = valid_idx[i+1:i+required_history+1]
                        past_window = agent_xy[past_indices]    # (8,2)
                        future_window = agent_xy[future_indices]  # (8,2)
                        past_state = past_window[::-1].flatten()
                        future_state = future_window[::-1].flatten()
                        all_past.append(past_state)
                        all_future.append(future_state)
        if len(all_past) == 0:
            raise ValueError(f"No valid training data found in {train_dir}.")
        all_past = np.array(all_past).T  # (16, N)
        all_future = np.array(all_future).T  # (16, N)
        return all_past, all_future

    def _observable_dim(self):
        # state (16) + speed_mean (1) + omega_mean (1)
        return 16+16
    

    def _compute_observables(self, state):
        state = state.flatten()  # (16,)
        dt = self._dt  # Time step

        # 1차 미분 (속도)
        dx = (state[::2][1:] - state[::2][:-1]) / dt  # (7,)
        dy = (state[1::2][1:] - state[1::2][:-1]) / dt  # (7,)

        # 선속도 (magnitude of velocity)
        speed = np.sqrt(dx**2 + dy**2)  # (7,)
        speed_mean = np.mean(speed)

        # 각속도 (변화량)
        theta = np.arctan2(dy, dx)  # (7,)
        omega = (theta[1:] - theta[:-1]) / dt  # (6,)
        omega_mean = np.mean(omega)

        observables = np.concatenate([state, state**2])
        return observables


    def _get_K(self, state):
        """
        예측할 state(16차원)에서 clustering feature를 계산할 때
        평균 vx, 평균 vy와 함께 각속도 관련 feature도 포함시킵니다.
        """
        xs = state[0::2]
        ys = state[1::2]
        vx = np.diff(xs) / self._dt
        vy = np.diff(ys) / self._dt
        avg_vx = np.mean(vx)
        avg_vy = np.mean(vy)

        angles = np.arctan2(vy, vx)
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        avg_ang_vel = np.mean(angle_diffs) / self._dt
        avg_abs_ang_vel = np.mean(np.abs(angle_diffs)) / self._dt

        feature = np.array([[avg_vx, avg_vy, avg_ang_vel]])
        cluster_label = self.kmeans.predict(feature)[0]
        K = self.local_Ks[cluster_label]
        if K.shape[0] != self._observable_dim():
            K = np.eye(self._observable_dim())
            self.local_Ks[cluster_label] = K
        return K

    def _forward(self, state):
        K = self._get_K(state)
        o = self._compute_observables(state)
        o_next = K @ o
        next_state = self._to_state(o_next)
        return next_state

    def _to_state(self, observables):
        # observables에서 원래 16차원 state 복원
        return observables[:16]

    # 테스트 관련 함수들 (testtraj, testtraj2, evaluate_test)
    def testtraj(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = []
        future = []
        prediction_len = 12
        history_len = 8
        diff = 8 - history_len
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
                    if len(valid_idx) < (history_len + prediction_len):
                        continue
                    for start_idx in range(valid_idx[0] + diff):
                        gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                        all_ade.append(gt_future)
                        all_fde.append(gt_future)
                    for start_idx in range(diff, len(valid_idx) - history_len - prediction_len + 1):
                        history_idx = valid_idx[start_idx:start_idx + history_len]
                        future_idx = valid_idx[start_idx + history_len:start_idx + history_len + prediction_len]
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
                    for start_idx in range(valid_idx[-1] - 1, len(agent_xy)):
                        gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                        all_ade.append(gt_future)
                        all_fde.append(gt_future)
                    n_windows = len(all_fde)
                    all_fde_arr = np.array(all_fde).reshape(n_windows, prediction_len, 1, 2)
                    all_ade_arr = np.array(all_ade).reshape(n_windows, prediction_len, 1, 2)
                    if len(goal):
                        future = np.concatenate([future, all_fde_arr], axis=2)
                        goal = np.concatenate([goal, all_ade_arr], axis=2)
                    else:
                        future = all_fde_arr
                        goal = all_ade_arr
        print(goal.shape)
        print(future.shape)
        return goal, future

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
                        # 관측 길이가 부족한 경우 NaN 패딩
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
                        history_idx = valid_idx[start_idx:start_idx + history_len]
                        future_idx = valid_idx[start_idx + history_len:start_idx + required_length]
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

    # ========================================
    # 모델 저장 및 불러오기 기능 (pickle 사용)
    # ========================================
    def save_model(self, filename):
        model_dict = {
            'prediction_len': self._prediction_len,
            'min_samples': self._min_samples,
            'dt': self._dt,
            'pattern': self.pattern,
            'n_clusters': self.n_clusters,
            'kmeans': self.kmeans,
            'local_Ks': self.local_Ks,
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)
    
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
        instance = cls.__new__(cls)
        instance._prediction_len = model_dict['prediction_len']
        instance._min_samples = model_dict['min_samples']
        instance._dt = model_dict['dt']
        instance.pattern = model_dict['pattern']
        instance.n_clusters = model_dict['n_clusters']
        instance.kmeans = model_dict['kmeans']
        instance.local_Ks = model_dict['local_Ks']
        return instance
