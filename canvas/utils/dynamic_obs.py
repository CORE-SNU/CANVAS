# code to simplify the dynamic observation detection process
import numpy as np
def dynamic_observation_filter(observation, position_x, position_y, prediction_len,observation_future_true=None,max_ped=40):
    valid_obs = {}
    valid_obs_future_true = {}
    if isinstance(observation, dict):
        for pid, traj in observation.items():
            # history
            try:
                arr_hist = np.asarray(traj, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if not (arr_hist.ndim == 2 and arr_hist.shape[0] == 8 and arr_hist.shape[1] >= 2 and np.isfinite(arr_hist[:, :2]).all()):
                continue
            valid_obs[pid] = arr_hist

            # GT future (kept for viz & later scoring; not used by controller here)
            fut = observation_future_true.get(pid, None) if isinstance(observation_future_true, dict) else None
            if fut is None:
                continue
            arr_fut = np.asarray(fut, dtype=np.float64)
            if not (arr_hist.ndim == 2 and arr_hist.shape[0] == 8 and arr_hist.shape[1] >= 2 and np.isfinite(arr_hist[:, :2]).all()):
                continue
            valid_obs_future_true[pid] = arr_fut[:prediction_len, :2]
            if len(valid_obs)   >= max_ped:
                print("Max pedestrians reached, skipping remaining...")
                break  # limit max pedestrians to max_ped
    # --------- Simple collision check (proximity to last history point) ---------
    dynamic_obs = {}
    if valid_obs:
        dynamic_obs = valid_obs
        initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
        robot_pos = np.array([position_x, position_y])
        distances = np.sqrt(np.sum((initial_positions - robot_pos) ** 2, axis=1))
        #if np.any(distances <= 0.7):
        #    print("Collision!")
        #    collision_count += int(np.sum(distances <= 0.7))
    if isinstance(observation_future_true,dict):
        return dynamic_obs, valid_obs_future_true
    else:
        return dynamic_obs