import numpy as np
import os
import pathlib
from typing import Union, Dict
import math


class Dataset:
    """
    A class containing the metadata of a dataset
    """

    # TODO: let's just use trajdata
    def __init__(
            self,
            name: str,
            data_path: Union[str, pathlib.Path],
            dt: float
    ):
        # TODO: visualization path
        self._name = name  # dataset name

        assert os.path.exists(data_path)
        assert str(data_path).endswith('.npy')

        self._path = data_path
        self._data = load_data(self._path)          # shape: (# frames, # agents, feat. dim.)

        self._active_intervals = self._get_active_time_intervals()
        self._active_agents = self._get_active_agents()
        self._dt = dt

        return

    @property
    def dt(self):
        return self._dt

    @property
    def name(self):
        return self._name

    @property
    def max_timesteps(self):
        return self._data.shape[0]

    @property
    def num_agents(self):
        return self._data.shape[1]

    def asarray(self):
        return np.copy(self._data)



    def _get_active_time_interval(self, agent_idx: int):
        # [t_begin, t_end)
        assert 0 <= agent_idx < self.num_agents

        # shape: (# steps,)
        x_pos_i = self._data[:, agent_idx, 0]
        ts_active, = np.nonzero(~np.isnan(x_pos_i))
        if ts_active.size > 0:
            t_begin, t_end = ts_active[0], ts_active[-1] + 1
            x_active = x_pos_i[t_begin: t_end]
            is_invalid = np.isnan(x_active)      # true iff not nan
            ts_invalid, = np.nonzero(is_invalid)
            # check if there is any intermediate nan's
            assert np.all(~ts_invalid), 'agent {}: {}'.format(agent_idx, t_begin+ts_invalid)
            return t_begin, t_end
        else:
            return -np.inf, np.inf

    def _get_active_time_intervals(self):
        intervals = np.zeros((self.num_agents, 2))
        for agent_idx in range(self.num_agents):
            t_begin, t_end = self._get_active_time_interval(agent_idx=agent_idx)
            intervals[agent_idx, 0], intervals[agent_idx, 1] = t_begin, t_end
        return intervals

    def _get_active_agents(self):
        agents = {}
        for t in range(self.max_timesteps):
            # filter active agents by inspecting if their x positions are not nan
            x_pos_t = self._data[t, :, 0]
            agent_indices, = np.nonzero(~np.isnan(x_pos_t))
            agents[t] = agent_indices
        return agents

    def get_scene(
            self,
            timestep: int,
            history_length: int
    ) -> Dict[int, np.ndarray]:
        """
        Return a dictionary of agent-history pairs
        """
        assert timestep < self.max_timesteps
        assert history_length > 0

        scene = {}
        for idx in self._active_agents[timestep]:
            t_begin = self._active_intervals[idx, 0]
            if t_begin >= timestep - history_length + 1:
                continue
            
            # edited this section due to all predictors implemented requiring
            # contiguous history of length 8
            t_begin = int(max(t_begin, timestep-history_length+1))

            history_array = self._data[t_begin:timestep+1, idx]
            if not np.isfinite(history_array).all():
                continue
            #added this to avoid issues with incomplete history
            scene[idx] = np.copy(history_array)

        return scene

    def get_future(self, timestep: int, future_length: int,history_length:int) -> Dict[int, np.ndarray]:

        assert timestep < self.max_timesteps
        assert future_length > 0
        assert history_length > 0

        future = {}
        for idx in self._active_agents[timestep]:
            t_begin = self._active_intervals[idx, 0]
            if t_begin >= timestep - history_length + 1:
                continue
            t_begin = int(max(t_begin, timestep-history_length+1))

            history_array = self._data[t_begin:timestep+1, idx]
            if not np.isfinite(history_array).all():
                continue
            # edited this section due to all predictors implemented requiring
            # contiguous history of length 8 without nans
            t_end = self._active_intervals[idx, 1]
            if t_end > timestep + 1:
                t_end = int(min(t_end, timestep + future_length + 1))
                future_array = self._data[timestep+1:t_end, idx]
                future[idx] = np.copy(future_array)

        return future

def load_data(path):
    data = np.load(path)
    data = fill_nans(data)
    return data


def fill_nans(x: np.ndarray) -> np.ndarray:
    """
    Fill NaNs that are *between* the first and last non-NaN in each row of a 2D array
    using linear interpolation. NaNs before the first non-NaN and after the last
    non-NaN are left as NaN.

    Parameters
    ----------
    x : np.ndarray
        A 3D NumPy array of shape (# frames, # agents, feature dim.) possibly containing NaNs.

    Returns
    -------
    np.ndarray
        A copy of x with internal NaNs filled by linear interpolation per row.
    """

    y = x.copy()

    n_frames, n_agents, feat_dim = y.shape
    ts = np.arange(n_frames)

    for j in range(feat_dim):
        for i in range(n_agents):
            traj = y[:, i, j]
            mask = ~np.isnan(traj)  # false <-> nan
            # Need at least two non-NaNs to interpolate between
            if mask.sum() < 2:
                continue

            ts_valid = ts[mask]        # valid timesteps
            b, e = ts_valid[0], ts_valid[-1]    # begin & end

            # Interpolate only on [b, e]; outside remains unchanged (NaN if it was NaN)
            ts_seg = ts[b:e+1]
            traj_seg = np.interp(ts_seg, ts_valid, traj[mask])

            traj[b:e+1] = traj_seg
            y[:, i, j] = traj
    return y



