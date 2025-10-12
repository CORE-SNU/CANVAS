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
        self._data = np.load(self._path)

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
            return ts_active[0], ts_active[-1] + 1
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
            # edited this section due to all predictors implemented requiring
            # contiguous history of length 8
            t_end = self._active_intervals[idx, 1]
            if t_end > timestep + 1:
                t_end = int(min(t_end, timestep + future_length + 1))
                future_array = self._data[timestep+1:t_end, idx]
                future[idx] = np.copy(future_array)

        return future