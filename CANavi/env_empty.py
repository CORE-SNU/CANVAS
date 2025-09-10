import os
import numpy as np


class Environment:
    def __init__(self, filepath, dt, init_robot_pose, n_pedestrians=4, t_begin=40, t_end=160):

        self._dt = dt

        self._data = np.load(filepath)
        print(self._data.shape)
        print(filepath)
        #assert self._data.shape == (201, n_pedestrians, 2)
        self._track_id = list(range(self._data.shape[1]))

        # initialized with first t_begin steps
        self._tracking_result = None

        self._step = None
        self._first_step = 40
        self._final_step = min(400, self._data.shape[0] - 1)
#utilize valid pedestrians to get data out of here
    def _get_obs(self):
        return {i: self._data[self._step-7:self._step+1, i, :] for i in self._track_id }
    def reset(self):
        self._step = self._first_step
        return None
    def step(self):
        """
        Simulation of a differential drive robot
        :param velocity:
        :return:
        """

        self._step += 1

        if self._step > self._final_step:
            done = True
        else:
            done = False

        return None, done
