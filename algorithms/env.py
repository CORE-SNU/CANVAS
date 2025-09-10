import os
import numpy as np


class Environment:
    def __init__(self, filepath, dt, n_pedestrians=4, t_begin=40, t_end=160):

        self._dt = dt

        self._data = np.load(filepath)
        #assert self._data.shape == (201, n_pedestrians, 2)
        self._track_id = list(range(self._data.shape[1]))

        #self._init_pose = init_robot_pose

        # initialized with first t_begin steps
        self._tracking_result = None

        self._step = None
        self._first_step = t_begin
        self._final_step = t_end - 1
#utilize valid pedestrians to get data out of here
    def _get_obs(self,valid):
        return {i: self._data[self._step-7:self._step+1, i, :] for i in valid}

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
