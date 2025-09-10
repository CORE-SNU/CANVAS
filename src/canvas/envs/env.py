import os
import numpy as np


class Environment:
    def __init__(self, filepath, dt, init_robot_pose, n_pedestrians=4, t_begin=40, t_end=160):

        self._dt = dt

        self._data = np.load(filepath)
        #assert self._data.shape == (201, n_pedestrians, 2)
        self._track_id = list(range(self._data.shape[1]))

        self._init_pose = init_robot_pose

        # initialized with first t_begin steps
        self._tracking_result = None
        self._robot_pose = None
        self.robot_velocity = [0,0]
        self._step = None
        self._first_step = t_begin
        self._final_step = t_end - 1
#utilize valid pedestrians to get data out of here
    def _get_obs(self):
        return {i: self._data[self._step-7:self._step+1, i, :] for i in self._track_id }
    def _get_obs_future(self):
        return {i: self._data[self._step+1:self._step+13, i, :] for i in self._track_id }
    def reset(self):
        self._step = self._first_step
        self._robot_pose = np.copy(self._init_pose)
        return np.copy(self._robot_pose)
    def get_velocity(self):
        return self.robot_velocity
    def step(self, velocity):
        """
        Simulation of a differential drive robot
        :param velocity:
        :return:
        """
        position_x, position_y, orientation_z = self._robot_pose
        linear_x, angular_z = velocity
        self.robot_velocity=velocity
        position_x += self._dt * linear_x * np.cos(orientation_z)
        position_y += self._dt * linear_x * np.sin(orientation_z)
        orientation_z += self._dt * angular_z

        self._robot_pose = np.array([position_x, position_y, orientation_z])
        self._step += 1

        if self._step > self._final_step:
            done = True
        else:
            done = False

        return np.copy(self._robot_pose), done
