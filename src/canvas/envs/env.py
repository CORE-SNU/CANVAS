import os
import numpy as np
from ..datasets import load_dataset

class Environment:
    def __init__(self, filepath, dt, init_robot_pose, t_begin=40, t_end=160,history_len=8,prediction_len=12):
        """
        A simple environment for simulating a differential drive robot
        moving in a crowd of pedestrians.
        :param filepath: path to the dataset file (can be passed in short code form like 'ETH', 'ZARA1', etc.)
        :param dt: timestep for the robot simulation   
        :param init_robot_pose: initial pose of the robot [x(m),y(m),theta(rad)]
        :param t_begin: first timestep to use from the dataset (default: 40)
        :param t_end: last timestep to use from the dataset (default: 160)
        :param history_len: number of observed steps provided to the model.
        :param prediction_len: number of future steps to predict.
        """
        self._dt = dt

        self._data = load_dataset(filepath)
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
        self._history_len=history_len
        self._prediction_len=prediction_len
#utilize valid pedestrians to get data out of here
    def _get_obs(self):
        """Get the history length amount of observed trajectories of all pedestrians up to the current step."""
        return {i: self._data[self._step-self._history_len-1:self._step+1, i, :] for i in self._track_id }
    def _get_obs_future(self):
        """Get the prediction length amount of future trajectories of all pedestrians after the current step."""
        return {i: self._data[self._step+1:self._step+self._prediction_len+1, i, :] for i in self._track_id }
    def reset(self):
        """
        Reset the environment to the initial state."""
        self._step = self._first_step
        self._robot_pose = np.copy(self._init_pose)
        return np.copy(self._robot_pose)
    def get_velocity(self):
        """
        Get the current velocity of the robot"""
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
