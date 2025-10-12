import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

from canvas.datasets import Dataset
from .visualization_utils import visualize_trajectory, draw_robot, add_arrow, visualize_point

ASSET_DIR = pathlib.Path(__file__).parent.parent.parent / 'assets'


class Environment:
    # TODO: environmental geometry
    # TODO: initial state of the robot
    # TODO: dynamics
    # TODO: goal task (task)
    # TODO: time interval
    # TODO: state bounds & control bounds

    def __init__(
            self,
            dataset: Dataset,
            init_robot_state,
            goal_pos,
            t_begin=40,
            t_end=160,
            history_len=8,
            prediction_horizon=12,
            path_to_frames=None,
            path_to_save=None,
    ):
        """
        A simple environment for simulating a differential drive robot
        moving in a crowd of pedestrians.
        :param filepath: path to the dataset file (can be passed in short code form like 'ETH', 'ZARA1', etc.)
        :param dt: timestep for the robot simulation   
        :param init_robot_pose: initial pose of the robot [x(m),y(m),theta(rad)]
        :param t_begin: first timestep to use from the dataset (default: 40)
        :param t_end: last timestep to use from the dataset (default: 160)
        :param history_len: number of observed steps provided to the model.
        :param prediction_horizon: number of future steps to predict.
        """

        self._dataset: Dataset = dataset

        self._dt = dataset.dt

        # assert self._data.shape == (201, n_pedestrians, 2)

        self._init_state = init_robot_state

        self._goal = goal_pos

        self._first_step = t_begin
        self._final_step = t_end - 1
        self._history_len = history_len
        self._prediction_len = prediction_horizon

        # placeholders for simulations
        # timestep
        self._step = None
        # robot state variables
        self._x = None
        self._y = None
        self._th = None

        # initialized with first t_begin steps
        self._tracking_result = None

        self._step = None

        # for rendering
        self._x_traj = None
        self._y_traj = None
        self._th_traj = None

        # homography matrix (for visualization)
        if "zara01" in dataset.name:
            dataset_label = "zara1"
        elif dataset.name == "snu-asri":
            dataset_label = "Lobby"
        else:
            dataset_label = dataset.name
        
        homography_path = ASSET_DIR / 'homographies' / (dataset_label + '.txt')
        self.dataset_label=dataset_label
        assert os.path.exists(homography_path)
        self._H = np.loadtxt(homography_path, dtype=float)

        self._path_to_frames = path_to_frames
        self._path_to_save = path_to_save

        self._robot_img = Image.open(ASSET_DIR / 'robot.png')

        if path_to_save is not None:
            os.makedirs(path_to_save, exist_ok=True)

    # utilize valid pedestrians to get data out of here

    @property
    def timestep(self):
        return self._step

    @property
    def dt(self):
        return self._dt

    def _get_obs(self):
        """Get the history length amount of observed trajectories of all pedestrians up to the current step."""

        return {
            'ego': self._get_robot_state(),
            'non-ego': self._dataset.get_scene(timestep=self._step, history_length=self._history_len)
        }
        # return {i: self._data[self._step-self._history_len+1:self._step+1, i, :] for i in self._track_id }

    '''
    def _get_obs_future(self):
        """Get the prediction length amount of future trajectories of all pedestrians after the current step."""
        return {i: self._data[self._step + 1:self._step + self._prediction_len + 1, i, :] for i in self._track_id}
    '''

    def _get_robot_state(self):
        return {'position_x': self._x, 'position_y': self._y, 'orientation_z': self._th}

    def reset(self):
        """
        Reset the environment to the initial state."""
        self._step = self._first_step
        self._x = self._init_state['position_x']
        self._y = self._init_state['position_y']
        self._th = self._init_state['orientation_z']

        self._initialize_buffers()
        self._update_buffers()

        return self._get_obs(), self._get_side_info()

    '''
    def get_velocity(self):
        """
        Get the current velocity of the robot"""
        return self.robot_velocity
    '''

    def _initialize_buffers(self):
        self._x_traj = []
        self._y_traj = []
        self._th_traj = []

    def _update_buffers(self):
        self._x_traj.append(self._x)
        self._y_traj.append(self._y)
        self._th_traj.append(self._th)

    def _get_side_info(self):
        # TODO: Is goal reached? Does a collision occur?
        return {
            'distance_to_goal': ((self._x - self._goal[0]) ** 2 + (self._y - self._goal[1]) ** 2) ** .5,
            'goal_reached': np.abs(self._x - self._goal[0]) < 0.3 and np.abs(self._y - self._goal[1]) < 0.3,
            'future': self._dataset.get_future(timestep=self._step, future_length=self._prediction_len, history_length=self._history_len)
        }

    def step(self, action):
        """
        Simulation of a differential drive robot
        :param velocity:
        :return:
        """

        linear_x, angular_z = action
        # self.robot_velocity = velocity

        self._x += self._dt * linear_x * np.cos(self._th)
        self._y += self._dt * linear_x * np.sin(self._th)
        self._th += self._dt * angular_z

        self._step += 1

        info = self._get_side_info()

        terminated = info['goal_reached']
        truncated = (self._step > self._final_step)

        self._update_buffers()

        return self._get_obs(), terminated, truncated, info

    def render(self, **kwargs):
        assert self._path_to_frames is not None and self._path_to_save is not None
        plt.clf(), plt.cla()
        fig, ax = plt.subplots()
        if str(getattr(self, "dataset_label", "")).lower() == "lobby":
            frame_path = "lobby3.png"  # local file
        else:
            frame_path = os.path.join(self._path_to_frames, self.dataset_label, f"{self._step}.png")
        #frame_path = os.path.join(self._path_to_frames, self.dataset_label, '{}.png'.format(self._step))
        image = cv2.imread(frame_path)
        if self.dataset_label.lower() == "lobby":
            ax.imshow(image, cmap='gray', alpha=0.6,extent=(-3.0, 8.5, -9.5, 1.5))
        else:
            ax.imshow(image, cmap='gray', alpha=0.6)


        ax.axis('off')

        H = self._H

        # visualization parameters
        ego_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 2}
        non_ego_h_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 1}
        non_ego_f_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 1, 'color': 'tab:blue', 'alpha': 0.8, 'label': 'future'}
        arrowprops = {'arrowstyle': 'Fancy', 'mutation_scale': 30}

        if 'c' not in kwargs:
            # no color overlay -> default colors
            ego_params['color'] = 'cyan'
            non_ego_h_params['color'] = 'tab:red'
            kwargs['c'] = None

        # ego-robot
        ego_pos_trajectory = np.array([self._x_traj, self._y_traj]).T
        visualize_trajectory(
            trajectory=ego_pos_trajectory,
            H=H,
            ax=ax,
            c=kwargs['c'],
            **ego_params
        )
        draw_robot(ax, H, self._x, self._y, self._th, robot_img=self._robot_img)        # robot figure

        # non-ego agents: history
        scene = self._dataset.get_scene(timestep=self._step, history_length=self._step-self._first_step)

        for _, h in scene.items():
            visualize_trajectory(
                trajectory=h,
                H=H,
                ax=ax,
                c=kwargs['c'],
                **non_ego_h_params
            )

        # non-ego agents: future
        future_scene = self._dataset.get_future(timestep=self._step, future_length=self._prediction_len,history_length=self._step-self._first_step)
        labeled = False
        for _, f in future_scene.items():
            visualize_trajectory(
                trajectory=f,
                H=H,
                ax=ax,
                **non_ego_f_params
            )
            if not labeled:
                del non_ego_f_params['label']
                labeled = True

        # non-ego agents: draw arrows indicating the directions of the agents
        for node in (scene.keys() & future_scene.keys()):
            h, f = scene[node], future_scene[node]
            x_next, y_next = f[0]
            x, y = h[-1]
            add_arrow(x=x, y=y, x_next=x_next, y_next=y_next, H=H, ax=ax, arrowprops=arrowprops)

        norm = mpl.colors.Normalize(vmin=0., vmax=1., clip=True)
        cmap = cm.get_cmap('plasma')

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cb = plt.colorbar(mappable, cax=cax)
        cb.set_label('competency index', rotation=90, labelpad=12, va='center')

        # mark the goal position
        visualize_point(self._goal, H, ax, color='tab:pink', marker='*', s=160, label='goal', zorder=500)

        h, w, _ = image.shape
        if self.dataset_label.lower() == "lobby":
            ax.set_xlim(-3.0, 8.5)
            ax.set_ylim(1.5, -9.5)
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        ax.legend()
        # fig.savefig(os.path.join(self._path_to_save, '{:03d}.png'.format(self._step)), bbox_inches='tight', pad_inches=0)
        # plt.close()

        return fig, ax