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
from .visualization_utils import visualize_trajectory, draw_robot, add_arrow, visualize_point, project_rectangle_to_image
from canvas.envs.env_utils import Geometry, load_geometry


ASSET_DIR = pathlib.Path(__file__).parent.parent.parent / 'assets'


class Environment:
    # TODO: dynamics
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
        geom_path = ASSET_DIR / 'geometries' / '{}.json'.format(dataset.name)
        self.geometry: Geometry = load_geometry(geom_path)

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

    @property
    def goal(self):
        return np.copy(self._goal)

    @property
    def t_begin(self):
        return self._first_step

    @property
    def t_end(self):
        return self._final_step + 1

    @property
    def homography(self):
        return np.copy(self._H)

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

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42


        plt.clf(), plt.cla()
        fig, ax = plt.subplots()
        if self._dataset.name in ['snu-asri', 'snu-asri-ood']:
            frame_path = ASSET_DIR / "snu-asri.png"  # local file
        else:
            frame_path = os.path.join(self._path_to_frames, self.dataset_label, f"{self._step}.png")

        assert os.path.exists(frame_path), frame_path

        image = cv2.imread(frame_path)
        if self._dataset.name in ['snu-asri', 'snu-asri-ood']:
            ax.imshow(image, cmap='gray', alpha=0.6, extent=(-3.0, 8.5, -9.5, 1.5))
        else:
            ax.imshow(image, cmap='gray', alpha=0.6)

        ax.axis('off')

        H = self._H

        for obstacle in self.geometry:
            projected = project_rectangle_to_image(rectangle=obstacle, H=H, color='tab:red', alpha=0.2, zorder=1)
            ax.add_patch(projected)

        # visualization parameters
        ego_params = {'linestyle': 'solid', 'linewidth': 3, 'zorder': 2, 'alpha': 0.3, 'color': 'tab:gray'}
        non_ego_h_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 1}
        non_ego_f_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 1, 'color': 'tab:blue', 'alpha': 0.3, 'label': 'future'}
        arrowprops = {'arrowstyle': 'Fancy', 'mutation_scale': 30}

        if 'c' not in kwargs:
            # no color overlay -> default colors
            ego_params['color'] = 'tab:gray'
            non_ego_h_params['color'] = 'tab:red'
            kwargs['c'] = None

        # ego-robot
        ego_pos_trajectory = np.array([self._x_traj, self._y_traj]).T
        visualize_trajectory(
            trajectory=ego_pos_trajectory,
            H=H,
            ax=ax,
            c=None,
            **ego_params
        )
        draw_robot(ax, H, self._x, self._y, self._th, robot_img=self._robot_img)        # robot figure

        # visualization of non-ego agents
        # For visibility, we only visualized the agents nearby the ego-agents.

        # history
        hlen = max(1, min(self._history_len, self._step - self._first_step))
        #scene = self._dataset.get_scene(timestep=self._step, history_length=self._step-self._first_step)
        scene = self._dataset.get_scene(timestep=self._step, history_length=hlen)

        # TODO: determine the distance threshold
        # TODO: alternative: agents that changes the ego decision
        dist_thres = 4.
        xy_robot = np.array([self._x, self._y])
        nearby_agents = []
        for agent, h in scene.items():
            pos = h[-1]
            if np.sum((pos - xy_robot) ** 2) ** .5 < dist_thres:
                nearby_agents.append(agent)

        for agent, h in scene.items():
            if agent in nearby_agents:
                visualize_trajectory(
                    trajectory=h,
                    H=H,
                    ax=ax,
                    c=kwargs['c'],
                    **non_ego_h_params
                )

        # non-ego agents: future
        #future_scene = self._dataset.get_future(timestep=self._step, future_length=self._prediction_len,history_length=self._step-self._first_step)
        future_scene = self._dataset.get_future(timestep=self._step, future_length=self._prediction_len, history_length=hlen)
        labeled = False
        for agent, f in future_scene.items():
            if agent in nearby_agents:
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
            if node in nearby_agents:
                h, f = scene[node], future_scene[node]
                x_next, y_next = f[0]
                x, y = h[-1]
                add_arrow(x=x, y=y, x_next=x_next, y_next=y_next, H=H, ax=ax, arrowprops=arrowprops)
        '''
        norm = mpl.colors.Normalize(vmin=0., vmax=1., clip=True)
        cmap = cm.get_cmap('plasma')

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)

        cb = plt.colorbar(mappable, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_ticks([0, 0.5, 1])
        cb.ax.xaxis.set_ticklabels([0, 0.5, 1])
        # cb.set_label('competency index', rotation=0, labelpad=12, va='center')
        ax.text(x=1.05, y=1.02, s='competency index', fontsize=16, transform = ax.transAxes)
        '''
        # mark the goal position
        visualize_point(self._goal, H, ax, color='tab:pink', marker='*', s=160, label='goal', zorder=500)

        # (optional) ego-motion plan



        if 'open_loop_base' in kwargs:
            ego_plan_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 2, 'color': 'tab:pink', 'label': 'open loop (baseline)', 'alpha': 0.9}
            visualize_trajectory(
                trajectory=kwargs['open_loop_base'],
                H=H,
                ax=ax,
                c=None,
                **ego_plan_params
            )

        if 'open_loop_gt' in kwargs:
            ego_plan_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 2, 'color': 'tab:gray', 'label': 'open loop (oracle)', 'alpha': 0.9}
            visualize_trajectory(
                trajectory=kwargs['open_loop_gt'],
                H=H,
                ax=ax,
                c=None,
                **ego_plan_params
            )

        if 'open_loop' in kwargs:
            ego_plan_params = {'linestyle': 'solid', 'linewidth': 5, 'zorder': 2, 'color': 'tab:cyan', 'label': 'open loop', 'alpha': 0.9}
            visualize_trajectory(
                trajectory=kwargs['open_loop'],
                H=H,
                ax=ax,
                c=None,
                **ego_plan_params
            )

        ax_for_idx = fig.add_axes([0.7, 0.1, .5, .5])
        ax_for_idx.yaxis.set_ticks_position('right')
        ax_for_idx.plot(kwargs['hc'][-10:], label='PR (hindsight)', linestyle='dashed', color='#808000', linewidth=3, alpha=0.5)
        # ax_for_idx.plot(kwargs['c'][-10:], label='PR', linestyle='solid', color='#808000', linewidth=3)
        ax_for_idx.grid(True)
        ax_for_idx.legend()

        h, w, _ = image.shape
        if self.dataset_label.lower() == "snu-asri" or self.dataset_label.lower() == "lobby":
            ax.set_xlim(-3.0, 8.5)
            ax.set_ylim(1.5, -9.5)
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        ax.legend(fontsize=12, ncols=1, loc='upper left', bbox_to_anchor=(1, 0., .3, 1.))



        fig.savefig(os.path.join(self._path_to_save, '{:03d}.png'.format(self._step)), bbox_inches='tight', pad_inches=0)
        plt.close()

        return fig, ax