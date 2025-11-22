import numpy as np
from itertools import product
from canvas.envs.env_utils import Geometry
from canvas.controllers.optim_solver import solve


class BaseMPC:
    def __init__(self, prediction_horizon, dt, goal, d_min, geometry: Geometry, use_ipopt: bool = False):
        self._n_steps = prediction_horizon
        self._dt = dt
        self._goal = goal
        self._d_min = d_min
        self._geom: Geometry = geometry
        self.use_ipopt = use_ipopt

    def __call__(self, obs, prediction_res, change_controller_state=False):
        o = obs['ego']
        x, y, th = o['position_x'], o['position_y'], o['orientation_z']
        Xs, Us = self.generate_paths(x, y, th, n_skip=4)        # shape of Xs: (# samples, prediction horizon + 1, dim.)

        X, U, min_cost_to_go = self.score_paths(Xs, Us, prediction_res)

        if self.use_ipopt:
            # IPOPT-based path refinement
            X, U = solve(
                ts=self._n_steps+1,
                dt=self._dt,
                ulb=np.array([-.8, -.7]),
                uub=np.array([.8, .7]),
                geometry=self._geom,
                predictions=prediction_res,
                r_robot=.4,
                r_agent=.1 / np.sqrt(2.),
                initial_pose=np.array([x, y, th]),
                goal=self._goal,
                X=X,
                U=U
            )
        min_cost_to_go = self.cost_to_go(obs, prediction_res, U_refined)

        info = {
            'X': X,
            'U': U,
            'cost_to_go': min_cost_to_go
        }
        return U[0], info

    def score_paths(self, Xs, Us, prediction_res):
        intermediate_cost = np.sum((Xs[:, :-1, :2] - self._goal) ** 2, axis=(-2,-1))
        # control_cost = .001 * np.sum(Us ** 2, axis=(-2, -1))
        terminal_cost = 10. * np.sum((Xs[:, -1, :2] - self._goal) ** 2, axis=-1)

        collision_cost = np.zeros_like(terminal_cost)
        weight = 1e3
        if prediction_res:
            for i in range(self._n_steps):
                       # shape: (# agents, # samples) -> (# samples,)
                d_static = self._geom.distance_from(points=Xs[:, i + 1, :2])
                d_dynamic = np.min([np.sum((Xs[:, i + 1, :2] - p[i]) ** 2, axis=-1) ** .5 if p.shape[0] > i else 1e5 * np.ones_like(terminal_cost) for p in prediction_res.values()], axis=0)
                d = np.minimum(d_static, d_dynamic)
                collision_cost += weight * np.where(d <= self._d_min, 1., 0.)

        c = intermediate_cost + terminal_cost + collision_cost

        idx_min = np.argmin(c)
        min_cost_to_go = np.min(c)
        return Xs[idx_min], Us[idx_min], min_cost_to_go

    def rollout(self, state, U) -> np.ndarray:
        """
        Unroll a single state trajectory from an action sequence
        """
        X = np.zeros((self._n_steps+1, 3))
        X[0] = state
        dt = self._dt
        for t in range(self._n_steps):
            v, w = U[t, 0], U[t, 1]
            X[t + 1, 0] = X[t, 0] + dt * v * np.cos(X[t, 2])
            X[t + 1, 1] = X[t, 1] + dt * v * np.sin(X[t, 2])
            X[t + 1, 2] = X[t, 2] + dt * w
        return X

    def cost_to_go(self, obs, prediction_res, U):
        o = obs['ego']
        x, y, th = o['position_x'], o['position_y'], o['orientation_z']
        state = np.array([x, y, th])
        X = self.rollout(state, U)
        intermediate_cost = np.sum((X[:-1, :2] - self._goal) ** 2, axis=(-2, -1))
        # control_cost = .001 * np.sum(U ** 2, axis=(-2, -1))
        terminal_cost = 10. * np.sum((X[-1, :2] - self._goal) ** 2, axis=-1)

        weight = 1e3
        collision_cost = 0.
        if prediction_res:
            for i in range(self._n_steps):
                d = min([np.sum((X[i+1, :2] - p[i]) ** 2) ** .5 if p.shape[0] > i else 1e5 for p in prediction_res.values()])
                collision_cost += weight * (d <= self._d_min)
        return intermediate_cost + terminal_cost + collision_cost

    @staticmethod
    def filter_unsafe_paths(paths, vels, boxes, predictions, confidence_intervals):
        """
        Given a set of  xy-paths and a collection of rectangles, determine if the path intersects with one of the rectangles.
        :param paths: numpy array of shape (# paths, # steps, 2)
        :param boxes: list of rectangles, where each rectangle is defined as (center, size, angle)

        :return: safe paths of shape (# paths, # steps, 2), or None if all paths are unsafe
        """
        ROBOT_RAD = 0.4

        n_paths = paths.shape[0]

        masks = []
        for box in boxes:

            center = box.pos
            sz = np.array([box.w, box.h])
            th = box.rad
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])     # rotate by -th w.r.t. the origin
            lb, ub = -.5 * sz - ROBOT_RAD, .5 * sz + ROBOT_RAD
            # robot's current coordinate frame -> rectangle's coordinate frame
            transformed_paths = (paths[:, 1:, :] - center) @ R      # first state: observed from the system
            # boolean array of shape (# paths, # steps)
            # True = collision
            mask = np.logical_and(np.all(transformed_paths <= ub, axis=-1), np.all(transformed_paths >= lb, axis=-1))
            masks.append(mask)
        masks = np.array(masks)

        mask_union_per_point = np.sum(masks, axis=0, dtype=bool)

        mask_union_per_path = np.sum(mask_union_per_point, axis=-1)

        mask_p_per_path = np.zeros((n_paths,), dtype=bool)
        for obj_id, prediction in predictions.items():
            obj_mask = np.any(np.sum((paths[:, 1:, :] - prediction) ** 2, axis=-1) < (ROBOT_RAD + .1 / np.sqrt(2.) + confidence_intervals) ** 2, axis=-1)

            mask_p_per_path += obj_mask

        # True = no collision
        mask_final = np.logical_and(np.logical_not(mask_union_per_path), np.logical_not(mask_p_per_path))
        if np.any(mask_final):
            return paths[mask_final], vels[mask_final]
        else:
            # print('no safe paths found')
            return None, None

    def generate_paths(
            self,
            pos_x,
            pos_y,
            orientation_z,
            n_skip=10
    ):
        """
        Generate multiple paths starting at (x, y, theta)
        """

        # TODO: Employing pruning techniques would reduce the number of the paths, but would be also challenging to optimize...
        # TODO: use numba?
        # physical parameters
        dt = self._dt
        # velocity & acceleration ranges
        MAX_LINEAR_X = .8
        MIN_LINEAR_X = -.8
        MAX_ANGULAR_Z = .7
        MIN_ANGULAR_Z = -.7

        linear_xs = np.array([MIN_LINEAR_X, .0, MAX_LINEAR_X])
        angular_zs = np.array([MIN_ANGULAR_Z, .0, MAX_ANGULAR_Z])

        n_points = linear_xs.size * angular_zs.size

        linear_xs, angular_zs = np.meshgrid(linear_xs, angular_zs)

        linear_xs = np.reshape(linear_xs, newshape=(-1,))
        angular_zs = np.reshape(angular_zs, newshape=(-1,))

        # (# grid points, 2)
        # velocity_profile = np.stack((linear_xs, angular_zs), axis=0)

        n_decision_epochs = self._n_steps // n_skip

        # profiles = [velocity_profile for _ in range(n_decision_epochs)]

        # n_paths = n_points ** n_decision_epochs

        state_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self._n_steps+1,)
        x = np.zeros(state_shape)
        y = np.zeros(state_shape)
        th = np.zeros(state_shape)

        # state initialization
        x[..., 0] = pos_x
        y[..., 0] = pos_y
        th[..., 0] = orientation_z

        control_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self._n_steps,)
        v = np.zeros(control_shape)
        w = np.zeros(control_shape)

        for e in range(n_decision_epochs):
            augmented_shape = [1] * n_decision_epochs
            augmented_shape[e] = -1
            v_epoch = linear_xs.reshape(augmented_shape)
            w_epoch = angular_zs.reshape(augmented_shape)
            for t in range(e * n_skip, (e + 1) * n_skip):
                v[..., t] = v_epoch
                w[..., t] = w_epoch

                x[..., t + 1] = x[..., t] + dt * v_epoch * np.cos(th[..., t])
                y[..., t + 1] = y[..., t] + dt * v_epoch * np.sin(th[..., t])
                th[..., t + 1] = th[..., t] + dt * w_epoch

        x = np.reshape(x, (-1, self._n_steps+1))
        y = np.reshape(y, (-1, self._n_steps+1))
        th = np.reshape(th, (-1, self._n_steps+1))
        v = np.reshape(v, (-1, self._n_steps))
        w = np.reshape(w, (-1, self._n_steps))

        return np.stack((x, y, th), axis=-1), np.stack((v, w), axis=-1)
