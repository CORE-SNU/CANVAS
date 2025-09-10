import numpy as np


class SamplingBasedMPC:
    def __init__(self, n_paths=200, n_steps=20, dt=0.1):
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._dt = dt

    def __call__(self, pos_x, pos_y, orientation_z, linear_x, angular_z, boxes, predictions, confidence_intervals, goal):
        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, linear_x, angular_z)
        # paths, vels = self.generate_paths_wheel_vel(pos_x, pos_y, orientation_z, linear_x, angular_z)
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes, predictions, confidence_intervals)
        if safe_paths is None:
            # print('MPC infeasible')
            return None, {'feasible': False}
        else:
            path, vel = self.score_paths(safe_paths, vels, goal)
            info = {
                'feasible': True,
                'candidate_paths': paths,
                'safe_paths': safe_paths,
                'final_path': path
            }
            return vel[1], info

    @staticmethod
    def score_paths(paths, vels, goal):
        intermediate_cost = np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2,-1))
        control_cost = .001 * np.sum(vels ** 2, axis=(-2, -1))
        terminal_cost = 10. * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        minimum_cost = np.argmin(intermediate_cost + control_cost + terminal_cost)
        return paths[minimum_cost], vels[minimum_cost]

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
            transformed_paths = (paths - center) @ R
            # boolean array of shape (# paths, # steps)
            # True = collision
            mask = np.logical_and(np.all(transformed_paths <= ub, axis=-1), np.all(transformed_paths >= lb, axis=-1))
            masks.append(mask)

        mask_union_per_point = np.sum(masks, axis=0, dtype=bool)
        print(mask_union_per_point)
        mask_union_per_path = np.sum(mask_union_per_point, axis=-1)

        mask_p_per_path = np.zeros((n_paths,), dtype=bool)
        for obj_id, prediction in predictions.items():
            obj_mask = np.any(np.sum((paths - prediction) ** 2, axis=-1) < (ROBOT_RAD + .1 / np.sqrt(2.) + confidence_intervals) ** 2, axis=-1)

            mask_p_per_path += obj_mask

        # True = no collision
        mask_final = np.logical_and(np.logical_not(mask_union_per_path), np.logical_not(mask_p_per_path))
        if np.any(mask_final):
            return paths[mask_final], vels[mask_final]
        else:
            # print('no safe paths found')
            return None, None

    def generate_paths_wheel_vel(
            self,
            pos_x,
            pos_y,
            orientation_z,
            linear_x,
            angular_z
    ):
        """
        realization of diagonal input constraints for (v, w) by considering the left and right wheels separately
        """
        # TODO: use numba (but the current version is fast enough...)

        # distance between wheels
        v_left = linear_x - 4. * angular_z
        v_right = linear_x + 4. * angular_z

        # physical parameters
        dt = self._dt
        # velocity & acceleration ranges
        MAX_LINEAR_X = 1.
        MIN_LINEAR_X = -1.
        MAX_LINEAR_ACC_X = 1.
        MIN_LINEAR_ACC_X = -1.

        n_paths = self._n_paths
        n_steps = self._n_steps

        # a: linear acc / alpha: angular acc
        a_left = np.random.uniform(low=MIN_LINEAR_ACC_X, high=MAX_LINEAR_ACC_X, size=(n_paths, n_steps-1))
        a_right = np.random.uniform(low=MIN_LINEAR_ACC_X, high=MAX_LINEAR_ACC_X, size=(n_paths, n_steps-1))
        # linear & angular velocity

        v_left_seq = np.zeros((n_paths, n_steps))
        v_right_seq = np.zeros((n_paths, n_steps))

        v = np.zeros((n_paths, n_steps))
        w = np.zeros((n_paths, n_steps))

        v_left_seq[:, 0] = v_left
        v_right_seq[:, 0] = v_right

        v[:, 0] = linear_x
        w[:, 0] = angular_z


        # pose in 2d
        # initialize with the current measurements
        x = pos_x * np.ones((n_paths, n_steps))
        y = pos_y * np.ones((n_paths, n_steps))
        th = orientation_z * np.ones((n_paths, n_steps))

        # roll-out
        for t in range(n_steps-1):
            v_left_seq[:, t+1] = np.clip(v_left_seq[:, t] + dt * a_left[:, t], a_min=MIN_LINEAR_X, a_max=MAX_LINEAR_X)
            v_right_seq[:, t+1] = np.clip(v_right_seq[:, t] + dt * a_right[:, t], a_min=MIN_LINEAR_X, a_max=MAX_LINEAR_X)

            v[:, t+1] = (v_left_seq[:, t+1] + v_right_seq[:, t+1]) / 2.
            w[:, t+1] = (v_right_seq[:, t+1] - v_left_seq[:, t+1] / .5)
            x[:, t+1] = x[:, t] + dt * v[:, t] * np.cos(th[:, t])
            y[:, t+1] = y[:, t] + dt * v[:, t] * np.sin(th[:, t])
            # TODO: normalize
            th[:, t+1] = th[:, t] + dt * w[:, t]
        return np.stack((x, y), axis=-1), np.stack((v, w), axis=-1)

    def generate_paths(
            self,
            pos_x,
            pos_y,
            orientation_z,
            linear_x,
            angular_z
    ):
        """
        Generate multiple paths starting at (x, y, theta) = (0, 0, 0)
        """
        # TODO: use numba (but the current version is fast enough...)
        # physical parameters
        dt = self._dt
        # velocity & acceleration ranges
        MAX_LINEAR_X = 1.
        MIN_LINEAR_X = 0.
        MAX_ANGULAR_Z = 0.7
        MIN_ANGULAR_Z = -0.7
        MAX_LINEAR_ACC_X = .3
        MIN_LINEAR_ACC_X = -.3
        MAX_ANGULAR_ACC_Z = .5
        MIN_ANGULAR_ACC_Z = -.5

        n_paths = self._n_paths
        n_steps = self._n_steps

        # a: linear acc / alpha: angular acc
        a = np.random.uniform(low=MIN_LINEAR_ACC_X, high=MAX_LINEAR_ACC_X, size=(n_paths, n_steps-1))
        alpha = np.random.uniform(low=MIN_ANGULAR_ACC_Z, high=MAX_ANGULAR_ACC_Z, size=(n_paths, n_steps-1))
        # linear & angular velocity
        v = np.zeros((n_paths, n_steps))
        w = np.zeros((n_paths, n_steps))

        v[:, 0] = linear_x
        w[:, 0] = angular_z

        # pose in 2d
        # initialize with the current measurements
        x = pos_x * np.ones((n_paths, n_steps))
        y = pos_y * np.ones((n_paths, n_steps))
        th = orientation_z * np.ones((n_paths, n_steps))

        # roll-out
        for t in range(n_steps-1):
            v[:, t+1] = np.clip(v[:, t] + dt * a[:, t], a_min=MIN_LINEAR_X, a_max=MAX_LINEAR_X)
            w[:, t+1] = np.clip(w[:, t] + dt * alpha[:, t], a_min=MIN_ANGULAR_Z, a_max=MAX_ANGULAR_Z)
            x[:, t+1] = x[:, t] + dt * v[:, t] * np.cos(th[:, t])
            y[:, t+1] = y[:, t] + dt * v[:, t] * np.sin(th[:, t])
            # TODO: normalize
            th[:, t+1] = th[:, t] + dt * w[:, t]
        return np.stack((x, y), axis=-1), np.stack((v, w), axis=-1)