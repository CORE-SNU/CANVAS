import numpy as np
from scipy.special import softmax


class MPPI:
    def __init__(self, n_steps: int, dt: float, epsilon: float, control_weight, n_paths, collision_weight, gamma):

        assert n_steps > 0 and dt > 0.

        self._n_steps = n_steps

        self._dt = dt
        # dx_t = (f(x_t) + G(x_t) u_t) dt + epsilon dB_t
        self._epsilon = epsilon     # std. of the diffusion

        self._n_paths = n_paths

        self._ctrl_weight = control_weight

        self._collision_weight = collision_weight

        # cost function L(x, u) = q(x) + 1/2 u^T R u
        # where R = (2 * ctrl_weight) I_2

        # R = lambda G^T \Sigma^{-1} G
        self._lambda = control_weight * (2. * epsilon ** 2)

        self._u_seq = np.zeros((n_steps, 2))

        self._gamma = gamma     # step size

    def _shift_ctrl_seq(self):
        """
        apply the shift operator to the control input sequence
        """
        u_seq = np.zeros_like(self._u_seq)
        u_seq[:-1] = self._u_seq[1:]
        u_seq[-1] = self._u_seq[-1]
        return


    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, goal, history):

        x_seq, w_seq = self.forward(pos_x, pos_y, orientation_z)
        delta_u = self.compute_delta_u(x_seq, self._u_seq, w_seq, goal, predictions)
        self.update_u_seq(delta_u)

        u0 = np.copy(self._u_seq[0])
        self._shift_ctrl_seq()
        return u0

    def forward(self, pos_x, pos_y, orientation_z):
        """
        forward propagation of the dynamics

        dx_t = (f(x_t) + G(x_t) u_t) dt + epsilon dB_t
        """

        # white noise sequence; shape: (prediction horizon, # paths, state dim.)
        w_seq = self._epsilon * np.random.randn(self._n_steps, self._n_paths, 3)

        x_seq = np.zeros((self._n_steps+1, self._n_paths, 3))

        # initialize: i = 0
        # (x, y, \theta)
        x_seq[0, ...] = np.array([pos_x, pos_y, orientation_z])

        # propagation loop: 1 <= i <= N, where N: prediction horizon
        for i in range(self._n_steps):
            th = x_seq[i, :, -1]        # (# paths,)
            c, s = np.cos(th), np.sin(th)
            v, w = self._u_seq[i]
            Gu_i = np.stack((v*c, v*s, np.tile(w, self._n_paths)), axis=-1)     # (# paths, 3)

            x_seq[i+1, :] = x_seq[i, :] + self._dt * Gu_i + (self._dt ** .5) * w_seq[i, ...]

        return x_seq, w_seq

    def evaluate_costs(self, x_seq, u_seq, goal, predictions):
        # intermediate state cost q(x_i)
        # shape (# paths,)
        intermediate_cost = np.sum((x_seq[:-1] - goal) ** 2, axis=(0, -1))

        # sum of quadratic control costs 1/2 u^T R u across prediction horizon
        control_cost = self._ctrl_weight * np.sum(u_seq ** 2)

        # terminal state cost phi(x_N)
        # shape (# paths,)
        terminal_cost = 10. * np.sum((x_seq[-1] - goal) ** 2, axis=-1)

        # collision cost (part of the state cost)
        distances = []
        for track_id, prediction in predictions.items():
            # shape = (# paths, # steps)
            distance = np.sum((x_seq[1:] - prediction[:, None, :]) ** 2, axis=-1) ** .5
            distances.append(distance)  # (# tracked, # steps, # paths)
        min_distances = np.min(distances, axis=0)
        collision_cost = -self._collision_weight * np.sum(min_distances, axis=0)
        return intermediate_cost + control_cost + terminal_cost + collision_cost

    def compute_delta_u(self, x_seq, u_seq, w_seq, goal, predictions):
        costs = self.evaluate_costs(x_seq=x_seq, u_seq=u_seq, goal=goal, predictions=predictions)
        weights = softmax(-costs / self._lambda)        # (# paths,)

        weights /= self._dt ** .5

        delta_u = np.sum(weights[None, :, None] * w_seq, axis=1)
        return delta_u

    def update_u_seq(self, delta_u):
        self._u_seq += self._gamma * delta_u