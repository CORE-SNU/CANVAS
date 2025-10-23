from pytorch_mppi import mppi as torch_mppi
from functools import partial
import numpy as np
import torch
import pytorch_seed
# from arm_pytorch_utilities import linalg, handle_batch_input, sort_nicely, cache


def unicycle_dynamics(state, action, dt):
    """
    x[k+1] = x[k] + dt * v[k] * cos(th[k])
    y[k+1] = y[k] + dt * v[k] * sin(th[k])
    th[k+1] = th[k] + dt * w[k]

    To additionally account for non-ego dynamic agents, an extra N state variables are added:
    dist_0[k], dist_1[k], ..., dist_N[k]
    Each dist_i[k] represents the minimum distance between the ego-agent and the non-ego agents in the i-th future.
    (Note that only the distances matter when computing the cost function!)
    The dynamics of these variables (whose values are acquired by the prediction model) are simply given as the shift operator:

    dist_i[k+1] = dist_{i+1}[k], i = 0, ..., N-1
    dist_N[k+1] = dist_N[k].

    state: torch tensor of shape (batch size, state dim.)
    action: torch tensor of shape (batch size, action dim.)
    """
    x, y, th = state[..., 0], state[..., 1], state[..., 2]
    v, w = action[..., 0], action[..., 1]

    d = state[..., 3:]    # (batch size, N)

    x_next = x + dt * (v * torch.cos(th))
    y_next = y + dt * (v * torch.sin(th))
    th_next = th + dt * w

    ego_next = torch.stack((x_next, y_next, th_next), dim=-1)
    d_next = torch.cat((d[:, 1:], d[:, -1:]), dim=-1)
    return torch.cat((ego_next, d_next), dim=-1)



def running_cost(state, action, goal_x, goal_y, d_min):
    x, y = state[..., 0], state[..., 1]
    goal_cost = (x - goal_x) ** 2 + (y - goal_y) ** 2
    d = state[..., 3]     # dist_0[k]
    collision_cost = torch.where(d <= d_min, 1., 0.)    # binary variables indicating collisions
    weight = 1e3   # magnitude of the cost
    return goal_cost + weight * collision_cost

def terminal_cost(state, action, goal_x, goal_y, d_min):
    x, y = state[..., -1, 0], state[..., -1, 1]
    d = state[..., -1, 3]  # dist_0[k]
    goal_cost = (x - goal_x) ** 2 + (y - goal_y) ** 2

    collision_cost = torch.where(d <= d_min, 1., 0.)
    weight = 1e3
    return 10. * goal_cost + weight * collision_cost


def compute_mppi_state(obs, p_dict, prediction_horizon):
    x, y, th = obs['ego']['position_x'], obs['ego']['position_y'], obs['ego']['orientation_z']
    non_ego = obs['non-ego']        # observed trajectories of active non-ego agents
    # If there is no non-ego agent, set the min. distance to +inf.
    d0 = min(((x - h[-1, 0]) ** 2 + (y - h[-1, 1]) ** 2) ** .5 for h in non_ego.values()) if non_ego else 1e5
    state = [x, y, th, d0]

    xy = np.array([x, y])

    if p_dict:
        for i in range(prediction_horizon):
            ds = [np.sum((p[i] - xy) ** 2) ** .5 if p.shape[0] > i else 1e5 for p in p_dict.values()]

            state.append(min(ds))
    else:
        # no prediction made
        state += prediction_horizon * [1e5]
    return np.array(state)


class KernelMPPI:
    """
    A wrapper of pytorch_mppi.mppi
    """
    def __init__(self, prediction_horizon, dt, mppi_params, goal, d_min):


        pytorch_seed.seed(2)


        goal_x, goal_y = goal
        self._running_cost = partial(running_cost, goal_x=goal_x, goal_y=goal_y, d_min=d_min)
        self._terminal_cost = partial(terminal_cost, goal_x=goal_x, goal_y=goal_y, d_min=d_min)
        self._dynamics = partial(unicycle_dynamics, dt=dt)

        self.device = mppi_params['device']
        self._prediction_horizon = prediction_horizon
        self._mppi = torch_mppi.KMPPI(
            self._dynamics,
            self._running_cost,
            3+1+prediction_horizon,
            **mppi_params,
            terminal_state_cost=self._terminal_cost,
            horizon=prediction_horizon,
            kernel=torch_mppi.RBFKernel(sigma=2),
            num_support_pts=5
        )
        self._mppi.reset()

        return

    def __call__(self, obs, prediction_res, change_controller_state=False):
        state = compute_mppi_state(obs, prediction_res, prediction_horizon=self._prediction_horizon)
        u = self._mppi.command(state, shift_nominal_trajectory=change_controller_state)

        state_torch = torch.tensor(state).to(self.device)
        rollout = self._mppi.get_rollouts(state_torch)
        X = rollout[0]

        U = self._mppi.U
        cost_to_go = 0.
        for t in range(len(rollout) - 1):
            cost_to_go = cost_to_go + self._running_cost(X[t], U[t])

        cost_to_go = cost_to_go + self._terminal_cost(X, U)
        controller_info = {'X': X, 'U': U, 'cost_to_go': cost_to_go}

        return u.detach().cpu().numpy(), controller_info

    def cost_to_go(self, obs, prediction_res, U):
        state = compute_mppi_state(obs, prediction_res, prediction_horizon=self._prediction_horizon)
        state_torch = torch.tensor(state).to(self.device)
        rollout = self._mppi.get_rollouts(state_torch, U=U)
        X = rollout[0]

        rollout_cost = 0.
        for t in range(len(rollout) - 1):
            rollout_cost = rollout_cost + self._running_cost(X[t], U[t])
        rollout_cost = rollout_cost + self._terminal_cost(X, U)
        return rollout_cost