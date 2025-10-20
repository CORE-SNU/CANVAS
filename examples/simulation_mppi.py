import argparse
import numpy as np
import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch
from pytorch_mppi import mppi
import pytorch_seed
from arm_pytorch_utilities import linalg, handle_batch_input, sort_nicely, cache

from canvas.datasets import get_dataset_spec
from canvas.datasets import RegisteredDatasets
from canvas.envs.env_new import Environment
from canvas import AdaptiveConformalPredictionModule, Predictors, region_to_box

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""


def compute_mppi_state(obs, p_dict, prediction_horizon):

    x, y, th = obs['ego']['position_x'], obs['ego']['position_y'], obs['ego']['orientation_z']
    non_ego = obs['non-ego']        # observed trajectories of active non-ego agents
    # If there is no non-ego agent, set the min. distance to +inf.
    d0 = min(((x - h[-1, 0]) ** 2 + (y - h[-1, 1]) ** 2) ** .5 for h in non_ego.values()) if non_ego else 1e5
    state = [x, y, th, d0]

    if p_dict:
        for i in range(prediction_horizon):
            d = min(((x - p[i, 0]) ** 2 + (y - p[i, 1]) ** 2) ** .5 for p in p_dict.values())
            state.append(d)
    else:
        # no prediction made
        state += prediction_horizon * [1e5]
    return np.array(state)

def main(goal_x, goal_y, num_iter, dataset_name, predictor):

    init_robot_pose = {
        'position_x': 12.,
        'position_y': 5.,
        'orientation_z': np.pi
    }
    goal_pos = np.array([goal_x, goal_y])
    t_begin = 40
    t_end = 200

    dataset = RegisteredDatasets[dataset_name]

    # Predictor horizon
    prediction_horizon = 12
    history_len = 8

    env = Environment(
        dataset=dataset,
        init_robot_state=init_robot_pose,
        goal_pos=goal_pos,
        t_begin=t_begin,
        t_end=t_end,
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        path_to_frames='/media/sju5379/F6340D35340CF9FF/euped_assets/frames',
        # directory from which the parsed frames are loaded
        path_to_save='./viz_mppi_example'  # directory to save the visualization result
    )

    # -----------------------------
    # GLOBAL BUFFERS (logging)
    # -----------------------------
    # TODO: add a logger to manage these

    for times in range(num_iter):
        print("==================================")
        print("SIMULATION PIPELINE Started")

        # your predictor goes here
        obj_predictor = Predictors(
            chosen_predictor=predictor,
            prediction_len=prediction_horizon,
            history_len=history_len,
            dt=env.dt,
            dataset=dataset_name,
            device='cpu'
        )

        # your controller goes here

        def unicycle_dynamics(state, action):
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

            x_next = x + env.dt * (v * torch.cos(th))
            y_next = y + env.dt * (v * torch.sin(th))
            th_next = th + env.dt * w

            ego_next = torch.stack((x_next, y_next, th_next), dim=-1)
            d_next = torch.cat((d[:, 1:], d[:, -1:]), dim=-1)
            return torch.cat((ego_next, d_next), dim=-1)

        ROBOT_RAD = .4
        d_min = ROBOT_RAD + .1 / np.sqrt(2.)

        def running_cost(state, action):
            x, y = state[..., 0], state[..., 1]
            goal_cost = (x - goal_x) ** 2 + (y - goal_y) ** 2
            d = state[..., 3]     # dist_0[k]
            collision_cost = torch.where(d <= d_min, 1., 0.)    # binary variables indicating collisions
            weight = 1e-3   # magnitude of the cost
            return goal_cost + weight * collision_cost

        def terminal_cost(state, action):
            x, y = state[..., -1, 0], state[..., -1, 1]
            d = state[..., -1, 3]  # dist_0[k]
            goal_cost = (x - goal_x) ** 2 + (y - goal_y) ** 2

            collision_cost = torch.where(d <= d_min, 1., 0.)
            weight = 1e-3
            return 10. * goal_cost + weight * collision_cost

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.double
        pytorch_seed.seed(2)

        mppi_params = {
            "num_samples": 500,
            "horizon": 12,
            "noise_mu": torch.zeros(2, dtype=dtype, device=device),
            "noise_sigma": torch.diag(torch.tensor([1., 1.], dtype=dtype, device=device)),
            "u_max": torch.tensor([.8, .7], dtype=dtype, device=device),
            "terminal_state_cost": terminal_cost,
            "lambda_": 1,
            "device": device
        }

        kmppi = mppi.KMPPI(
            unicycle_dynamics, running_cost, 3+1+prediction_horizon,
            **mppi_params,
            kernel=mppi.RBFKernel(sigma=2),
            num_support_pts=5
        )

        # ---- CP module (updated once per frame) ----
        max_interval_lengths = 0.3 * env.dt * np.arange(1, prediction_horizon + 1)
        offline_calibration_set = {i: [] for i in range(prediction_horizon)}
        cp_module = AdaptiveConformalPredictionModule(
            target_miscoverage_level=0.2,
            step_size=0.05,
            n_scores=prediction_horizon,
            max_interval_lengths=max_interval_lengths,
            sample_size=20,
            offline_calibration_set=offline_calibration_set
        )

        obs, simulation_info = env.reset()
        truncated = False

        frame = 0

        kmppi.reset()

        while not truncated:
            # simulation loop

            # --------- Predictor (once per frame) ---------
            prediction_res = obj_predictor(obs['non-ego'])

            num_refinement_steps = 1
            u = None

            state = compute_mppi_state(obs, prediction_res, prediction_horizon=prediction_horizon)
            for k in range(num_refinement_steps):
                last_refinement = k == num_refinement_steps - 1
                u = kmppi.command(state, shift_nominal_trajectory=last_refinement)

            state_torch = torch.tensor(state).to(device)
            rollout = kmppi.get_rollouts(state_torch)
            rollout = rollout[0]
            # here we evaluate on the rollout MPPI cost of the resulting trajectories
            # alternative costs for tuning the parameters are possible, such as just considering terminal cost

            obs, terminated, truncated, simulation_info = env.step(u)
            fig, ax = env.render(open_loop=rollout[:, :2])

            ax.legend()
            fig.savefig(os.path.join('./viz_mppi_example', '{:03d}.png'.format(env.timestep)), bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            # --------- Goal check ---------
            if terminated:
                print('[frame {}] Goal reached!'.format(frame))
                break

            frame += 1

    return


if __name__ == "__main__":
    # TODO: write -h?
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="zara1")
    parser.add_argument('--predictor', type=str, default="traj")

    parser.add_argument('--goal_x', type=float, default=3.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=6.0)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument('--video_fps', type=float, default=2.5)

    parser.add_argument("--overlay", type=bool, default=True,
                        help="Use homography to draw history/GT/prediction on extracted real frames")
    parser.add_argument("--frame_offset", type=int, default=40,
                        help="Align sim time to real frames (index shift)")
    parser.add_argument("--extracted_fps", type=float, default=2.5,
                        help="FPS used by video_parser.py to extract frames")
    parser.add_argument("--output_fps", type=float, default=10.0,
                        help="Output MP4 FPS; defaults to extracted_fps")
    parser.add_argument("--max_ped", type=float, default=3.0,
                        help="Max pedestrians to consider (others ignored)")
    parser.add_argument("--cont_CI", type=str, default="traj",
                        help="Continuous CI type: traj, control, obj, ctrl_cost to map on video.")
    parser.add_argument("--CI_t", type=int, default=3,
                        help="CI to use for pred.")
    args = parser.parse_args()
    main(args.goal_x, args.goal_y, args.num_iter, args.dataset, args.predictor)
