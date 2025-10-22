import argparse
import numpy as np
import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

from canvas.controllers import KernelMPPI

from canvas.datasets import get_dataset_spec
from canvas.datasets import RegisteredDatasets
from canvas.envs.env_new import Environment
from canvas.conformal_predictors.scores import ActionDivergenceScoreFunction
from canvas.conformal_predictors.aci import DelayedACI
from canvas.conformal_predictors.cp_utils import HistoryBuffer
from canvas.predictors import Predictors

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""


def main(goal_x, goal_y, num_iter, dataset_name, predictor, predictor_base):
    init_robot_pose = {
        'position_x': 12.,
        'position_y': 5.,
        'orientation_z': np.pi
    }
    goal_pos = np.array([goal_x, goal_y])
    t_begin = 1
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

    for _ in range(num_iter):
        print("==================================")
        print("SIMULATION PIPELINE Started")

        # your predictor goes here
        prediction_model = Predictors(
            chosen_predictor=predictor,
            prediction_len=prediction_horizon,
            history_len=history_len,
            dt=env.dt,
            dataset=dataset_name,
            device='cpu'
        )

        prediction_model_baseline = Predictors(
            chosen_predictor=predictor_base,
            prediction_len=prediction_horizon,
            history_len=history_len,
            dt=env.dt,
            dataset=dataset_name,
            device='cpu'
        )


        # your controller goes here

        ROBOT_RAD = .4
        d_min = ROBOT_RAD + .1 / np.sqrt(2.)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mppi_params = {
            "num_samples": 500,
            "noise_mu": torch.zeros(2, dtype=torch.float, device=device),
            "noise_sigma": torch.diag(torch.tensor([1., 1.], dtype=torch.float, device=device)),
            "u_max": torch.tensor([.8, .7], dtype=torch.float, device=device),
            "lambda_": 1,
            "device": device
        }

        kmppi = KernelMPPI(prediction_horizon=prediction_horizon, dt=env.dt, mppi_params=mppi_params, goal=goal_pos, d_min=d_min)

        # ---- CP module (updated once per frame) ----

        max_score = (1.6 ** 2 + 1.4 ** 2) ** .5  # diameter of the action space

        score_ftn = ActionDivergenceScoreFunction(prediction_len=prediction_horizon)

        conformal_predictor = DelayedACI(
            target_miscoverage_level=0.2,
            step_size=0.05,
            delay=prediction_horizon,
            max_score=max_score,
            sample_size=20
        )


        obs, simulation_info = env.reset()
        truncated = False

        competency_indices = t_begin * [.5]

        frame = 0

        while not truncated:
            # simulation loop
            # --------- Predictor (once per frame) ---------

            prediction_res = prediction_model(obs['non-ego'])
            prediction_res_base = prediction_model_baseline(obs['non-ego'])

            score_ftn.update(obs=obs)

            if frame >= prediction_horizon:
                score = score_ftn(obs=obs)
                conformal_predictor.update(score)

                err_ub = conformal_predictor.fit()
                competency_idx = 1. / (1. + err_ub)

            else:
                competency_idx = .5
            competency_indices.append(competency_idx)

            kmppi_base = deepcopy(kmppi)
            u, controller_info = kmppi(obs, prediction_res)
            u2, controller_info2 = kmppi_base(obs, prediction_res_base)

            score_ftn.save_snapshot(
                obs=obs,
                controller=deepcopy(kmppi),
                action=u,
                action_base=u2,
                prediction_res=prediction_res,
                context={}
            )

            obs, terminated, truncated, simulation_info = env.step(u)
            fig, ax = env.render(c=competency_indices, open_loop=controller_info['rollout'][:, :2])

            ax.legend()
            fig.savefig(os.path.join('./viz_mppi_example', '{:03d}.png'.format(env.timestep)), bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            # --------- Goal check ---------
            if terminated:
                print('[frame {}] Goal reached!'.format(frame))
                break

            frame += 1

        print('avg. competency index={}'.format(np.mean(competency_indices)))

    return


if __name__ == "__main__":
    # TODO: write -h?
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="zara1")
    parser.add_argument('--predictor', type=str, default="traj")
    parser.add_argument('--predictor_base', type=str, default="linear")

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
    main(args.goal_x, args.goal_y, args.num_iter, args.dataset, args.predictor, args.predictor_base)
