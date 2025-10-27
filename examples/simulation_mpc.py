import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

from canvas.controllers import BaseMPC

from canvas.datasets import RegisteredDatasets
from canvas.envs.env_new import Environment
from canvas.conformal_predictors.scores_new import ActionDivergenceScoreFunction, PlanningRegretScoreFunction, PositionalDisplacementScoreFunction
from canvas.conformal_predictors.aci import DelayedACI
from canvas.competency_indices.core import CompetencyIndex

from canvas.predictors import Predictors

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""


def state_dict_from_vec(v):
    return {'position_x': v[0], 'position_y': v[1], 'orientation_z': v[2]}


def main(num_iter, dataset_name, predictor, predictor_base, visualize: bool = False):

    dataset = RegisteredDatasets[dataset_name]

    # TODO: snu-asri
    # TODO: manage as a config file?
    scenario_configs = {
        # 'zara1': {'init_robot_pose': np.array([14., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
        'zara1': {'init_robot_state': state_dict_from_vec(np.array([12., 5., np.pi])), 'goal_pos': np.array([3., 6.]), 't_begin': 1, 't_end': 100},
        'zara2': {'init_robot_state': state_dict_from_vec(np.array([1., 6., 0.])), 'goal_pos': np.array([14., 5.]), 't_begin': 1, 't_end': 200},
        'hotel': {'init_robot_state': state_dict_from_vec(np.array([-1.5, 0., -np.pi / 2])), 'goal_pos': np.array([2., -6.]), 't_begin': 78, 't_end': 200},
        'eth': {'init_robot_state': state_dict_from_vec(np.array([5., 1.0, np.pi / 2.])), 'goal_pos': np.array([3., 10.]), 't_begin': 1, 't_end': 100},
        'univ': {'init_robot_state': state_dict_from_vec(np.array([3.5, 2., np.pi / 4.])), 'goal_pos': np.array([11.5, 8.5]), 't_begin': 1, 't_end': 300},

    }

    # Predictor horizon
    prediction_horizon = 12
    history_len = 8

    env = Environment(
        dataset=dataset,
        **scenario_configs[dataset_name],
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

        mpc = BaseMPC(prediction_horizon=prediction_horizon, dt=env.dt, goal=env.goal, d_min=d_min)

        # ---- CP module (updated once per frame) ----
        max_score_pd = 10.
        score_ftn_pd = PositionalDisplacementScoreFunction(prediction_len=prediction_horizon, step=6)
        conformal_predictor_pd = DelayedACI(
            target_miscoverage_level=0.8,
            step_size=0.05,
            delay=prediction_horizon,
            max_score=max_score_pd,
            sample_size=12
        )


        max_score_ad = (1.6 ** 2 + 1.4 ** 2) ** .5  # diameter of the action space
        score_ftn_ad = ActionDivergenceScoreFunction(prediction_len=prediction_horizon)
        conformal_predictor_ad = DelayedACI(
            target_miscoverage_level=0.8,
            step_size=0.05,
            delay=prediction_horizon,
            max_score=max_score_ad,
            sample_size=12
        )

        max_score_pr = 800.
        score_ftn_pr = PlanningRegretScoreFunction(prediction_len=prediction_horizon)
        conformal_predictor_pr = DelayedACI(
            target_miscoverage_level=0.8,
            step_size=0.05,
            delay=prediction_horizon,
            max_score=max_score_pr,
            sample_size=12
        )

        indices = CompetencyIndex(prefix_len=scenario_configs[dataset_name]['t_begin'])
        indices.register(score_ftn_pd, conformal_predictor_pd, name='positional_displacement')
        indices.register(score_ftn_ad, conformal_predictor_ad, name='action_divergence')
        indices.register(score_ftn_pr, conformal_predictor_pr, name='planning_regret')

        obs, simulation_info = env.reset()
        truncated = False

        frame = 0

        while not truncated:
            # simulation loop
            # --------- Predictor (once per frame) ---------

            prediction_res = prediction_model(obs['non-ego'])
            prediction_res_base = prediction_model_baseline(obs['non-ego'])

            indices.update(obs)

            if frame >= prediction_horizon:
                # ACI -> competency idx computation
                indices.forward()

            else:
                indices.pad(val=0.5)

            mpc_base = deepcopy(mpc)
            u, controller_info = mpc(obs, prediction_res)
            u2, controller_info2 = mpc_base(obs, prediction_res_base)

            indices.save_snapshot(
                {
                    'obs': obs,
                    'controller': deepcopy(mpc),
                    'action': u,
                    'action_base': u2,
                    'U': controller_info['U'],
                    'U_base': controller_info2['U'],
                    'prediction': prediction_res,
                    'prediction_base': prediction_res_base,
                    'context': {}
                }
            )

            obs, terminated, truncated, simulation_info = env.step(u)

            if visualize:
                fig, ax = env.render(c=indices.get_history(name='planning_regret'), open_loop=controller_info['X'][:, :2])
                ax.legend()
                fig.savefig(os.path.join('./viz_mppi_example', '{:03d}.png'.format(env.timestep)), bbox_inches='tight',
                            pad_inches=0)
                plt.close()

            # --------- Goal check ---------
            if terminated:
                print('[frame {}] Goal reached!'.format(frame))
                break

            frame += 1


        print('avg. competency index:', indices.get_average_values())

        fig, ax = plt.subplots()

        ax.set_xlim(0, frame)
        ax.set_ylim(0., 1.)
        ax.grid(True)

        colors = {
            'positional_displacement': '#008080',
            'action_divergence': '#8f00ff',
            'planning_regret': '#808000'
        }
        for name in ['positional_displacement', 'action_divergence', 'planning_regret']:
            c = indices.get_history(name=name)
            ax.plot(c, label=name.replace('_', ' ').title(), linewidth=2, color=colors[name])
        ax.set_xlabel(r'$t$', fontsize=16)
        ax.set_ylabel(r'$\mathcal{L}_t$', fontsize=16)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig('indices.png')

    return


if __name__ == "__main__":
    # TODO: write -h?
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="zara1")
    parser.add_argument('--predictor', type=str, default="traj")
    parser.add_argument('--predictor_base', type=str, default="linear")

    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video_fps', type=float, default=2.5)

    args = parser.parse_args()
    main(args.num_iter, args.dataset, args.predictor, args.predictor_base, visualize=args.visualize)
