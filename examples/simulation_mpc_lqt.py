import argparse
import scipy
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
# import torch
from copy import deepcopy

from canvas.controllers import BaseMPC

from canvas.datasets import RegisteredDatasets
from canvas.envs.env import Environment
from canvas.conformal_predictors.scores import ActionDivergenceScoreFunction, PlanningRegretScoreFunction, PositionalDisplacementScoreFunction
from canvas.conformal_predictors.hindsight_scores import HindsightActionDivergenceScoreFunction, HindsightPlanningRegretScoreFunction, HindsightPositionalDisplacementScoreFunction
from canvas.conformal_predictors import LinearQuantileTracker
from canvas.competency_indices.core import ConformalizedCompetencyIndex, HindsightCompetencyIndex

from canvas.predictors import Predictors


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
        'snu-asri': {'init_robot_state': state_dict_from_vec(np.array([0., 0., 0.])), 'goal_pos': np.array([6., -5.]), 't_begin': 100, 't_end': 250}

    }

    # Predictor horizon
    prediction_horizon = 16
    history_len = 8

    env = Environment(
        dataset=dataset,
        **scenario_configs[dataset_name],
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        path_to_frames='/media/sju5379/F6340D35340CF9FF/euped_assets/frames',
        # directory from which the parsed frames are loaded
        path_to_save='./viz_mpc_{}'.format(dataset_name)  # directory to save the visualization result
    )

    # -----------------------------
    # GLOBAL BUFFERS (logging)
    # -----------------------------
    # TODO: add a logger to manage these

    for _ in range(num_iter):

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

        mpc = BaseMPC(prediction_horizon=prediction_horizon, dt=env.dt, goal=env.goal, d_min=d_min, geometry=env.geometry, use_ipopt=False)

        # ---- CP module (updated once per frame) ----

        max_ratio = 100.

        cp_params = {'target_miscoverage_level': .4, 'step_size': .02, 'delay': prediction_horizon, 'sample_size': 6}

        score_ftn_pd = PositionalDisplacementScoreFunction(prediction_len=prediction_horizon, step=6, clip=max_ratio)
        conformal_predictor_pd = LinearQuantileTracker(**cp_params)

        score_ftn_ad = ActionDivergenceScoreFunction(prediction_len=prediction_horizon, clip=max_ratio)
        conformal_predictor_ad = LinearQuantileTracker(**cp_params)


        score_ftn_pr = PlanningRegretScoreFunction(prediction_len=prediction_horizon, clip=max_ratio)
        conformal_predictor_pr = LinearQuantileTracker(**cp_params)

        indices = ConformalizedCompetencyIndex(prefix_len=env.t_begin, momentum=0.7)
        indices.register(score_ftn_pd, conformal_predictor_pd, name='PD')
        indices.register(score_ftn_ad, conformal_predictor_ad, name='AD')
        indices.register(score_ftn_pr, conformal_predictor_pr, name='PR')

        # hindsight competency indices (for comparative evaluation)

        h_score_ftn_pd = HindsightPositionalDisplacementScoreFunction(prediction_len=prediction_horizon, step=6, clip=max_ratio)
        h_score_ftn_ad = HindsightActionDivergenceScoreFunction(prediction_len=prediction_horizon, clip=max_ratio)
        h_score_ftn_pr = HindsightPlanningRegretScoreFunction(prediction_len=prediction_horizon, clip=max_ratio)

        h_indices = HindsightCompetencyIndex(momentum=0.7)
        h_indices.register(h_score_ftn_pd, name='PD')
        h_indices.register(h_score_ftn_ad, name='AD')
        h_indices.register(h_score_ftn_pr, name='PR')

        obs, simulation_info = env.reset()
        truncated = False

        frame = 0

        while not truncated:
            print('[frame {}]'.format(env.timestep), end='')
            # simulation loop
            # --------- Predictor (once per frame) ---------

            prediction_res = prediction_model(obs['non-ego'])
            prediction_res_base = prediction_model_baseline(obs['non-ego'])

            indices.update(obs)

            # ACI -> competency idx computation
            if frame >= prediction_horizon:
                indices.forward()
            else:
                indices.pad(0.5)

            mpc_base = deepcopy(mpc)
            u, controller_info = mpc(obs, prediction_res)
            u2, controller_info2 = mpc_base(obs, prediction_res_base)

            _, controller_info_gt = mpc(obs, simulation_info['future'])

            h_indices.forward(
                {
                    'obs': obs,
                    'controller': deepcopy(mpc),
                    'action': u,
                    'action_base': u2,
                    'U': controller_info['U'],
                    'U_base': controller_info2['U'],
                    'prediction': prediction_res,
                    'prediction_base': prediction_res_base,
                    'context': {},
                    'future': simulation_info['future']
                }
            )

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

            X = controller_info['X'][:, :2]
            X_gt = controller_info_gt['X'][:, :2]
            X_base = controller_info2['X'][:, :2]

            if visualize:
                c = indices.get_history(name='PR')
                hc = h_indices.get_history(name='PR')
                # TODO: manage by dictionary, or define a class registering extra trajectories to an existing fig
                fig, ax = env.render(c=c, hc=hc, open_loop=X, open_loop_gt=X_gt, open_loop_base=X_base)
                fig.tight_layout()
                fig.savefig(os.path.join('./viz_mpc_{}'.format(dataset_name), '{:03d}.pdf'.format(env.timestep)), bbox_inches='tight',
                            pad_inches=0)
                plt.close()

            obs, terminated, truncated, simulation_info = env.step(u)

            # --------- Goal check ---------
            if terminated:
                print('[frame {}] Goal reached!'.format(frame))
                break

            frame += 1

        print('avg. competency index:', indices.get_average_values())

        print('coverage rate:', indices.get_coverage_rate())
        print('avg. h-competency index:', h_indices.get_average_values())
        print('max. score:', h_indices.get_max_scores())

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.set_xlim(env.t_begin+prediction_horizon, env.t_begin+frame)
        ax.set_ylim(-0.01, 1.01)
        ax.grid(True)

        colors = {
            'PD': '#008080',
            'AD': '#8f00ff',
            'PR': '#808000'
        }
        for name in ['PD', 'PR']:
            c = indices.get_history(name=name)
            hc = h_indices.get_history(name=name)

            ax.plot(c, label='predicted', linewidth=4, color=colors[name])
            # ax.fill_between(x=np.arange(frame), y1=c, y2=1., color=colors[name], alpha=0.2)
            ax.plot(np.arange(env.t_begin, env.t_begin+hc.shape[0]), hc, label='hindsight', linewidth=4, color=colors[name], linestyle='dashed', alpha=0.4)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(r'$t$', fontsize=20)
        ax.set_ylabel('competency index', fontsize=20)
        # ax.legend(fontsize=20, ncols=3, loc='lower left', bbox_to_anchor=(0, 1., 1., 0.2), mode='expand')
        ax.legend(fontsize=20)
        fig.tight_layout()
        fig.savefig('indices.pdf')

        plt.close()

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.set_xlim(prediction_horizon, frame)
        ax.grid(True)

        for name in ['PD', ]:
            # c = indices.get_history(name=name)
            hc = h_indices.get_score_history(name=name)
            # ax.plot(c, label=name, linewidth=4, color=colors[name])
            # ax.fill_between(x=np.arange(len(c)), y1=c, y2=1., color=colors[name], alpha=0.2)
            ax.plot(hc, label='predicted', linewidth=4, color=colors[name], linestyle='dashed', alpha=0.4)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(r'$t$', fontsize=20)
        ax.set_ylabel('competency index', fontsize=20)
        ax.legend(fontsize=20, ncols=3, loc='lower left', bbox_to_anchor=(0, 1., 1., 0.2), mode='expand')
        fig.tight_layout()
        fig.savefig('scores.pdf')

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
