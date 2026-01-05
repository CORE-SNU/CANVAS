import argparse
import scipy
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

from canvas.controllers import BaseMPC, KernelMPPI

from canvas.datasets import RegisteredDatasets
from canvas.envs.env import Environment
from canvas.conformal_predictors.scores import (
    ActionDivergenceScoreFunction,
    PlanningRegretScoreFunction,
    PositionalDisplacementScoreFunction,
)
from canvas.conformal_predictors.hindsight_scores import (
    HindsightActionDivergenceScoreFunction,
    HindsightPlanningRegretScoreFunction,
    HindsightPositionalDisplacementScoreFunction,
)
from canvas.conformal_predictors import LinearQuantileTracker
from canvas.competency_indices.core import (
    ConformalizedCompetencyIndex,
    HindsightCompetencyIndex,
    MovingAverageCompetencyIndex,
)

from canvas.predictors import Predictors


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def state_dict_from_vec(v):
    return {'position_x': v[0], 'position_y': v[1], 'orientation_z': v[2]}


def main(num_iter, dataset_name, predictor, predictor_base,
         visualize: bool = False, controller_type: str = "mpc"):

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
        'snu-asri': {'init_robot_state': state_dict_from_vec(np.array([0., 0., 0.])), 'goal_pos': np.array([6., -5.]), 't_begin': 100, 't_end': 250},
             'snu-asri-ood': {'init_robot_state': state_dict_from_vec(np.array([0., 0., 0.])), 'goal_pos': np.array([6., -5.]), 't_begin': 100, 't_end': 250}
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
        path_to_save='./viz_mpc_{}'.format(dataset_name)  
    )

    run_summaries = []

    # -----------------------------
    # GLOBAL BUFFERS (logging)
    # -----------------------------
    for it in range(num_iter):

        # ----- predictor(s) -----
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

        # ----- controller -----

        ROBOT_RAD = .4
        d_min = ROBOT_RAD + .1 / np.sqrt(2.)

        if controller_type == "mpc":
            # original MPC controller
            mpc = BaseMPC(
                prediction_horizon=prediction_horizon,
                dt=env.dt,
                goal=env.goal,
                d_min=d_min,
                geometry=env.geometry,
                use_ipopt=False
            )
        elif controller_type == "mppi":
            # KernelMPPI controller (same horizon & dt)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mppi_params = {
                "num_samples": 500,
                "noise_mu": torch.zeros(2, dtype=torch.float, device=device),
                "noise_sigma": torch.diag(torch.tensor([1., 1.], dtype=torch.float, device=device)),
                "u_max": torch.tensor([.8, .7], dtype=torch.float, device=device),
                "lambda_": 1,
                "device": device,
            }
            mpc = KernelMPPI(
                prediction_horizon=prediction_horizon,
                dt=env.dt,
                mppi_params=mppi_params,
                goal=env.goal,
                d_min=d_min,
            )
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        # ---- CP module (updated once per frame) ----

        max_ratio = 100.

        cp_params = {
            'target_miscoverage_level': .4,
            'step_size': .02,
            'delay': prediction_horizon,
            'sample_size': 3
        }

        score_ftn_pd = PositionalDisplacementScoreFunction(
            prediction_len=prediction_horizon, step=6, clip=max_ratio
        )
        conformal_predictor_pd = LinearQuantileTracker(**cp_params)

        score_ftn_ad = ActionDivergenceScoreFunction(
            prediction_len=prediction_horizon, clip=max_ratio
        )
        conformal_predictor_ad = LinearQuantileTracker(**cp_params)

        score_ftn_pr = PlanningRegretScoreFunction(
            prediction_len=prediction_horizon, clip=max_ratio
        )
        conformal_predictor_pr = LinearQuantileTracker(**cp_params)

        indices = ConformalizedCompetencyIndex(prefix_len=env.t_begin, momentum=0.7)
        indices.register(score_ftn_pd, conformal_predictor_pd, name='PD')
        indices.register(score_ftn_ad, conformal_predictor_ad, name='AD')
        indices.register(score_ftn_pr, conformal_predictor_pr, name='PR')

        m_indices = MovingAverageCompetencyIndex(prefix_len=env.t_begin, window=3)
        m_indices.register(score_ftn_pd, name='PD')
        m_indices.register(score_ftn_ad, name='AD')
        m_indices.register(score_ftn_pr, name='PR')

        # hindsight competency indices (for comparative evaluation)

        h_score_ftn_pd = HindsightPositionalDisplacementScoreFunction(
            prediction_len=prediction_horizon, step=6, clip=max_ratio
        )
        h_score_ftn_ad = HindsightActionDivergenceScoreFunction(
            prediction_len=prediction_horizon, clip=max_ratio
        )
        h_score_ftn_pr = HindsightPlanningRegretScoreFunction(
            prediction_len=prediction_horizon, clip=max_ratio
        )

        h_indices = HindsightCompetencyIndex(momentum=0.7)
        h_indices.register(h_score_ftn_pd, name='PD')
        h_indices.register(h_score_ftn_ad, name='AD')
        h_indices.register(h_score_ftn_pr, name='PR')

        obs, simulation_info = env.reset()
        truncated = False

        frame = 0

        while not truncated:
            print('[frame {}]'.format(env.timestep), end='')

            # --------- Predictor (once per frame) ---------
            prediction_res = prediction_model(obs['non-ego'])
            prediction_res_base = prediction_model_baseline(obs['non-ego'])

            indices.update(obs)
            m_indices.update(obs)
            # ACI -> competency idx computation
            if frame >= prediction_horizon:
                indices.forward()
                m_indices.forward()
            else:
                indices.pad(0.5)
                m_indices.pad(0.5)

            mpc_base = deepcopy(mpc)

            # controller call differs slightly between MPC and MPPI
            if controller_type == "mpc":
                u, controller_info = mpc(obs, prediction_res)
                u2, controller_info2 = mpc_base(obs, prediction_res_base)
                _, controller_info_gt = mpc(obs, simulation_info['future'])
            else:
                # MPPI: update internal control sequence on the actual control call
                u, controller_info = mpc(obs, prediction_res, change_controller_state=True)
                # baseline uses a copy; no need to change its internal state
                u2, controller_info2 = mpc_base(obs, prediction_res_base)
                # hindsight / GT open-loop: do not modify state
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

            m_indices.save_snapshot(
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
                fig, ax = env.render(
                    c=c,
                    hc=hc,
                    open_loop=X,
                    open_loop_gt=X_gt,
                    open_loop_base=X_base
                )
                fig.tight_layout()
                fig.savefig(
                    os.path.join('./viz_mpc_{}'.format(dataset_name),
                                 '{:03d}.pdf'.format(env.timestep)),
                    bbox_inches='tight',
                    pad_inches=0
                )
                plt.close()

            obs, terminated, truncated, simulation_info = env.step(u)

            # --------- Goal check ---------
            if terminated:
                print('[frame {}] Goal reached!'.format(frame))
                break

            frame += 1

        # ---- metrics for this iteration ----
        avg_ci   = indices.get_average_values()
        cov_rate = indices.get_coverage_rate()
        avg_h_ci = h_indices.get_average_values()
        max_score = h_indices.get_max_scores()

        # original prints (kept)
        print('\navg. competency index:', avg_ci)
        print('coverage rate:', cov_rate)
        print('avg. h-competency index:', avg_h_ci)
        print('max. score:', max_score)

        # store summary for this (dataset, predictor, baseline, controller, iteration)
        run_summaries.append({
            'dataset': dataset_name,
            'predictor': predictor,
            'predictor_base': predictor_base,
            'controller': controller_type,
            'iteration': it + 1,
            'avg_competency_index': avg_ci,
            'coverage_rate': cov_rate,
            'avg_h_competency_index': avg_h_ci,
            'max_score': max_score,
        })

        # === plotting code (unchanged) ===
        fig, ax = plt.subplots(figsize=(15, 5))

        t0 = env.t_begin
        ax.set_xlim(t0, env.t_begin + frame)
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

            ax.plot(c, label=f'predicted {name}', linewidth=4, color=colors[name])
            ax.plot(
                np.arange(env.t_begin, env.t_begin + hc.shape[0]),
                hc,
                label=f'hindsight {name}',
                linewidth=4,
                color=colors[name],
                linestyle='dashed',
                alpha=0.4
            )
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(r'$t$', fontsize=20)
        ax.set_ylabel('competency index', fontsize=20)
        ax.legend(fontsize=20, ncols=4)
        fig.tight_layout()
        fig.savefig('indices.pdf')

        plt.close()

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.set_xlim(prediction_horizon, frame)
        ax.grid(True)

        for name in ['PD']:
            hc = h_indices.get_score_history(name=name)
            ax.plot(hc, label='predicted', linewidth=4, color=colors[name],
                    linestyle='dashed', alpha=0.4)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(r'$t$', fontsize=20)
        ax.set_ylabel('competency index', fontsize=20)
        ax.legend(fontsize=20, ncols=3, loc='lower left',
                  bbox_to_anchor=(0, 1., 1., 0.2), mode='expand')
        fig.tight_layout()
        fig.savefig('scores.pdf')

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.set_ylim(0.)
        ax.grid(True)

        colors = {
            'PD': '#008080',
            'AD': '#8f00ff',
            'PR': '#808000'
        }
        for name in ['PD', 'PR']:
            c = indices.get_coverage(name=name)
            mc = m_indices.get_coverage(name=name)
            c_avg = 1. - np.cumsum(c) / np.arange(1, len(c) + 1)
            mc_avg = 1. - np.cumsum(mc) / np.arange(1, len(mc) + 1)
            ax.set_xlim(0., len(c) - 1)
            ax.plot(c_avg, label=f'{name} (conformalized)', linewidth=4, color=colors[name])
            ax.plot(mc_avg, label=f'{name} (moving avg.)', linewidth=4,
                    color=colors[name], linestyle='dotted')

        ax.axhline(y=.4, linestyle='--', linewidth=2, color='k')
        ax.annotate(text=r'$1-\alpha$', xy=(8., .4), xytext=(10., .35), fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel('CP step', fontsize=20)
        ax.set_ylabel('avg. miscoverage', fontsize=20)
        ax.legend(fontsize=20)
        fig.tight_layout()
        fig.savefig('coverage.pdf')

        plt.close()

    return run_summaries


DATASET_CHOICES   = ['snu-asri', 'snu-asri-ood']
PREDICTOR_CHOICES = ["koopcast", "traj", "linear", "eigen", "SocialVAE", "STGCNN"]


if __name__ == "__main__":
    # TODO: write -h?
    parser = argparse.ArgumentParser()

    # default is "all" now
    parser.add_argument(
        '--dataset',
        type=str,
        default="all",
        choices=DATASET_CHOICES + ["all"],
        help="Dataset name or 'all' to run over all datasets."
    )

    # default is "all" now
    parser.add_argument(
        '--predictor',
        type=str,
        default="all",
        choices=PREDICTOR_CHOICES + ["all"],
        help="Predictor name or 'all' to run over all predictors."
    )


    parser.add_argument(
        '--predictor_base',
        type=str,
        default="linear",
        choices=PREDICTOR_CHOICES,
        help="Baseline predictor to compare against."
    )

    # controller selector
    parser.add_argument(
        '--controller',
        type=str,
        default="both",
        choices=["mpc", "mppi", "both"],
        help="Controller type: 'mpc', 'mppi', or 'both' to run both controllers."
    )

    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video_fps', type=float, default=2.5)

    args = parser.parse_args()

    if args.dataset == "all":
        datasets = DATASET_CHOICES
    else:
        datasets = [args.dataset]

    if args.predictor == "all":
        predictors = PREDICTOR_CHOICES
    else:
        predictors = [args.predictor]

    if args.controller == "both":
        controllers = ["mpc", "mppi"]
    else:
        controllers = [args.controller]

    all_summaries = []

    for d in datasets:
        for p in predictors:
            if p == args.predictor_base:
                continue

            for ctrl in controllers:
                print(f"\n=== Running dataset={d}, predictor={p}, baseline={args.predictor_base}, controller={ctrl} ===\n")
                run_summaries = main(
                    num_iter=args.num_iter,
                    dataset_name=d,
                    predictor=p,
                    predictor_base=args.predictor_base,
                    visualize=args.visualize,
                    controller_type=ctrl,
                )
                all_summaries.extend(run_summaries)

    # ---- final summary over all runs ----
    print("\n\n========== SUMMARY OVER ALL RUNS ==========")
    for s in all_summaries:
        print(
            f"[dataset={s['dataset']}, predictor={s['predictor']}, "
            f"baseline={s['predictor_base']}, controller={s['controller']}, "
            f"iter={s['iteration']}]"
        )
        print("  avg. competency index:", s['avg_competency_index'])
        print("  coverage rate:", s['coverage_rate'])
        print("  avg. h-competency index:", s['avg_h_competency_index'])
        print("  max. score:", s['max_score'])
        print("----------------------------------------")
