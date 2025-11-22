import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

from canvas.controllers import BaseMPC

from canvas.datasets import RegisteredDatasets
from canvas.envs.env import Environment
from canvas.conformal_predictors.scores import ActionDivergenceScoreFunction, PlanningRegretScoreFunction, PositionalDisplacementScoreFunction
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

# NEW: now we track ADE / FDE
METRICS = ("ade", "fde")


def state_dict_from_vec(v):
    return {'position_x': v[0], 'position_y': v[1], 'orientation_z': v[2]}


def main(num_iter, dataset_name, predictor, predictor_base, visualize: bool = False):
    dataset = RegisteredDatasets[dataset_name]

    # TODO: snu-asri
    # TODO: manage as a config file?
    scenario_configs = {
        # 'zara1': {'init_robot_pose': np.array([14., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
        'zara1': {'init_robot_state': state_dict_from_vec(np.array([12., 5., np.pi])), 'goal_pos': np.array([3., 1.]), 't_begin': 0, 't_end': 901},
        'zara2': {'init_robot_state': state_dict_from_vec(np.array([1., 6.,0.])), 'goal_pos': np.array([10., 1.]), 't_begin': 0, 't_end': 1051},
        'hotel': {'init_robot_state': state_dict_from_vec(np.array([3., -8., -np.pi / 2])), 'goal_pos': np.array([-0.0, 0.0]), 't_begin': 0, 't_end': 1806},
        'eth': {'init_robot_state': state_dict_from_vec(np.array([-3., 10., np.pi / 2.])), 'goal_pos': np.array([5., 4.0]), 't_begin': 0, 't_end': 1160},
        'univ': {'init_robot_state': state_dict_from_vec(np.array([3.5, 2., np.pi / 4.])), 'goal_pos': np.array([11.5, 8.5]), 't_begin': 0, 't_end': 540},
    }

    # Predictor horizon
    prediction_horizon = 12
    history_len = 8

    env = Environment(
        dataset=dataset,
        **scenario_configs[dataset_name],
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        path_to_frames='/home/snowhan/CANVAS/assets/frames',
        # directory from which the parsed frames are loaded
        path_to_save='./viz_mpc_example/' + dataset_name  # directory to save the visualization result
    )
    # NEW: aggregate across num_iter (outer loop inside main)
    iter_sums = {m: 0.0 for m in METRICS}
    iter_count = 0

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


        # your controller goes here
        ROBOT_RAD = .4
        d_min = ROBOT_RAD + .1 / np.sqrt(2.)

        obs, simulation_info = env.reset()
        truncated = False

        frame = 0

        # NEW: collect ADE/FDE per frame, average at end of run
        ade_vals_frames = []
        fde_vals_frames = []

        while not truncated:
            # simulation loop
            prediction_res = prediction_model(obs['non-ego'])
            ground_truth = simulation_info['future']
            #print(ground_truth, prediction_res)
            # ---------------------------------------------
            # insert ade fde based on prediction res and ground_truth
            try:
                ade_sum = 0.0
                fde_sum = 0.0
                n_agents = 0
                for aid, gt_xy in ground_truth.items():
                    if aid not in prediction_res:
                        continue
                    P = np.asarray(prediction_res[aid])
                    G = np.asarray(gt_xy)
                    if P.ndim < 2 or G.ndim < 2 or P.shape[0] == 0 or G.shape[0] == 0:
                        continue
                    T = min(P.shape[0], G.shape[0])
                    dists = np.linalg.norm(P[:T, :2] - G[:T, :2], axis=1)
                    ade_sum += float(np.mean(dists))
                    fde_sum += float(np.linalg.norm(P[T-1, :2] - G[T-1, :2]))
                    n_agents += 1
                if n_agents > 0:
                    ade_vals_frames.append(ade_sum / n_agents)
                    fde_vals_frames.append(fde_sum / n_agents)
            except Exception:
                pass
            # ---------------------------------------------

            u = [0, 0]
            obs, terminated, truncated, simulation_info = env.step(u)

            frame += 1

        # NEW: average ADE/FDE over frames for this run
        run_ade = float(np.mean(ade_vals_frames)) if len(ade_vals_frames) > 0 else np.nan
        run_fde = float(np.mean(fde_vals_frames)) if len(fde_vals_frames) > 0 else np.nan

        # accumulate across num_iter
        if not np.isnan(run_ade):
            iter_sums["ade"] += run_ade
        if not np.isnan(run_fde):
            iter_sums["fde"] += run_fde
        iter_count += 1

    # return the mean over num_iter (outer loop inside main)
    mean_over_iters = {m: (iter_sums[m] / max(1, iter_count)) for m in METRICS}
    return mean_over_iters


DATASET_CHOICES   = ["eth", "hotel", "univ", "zara1", "zara2"]
#DATASET_CHOICES   = [ "zara1","zara2"]
PREDICTOR_CHOICES = ["koopcast","traj", "linear", "eigen", "SocialVAE", "STGCNN"]
#PREDICTOR_CHOICES = ["traj"]


def _print_metric_table(metric_name, datasets, predictors, metric_matrix):
    """
    metric_matrix: dict[predictor][dataset] -> float (or np.nan)
    """
    title = metric_name.replace('_', ' ').title()
    print(f"\n=== {title} (average) ===")

    # column widths
    first_col_w = max(len("Predictor"), max(len(p) for p in predictors)) if predictors else len("Predictor")
    col_widths = [max(len(ds), 8) for ds in datasets]

    # header
    header = "Predictor".ljust(first_col_w) + " | " + " | ".join(ds.ljust(w) for ds, w in zip(datasets, col_widths))
    print(header)
    print("-" * len(header))

    # rows
    for p in predictors:
        row_vals = []
        for j, ds in enumerate(datasets):
            v = metric_matrix.get(p, {}).get(ds, np.nan)
            cell = f"{v:.3f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else "—"
            row_vals.append(cell.ljust(col_widths[j]))
        row = p.ljust(first_col_w) + " | " + " | ".join(row_vals)
        print(row)


if __name__ == "__main__":
    # TODO: write -h?
    parser = argparse.ArgumentParser(
        description="Run MPC simulation over datasets/predictors."
    )

    # Keep your existing args
    parser.add_argument("--predictor_base", type=str, default="linear")
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--video_fps", type=float, default=2.5)

    args = parser.parse_args()

    # Resolve selections
    datasets   = DATASET_CHOICES
    predictors = PREDICTOR_CHOICES

    # NEW: global aggregation structure for final tables
    # results[metric][predictor][dataset] = float
    results = {m: {p: {ds: np.nan for ds in datasets} for p in predictors} for m in METRICS}

    # Run the grid
    for ds in datasets:
        for pred in predictors:
            mean_vals = main(
                args.num_iter,
                ds,
                pred,
                args.predictor_base,
                visualize=args.visualize
            )
            # store means for final tables
            for m in METRICS:
                results[m][pred][ds] = float(mean_vals[m])

    # NEW: print summary tables after all runs
    for m in METRICS:
        _print_metric_table(m, datasets, predictors, results[m])
