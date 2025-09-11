import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pathlib
import os
import subprocess
import random
import pickle
import csv

import sys
sys.path.append('/home/snowhan1021/tools_paper/CANavi')
from competency_index import CompetencyIndex
from env import Environment
from detection.detection_utils import Box
from control.grid_solver import GridMPC
# from control.sampling_based_mpc import SamplingBasedMPC
from conformal_prediction.adaptive_cp import AdaptiveConformalPredictionModule

# For predictors
from prediction.linear_predictor import LinearPredictor
from trajectron_predictor import TrajectronPredictor
from koopman.koopy_predictor_justmul import KoopmanPredictor
# from koopman.koopman_predictor_clu_geo import KoopmanPredictor
# from prediction.eigen.eigen_predictor import eigen_predictor

from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""

matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# -----------------------------
# Static map geometry → boxes
# -----------------------------
persistent_static_boxes = []    # Save static object as bounding box
regions = [
    {"name": "glass door below", "xmin": 0.3, "xmax": 5.0, "ymin": -12.0, "ymax": -6.3},
    {"name": "left glass", "xmin": -7.0, "xmax": 0.3, "ymin": -12.0, "ymax": -8.5},
    {"name": "right glass", "xmin": 5.0, "xmax": 13.0, "ymin": -12.0, "ymax": -8.5},
    {"name": "left wall", "xmin": -7.0, "xmax": -2.1, "ymin": -12.0, "ymax": -0.3},
    {"name": "right wall", "xmin": 7.8, "xmax": 13.0, "ymin": -12.0, "ymax": -0.3},
    {"name": "upper-left wall", "xmin": -7.0, "xmax": -1.9, "ymin": 1.1, "ymax": 5.0},
    {"name": "upper wall", "xmin": -0.5, "xmax": 13.0, "ymin": 0.9, "ymax": 5.0},
    {"name": "middle square", "xmin": 2.0, "xmax": 3.4, "ymin": -4.6, "ymax": -1.6},
    {"name": "left cylinder", "xmin": -0.7, "xmax": 0.5, "ymin": -1.5, "ymax": -0.6},
    {"name": "right cylinder", "xmin": 5.3, "xmax": 6.4, "ymin": -1.8, "ymax": -0.8},
]
for region in regions:
    x_center = (region["xmin"] + region["xmax"]) / 2
    y_center = (region["ymin"] + region["ymax"]) / 2
    width = region["xmax"] - region["xmin"]
    height = region["ymax"] - region["ymin"]
    box = Box(
        x=x_center, y=y_center, w=width, h=height, deg=0, rad=0,
        area=width * height,
        vertices=np.array([
            [x_center - width / 2, y_center - height / 2],
            [x_center + width / 2, y_center - height / 2],
            [x_center + width / 2, y_center + height / 2],
            [x_center - width / 2, y_center + height / 2]
        ]),
        resolution=0.001,
        pos=np.array([x_center, y_center])
    )
    persistent_static_boxes.append(box)

# -----------------------------
# Visualization helper (CI labels optional)
# -----------------------------
def save_frame_png(outdir,
                   frame_idx,
                   static_boxes,
                   robot_xy,
                   robot_traj_xy,
                   goal_xy,
                   valid_obs=None,
                   valid_obs_future_true=None,
                   prediction_res=None,
                   r_star=None,
                   steps_to_annotate=(2, 5, 10),
                   annotate_ci=False,   # default off for unified pipeline
                   ci_decimals=2,
                   ci_fontsize=7,
                   max_ci_annotations_per_step=None,
                   xlim=(-2.5, 10.0),  #(-7.5, 13.5)
                   ylim=(-10.0, 2.0)): #(-12.5, 5.5)
    """Draw history / GT future / prediction with static boxes and robot.
       CI circles/text are disabled by default (use annotate_ci=True to enable)."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1) Static boxes — gray
    if static_boxes:
        for b in static_boxes:
            poly = Polygon(b.vertices, closed=True,
                           facecolor='gray', edgecolor='gray',
                           linewidth=1, zorder=0.1)
            ax.add_patch(poly)

    # 2) Robot trajectory / current / goal
    px, py = robot_traj_xy
    if len(px) and len(py):
        ax.plot(px, py, linewidth=2)
    ax.scatter([robot_xy[0]], [robot_xy[1]], marker='o', s=30)
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker='*', s=80)

    # 3) History(8)
    if valid_obs:
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)

    # 4) GT future(12)
    if valid_obs_future_true:
        for _, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], linestyle='--', color='black', linewidth=1)

    # 5) Prediction(12)
    if prediction_res:
        for _, arr in prediction_res.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], color='red', linewidth=1.5)

    # 6) Optional CI annotations (off by default)
    if annotate_ci and (r_star is not None) and (r_star > 0) and valid_obs_future_true and prediction_res:
        common = set(prediction_res.keys()) & set(valid_obs_future_true.keys())
        offsets = {2: (0.05, 0.05), 5: (-0.05, 0.05), 10: (0.05, -0.05)}
        ann_counts = {s: 0 for s in steps_to_annotate}
        for s in steps_to_annotate:
            j = s - 1
            for pid in common:
                pred = np.asarray(prediction_res[pid], dtype=np.float64)
                gt   = np.asarray(valid_obs_future_true[pid], dtype=np.float64)
                if (pred.ndim == 2 and gt.ndim == 2 and pred.shape[1] >= 2 and gt.shape[1] >= 2
                    and len(pred) > j and len(gt) > j):
                    p = pred[j, :2]; g = gt[j, :2]
                    if not (np.isfinite(p).all() and np.isfinite(g).all()):
                        continue
                    err = float(np.linalg.norm(p - g))
                    ax.add_patch(Circle((p[0], p[1]), r_star,
                                        fill=True, edgecolor='none', facecolor='lightgray', zorder=0.5))
                    ax.add_patch(Circle((p[0], p[1]), err,
                                        fill=True, edgecolor='black', facecolor='orange', linewidth=1, zorder=1.0))
                    #ax.scatter([g[0]], [g[1]], marker='x', s=18, color='black', zorder=2.0)
                    if (max_ci_annotations_per_step is None) or (ann_counts[s] < max_ci_annotations_per_step):
                        ci = (r_star - err) / r_star
                        if np.isfinite(ci):
                            dx, dy = offsets.get(s, (0.04, 0.04))
                            ax.text(p[0] + dx, p[1] + dy,
                                    f"CI@t+{s}={ci:.{ci_decimals}f}",
                                    fontsize=ci_fontsize, zorder=3.0)
                            ann_counts[s] += 1

    legend_elements = [
        Line2D([0], [0], color='navy',  lw=1,   linestyle='-',  label='History (8)'),
        Line2D([0], [0], color='black', lw=1,   linestyle='--', label='GT future (12)'),
        Line2D([0], [0], color='red',   lw=1.5, linestyle='-',  label='Prediction (12)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"frame_{frame_idx:05d}.png")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main(goal_x, goal_y, num_iter, r_star):
    # Simulation rates
    dt = 0.10

    # Predictor horizon
    prediction_len = 12

    # Unified R* (kept for future score/CI; not used in controller)
    rstar = r_star
    #steps_to_eval = (2, 5, 10)

    # Output dirs (kept for compatibility)
    #results_dir = pathlib.Path("results")
    #results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # GLOBAL BUFFERS (logging)
    # -----------------------------
    buffer_collision_rate = []
    buffer_infeasible_rate = []
    buffer_avg_minimal_cost = []
    buffer_avg_intermediate_cost = []
    buffer_avg_terminal_cost = []
    buffer_avg_control_cost = []
    buffer_prediction_times = []
    buffer_travel_times = []
    success_count = 0
    buffer_pos_x_result = []
    buffer_pos_y_result = []

    # CI buffers (per-frame)
    buffer_ci_traj_series   = []  # list[np.ndarray], CI of trajectory error (full series)
    buffer_ci_ctrl_series   = []  # list[np.ndarray], CI of control (v,w) plan error (full series)
    buffer_ci_obj           = []  # list[float],      CI of objective error (scalar)
    buffer_ci_ctrl_cost     = []  # list[float],      CI of control-cost error (scalar)

    # CI Instance
    ci_traj     = CompetencyIndex(case="traj",      r_star=rstar, return_type="series")
    ci_ctrl     = CompetencyIndex(case="control",   r_star=rstar, return_type="series")
    ci_obj      = CompetencyIndex(case="obj",       r_star=rstar)            # scalar
    ci_ctrlcost = CompetencyIndex(case="ctrl_cost", r_star=rstar)            # scalar

    for times in range(num_iter):
        print("==================================")
        print("SIMULATION PIPELINE Started")
        frame = 0
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        # --- per-iteration CI buffers (for saving to csv file) ---
        it_ci_traj_series = []   # list[np.ndarray]  (T,) per frame
        it_ci_ctrl_series = []   # list[np.ndarray]  (T,) per frame
        it_ci_obj         = []   # list[float]
        it_ci_ctrl_cost   = []   # list[float]

        buffer_timestamp = []
        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []  # per-frame x within this run
        buffer_pos_y = []
        buffer_intermediate = []
        buffer_terminal = []
        buffer_control = []

        goal = np.array([goal_x, goal_y])

        # ---- Choose predictor ----
        data_dir = "/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_17_Mar_2025_22_52_52lobby_data_ar3"
        # obj_predictor = LinearPredictor(prediction_len=prediction_len, history_len=8, smoothing_factor=0.75, dt=dt)
        # obj_predictor = KoopmanPredictor(prediction_len=prediction_len, data_dir=data_dir, min_samples=100, dt=dt, pattern=r'^.*\d{2}\.npy$')
        obj_predictor = TrajectronPredictor(prediction_len=prediction_len, model_dir=data_dir, device='cpu')  # Trajectron++ predictor

        controller = GridMPC(n_steps=prediction_len, dt=dt)

        # ---- CP module (updated once per frame) ----
        max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1)
        offline_calibration_set = {i: [] for i in range(prediction_len)}
        cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                      step_size=0.05,
                                                      n_scores=prediction_len,
                                                      max_interval_lengths=max_interval_lengths,
                                                      sample_size=20,
                                                      offline_calibration_set=offline_calibration_set)

        begin = time.time()
        init_robot_pose = np.array([0, 0, np.pi / 2.])  # Initial robot position setting
        t_begin = 40
        t_end = 2000
        buffer_vel = []
        done = False

        # iteration output dir for viz
        iter_out_dir = pathlib.Path("viz") / f"iter_{times+1:03d}"
        iter_out_dir.mkdir(parents=True, exist_ok=True)

        environment = Environment(
            filepath=os.path.join('0.npy'),
            dt=dt,
            init_robot_pose=init_robot_pose,
            n_pedestrians=0,  # NOT USED
            t_begin=t_begin,
            t_end=t_end
        )
        position_x, position_y, orientation_z = environment.reset()

        while not done:
            detect_time = time.time()
            linear_x, angular_z = environment.get_velocity()

            # record robot trajectory
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # --------- Observations (history & GT futures) ---------
            observation = environment._get_obs()
            observation_future_true = environment._get_obs_future()

            # Filter valid histories and GT futures (finite & correct shape)
            valid_obs = {}
            valid_obs_future_true = {}
            if isinstance(observation, dict):
                for pid, traj in observation.items():
                    # history
                    try:
                        arr_hist = np.asarray(traj, dtype=np.float64)
                    except (TypeError, ValueError):
                        continue
                    if not (arr_hist.ndim == 2 and arr_hist.shape[0] == 8 and arr_hist.shape[1] >= 2 and np.isfinite(arr_hist[:, :2]).all()):
                        continue
                    valid_obs[pid] = arr_hist

                    # GT future (kept for viz & later scoring; not used by controller here)
                    fut = observation_future_true.get(pid, None) if isinstance(observation_future_true, dict) else None
                    if fut is None:
                        continue
                    arr_fut = np.asarray(fut, dtype=np.float64)
                    if not (arr_fut.ndim == 2 and arr_fut.shape[1] >= 2 and arr_fut.shape[0] >= prediction_len and np.isfinite(arr_fut[:prediction_len, :2]).all()):
                        continue
                    valid_obs_future_true[pid] = arr_fut[:prediction_len, :2]

            # --------- Simple collision check (proximity to last history point) ---------
            dynamic_obs = {}
            if valid_obs:
                dynamic_obs = valid_obs
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
                robot_pos = np.array([position_x, position_y])
                distances = np.sqrt(np.sum((initial_positions - robot_pos) ** 2, axis=1))
                if np.any(distances <= 0.7):
                    print("Collision!")
                    collision_count += int(np.sum(distances <= 0.7))

            # --------- Predictor (once per frame) ---------
            pred_start = time.time()
            prediction_res = obj_predictor(dynamic_obs if dynamic_obs else {})
            pred_time = time.time() - pred_start
            buffer_prediction_times.append(pred_time)

            # --------- CP update (once per frame) ---------
            confidence_intervals = cp_module.update(dynamic_obs, prediction_res if isinstance(prediction_res, dict) else {})

            # --------- Controller (once per frame, with predictions) ---------
            velocity, info, minimum, intermediate, terminal, control, minimal = controller(
                pos_x=position_x,
                pos_y=position_y,
                orientation_z=orientation_z,
                linear_x=linear_x,
                angular_z=angular_z,
                boxes=persistent_static_boxes,
                predictions=prediction_res if isinstance(prediction_res, dict) else {},
                confidence_intervals=confidence_intervals,
                goal=goal
            )

            # For GT(Oracle based) : no status update here, just for get controller input for GT
            velocity_gt, info_gt, minimum_gt, intermediate_gt, terminal_gt, control_gt, minimal_gt = controller(
                pos_x=position_x,
                pos_y=position_y,
                orientation_z=orientation_z,
                linear_x=linear_x,
                angular_z=angular_z,
                boxes=persistent_static_boxes,
                predictions=valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {},
                confidence_intervals=confidence_intervals,
                goal=goal
            )

            # 1) traj CI (series)
            ci_traj_series = ci_traj(
                prediction_res=prediction_res if isinstance(prediction_res, dict) else {},
                gt_future=valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {},
            )
            buffer_ci_traj_series.append(ci_traj_series)
            it_ci_traj_series.append(ci_traj_series)

            # 2) control CI (series)
            ci_ctrl_series = ci_ctrl(
                ctrl_pred=velocity if velocity is not None else [],
                ctrl_gt=velocity_gt if velocity_gt is not None else [],
            )
            buffer_ci_ctrl_series.append(ci_ctrl_series)
            it_ci_ctrl_series.append(ci_ctrl_series)

            # 3) obj CI (scalar) -- uses total objective (minimal vs minimal_gt)
            ci_obj_val = ci_obj(minimal=minimal, minimal_gt=minimal_gt)
            buffer_ci_obj.append(ci_obj_val)
            it_ci_obj.append(ci_obj_val)

            # 4) ctrl_cost CI (scalar) -- compares sums of (intermediate + terminal + control) pred vs gt
            ci_ctrlcost_val = ci_ctrlcost(
                intermediate=intermediate, terminal=terminal, control=control,
                intermediate_gt=intermediate_gt, terminal_gt=terminal_gt, control_gt=control_gt
            )
            buffer_ci_ctrl_cost.append(ci_ctrlcost_val)
            it_ci_ctrl_cost.append(ci_ctrlcost_val)

            '''
            # --------- Visualization (CI labels disabled by default) ---------
            try:
                save_frame_png(
                    outdir=iter_out_dir,
                    frame_idx=frame,
                    static_boxes=persistent_static_boxes,
                    robot_xy=(position_x, position_y),
                    robot_traj_xy=(buffer_pos_x, buffer_pos_y),
                    goal_xy=goal,
                    valid_obs=valid_obs if valid_obs else {},
                    valid_obs_future_true=valid_obs_future_true if valid_obs_future_true else {},
                    prediction_res=prediction_res if isinstance(prediction_res, dict) else {},
                    r_star=rstar,
                    steps_to_annotate=steps_to_eval,
                    annotate_ci=True  # keep False here; enable later if needed
                )
            except Exception as e:
                print(f"[WARN] viz save failed at frame {frame}: {e}")
            '''
            # --------- Feasibility handling ---------
            buffer_infeasibility.append(info.get('feasible', True))
            if not info.get('feasible', True):
                if infeasible_streak < len(buffer_vel):
                    v, w = buffer_vel[infeasible_streak]
                else:
                    v, w = 0.0, 0.0
                print(frame, 'No safe paths found, stopping robot movement for this frame.',
                      position_x, position_y, v, w, time.time() - detect_time)
                environment.step([v, w])
                frame += 1
                infeasible_count += 1
                infeasible_streak += 1
                if infeasible_streak >= max_infeasible_streak:
                    print(frame, 'Infeasible state lasted too long, failed')
                    break
                continue
            else:
                infeasible_streak = 0

            # --------- Goal check ---------
            if np.abs(position_x - goal[0]) < 0.3 and np.abs(position_y - goal[1]) < 0.3:
                print(frame, 'Goal reached!')
                environment.step([0, 0])
                is_success = True
                success_count += 1
                travel_time = time.time() - begin
                buffer_travel_times.append(travel_time)
                buffer_pos_x_result.append(buffer_pos_x)
                buffer_pos_y_result.append(buffer_pos_y)
                buffer_avg_minimal_cost.append(np.sum(minimum_cost) / len(minimum_cost) if minimum_cost else np.nan)
                buffer_avg_intermediate_cost.append(np.sum(buffer_intermediate) / len(buffer_intermediate) if buffer_intermediate else np.nan)
                buffer_avg_terminal_cost.append(np.sum(buffer_terminal) / len(buffer_terminal) if buffer_terminal else np.nan)
                buffer_avg_control_cost.append(np.sum(buffer_control) / len(buffer_control) if buffer_control else np.nan)
                break

            # --------- Apply first control step ---------
            if velocity is not None and len(velocity) > 0:
                cmd_linear_x, cmd_angular_z = velocity[0]
            else:
                cmd_linear_x, cmd_angular_z = 0.0, 0.0
            robot_pose, done = environment.step([cmd_linear_x, cmd_angular_z])
            position_x, position_y, orientation_z = robot_pose
            print(frame, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, time.time() - detect_time)

            # --- Per-frame CI printout ---
            if ci_traj_series.size:
                print(f"[frame {frame}] CI_traj  avg={np.nanmean(ci_traj_series):.3f}  final={ci_traj_series[-1]:.3f}")
            else:
                print(f"[frame {frame}] CI_traj  (empty)")

            if ci_ctrl_series.size:
                print(f"[frame {frame}] CI_ctrl  avg={np.nanmean(ci_ctrl_series):.3f}  final={ci_ctrl_series[-1]:.3f}")
            else:
                print(f"[frame {frame}] CI_ctrl  (empty)")

            print(f"[frame {frame}] CI_obj={ci_obj_val:.3f}  CI_ctrl_cost={ci_ctrlcost_val:.3f}")

            # --------- Accumulate costs ---------
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            frame += 1
            buffer_vel = velocity


        # ---- Iteration-level rates and summaries ----
        buffer_collision_rate.append(collision_count / max(1, frame))
        buffer_infeasible_rate.append(infeasible_count / max(1, frame))

        print("Next : #{}_scenario".format(times + 1))
        print("Collision_rate: ", collision_count / max(1, frame))
        print("Infeasible_rate: ", infeasible_count / max(1, frame))
        if buffer_prediction_times:
            print("Avg_prediction_time: ", np.sum(buffer_prediction_times) / len(buffer_prediction_times))
            print('Variance prediction time', np.var(buffer_prediction_times))
        if is_success and minimum_cost:
            print("Avg_minimal_cost: ", np.sum(minimum_cost) / len(minimum_cost))
            print("Avg_intermediate_cost: ", np.sum(buffer_intermediate) / len(buffer_intermediate) if buffer_intermediate else np.nan)
            print("Avg_terminal_cost: ", np.sum(buffer_terminal) / len(buffer_terminal) if buffer_terminal else np.nan)
            print("Avg_control_cost: ", np.sum(buffer_control) / len(buffer_control) if buffer_control else np.nan)
            print("Travel_time: ", travel_time)

    # Optionally return aggregated stats
    return {
        'collision_rate': buffer_collision_rate,
        'infeasible_rate': buffer_infeasible_rate,
        'avg_min_cost': buffer_avg_minimal_cost,
        'avg_inter_cost': buffer_avg_intermediate_cost,
        'avg_term_cost': buffer_avg_terminal_cost,
        'avg_ctrl_cost': buffer_avg_control_cost,
        'pred_times': buffer_prediction_times,
        'travel_times': buffer_travel_times,
        'success_count': success_count,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_x', type=float, default=8.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=0.2)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--r_star', type=float, default=0.5)
    args = parser.parse_args()

    main(args.goal_x, args.goal_y, args.num_iter, args.r_star)


