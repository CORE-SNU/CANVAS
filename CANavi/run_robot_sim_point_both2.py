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

import sys
sys.path.append('/home/snowhan1021/tools_paper/CANavi')
from env import Environment
from detection.detection_utils import Box
from control.grid_solver import GridMPC
# from control.sampling_based_mpc import SamplingBasedMPC
from conformal_prediction.adaptive_cp import AdaptiveConformalPredictionModule
from prediction.linear_predictor import LinearPredictor
from trajectron_predictor import TrajectronPredictor
from koopman.koopy_predictor_justmul import KoopmanPredictor
# from koopman.koopman_predictor_clu_geo import KoopmanPredictor
# from prediction.eigen.eigen_predictor import eigen_predictor

from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

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

def is_same_box(box1, box2, tol=0.1):
    center1 = np.array([box1.x, box1.y])
    center2 = np.array([box2.x, box2.y])
    return np.linalg.norm(center1 - center2) < tol

# -----------------------------
# Visualization helper
# -----------------------------
def save_blockA_frame_png(outdir,
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
                          annotate_ci=True,
                          ci_decimals=2,
                          ci_fontsize=7,
                          max_ci_annotations_per_step=None,
                          xlim=(-7.5, 13.5),
                          ylim=(-12.5, 5.5)):
    """
    Draws: static boxes (gray fill/edge), robot traj/current/goal,
           history (navy thin), GT-future (black dotted), prediction (red),
           circles at t+{2,5,10}: R* (lightgray fill), err (orange fill, black edge),
           CI text near predicted point, legend for history/GT/pred.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1) Static boxes — gray fill + gray edge (polygons)
    if static_boxes:
        for b in static_boxes:
            poly = Polygon(b.vertices, closed=True,
                           facecolor='gray', edgecolor='gray',
                           linewidth=1, zorder=0.1)
            ax.add_patch(poly)

    # 2) Robot trajectory / current / goal
    px, py = robot_traj_xy
    if len(px) and len(py):
        ax.plot(px, py, linewidth=2)  # color default
    ax.scatter([robot_xy[0]], [robot_xy[1]], marker='o', s=30)  # current
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker='*', s=80)

    # 3) History(8) — navy thin
    if valid_obs:
        for pid, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)

    # 4) GT future(12) — black dotted
    if valid_obs_future_true:
        for pid, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], linestyle='--', color='black', linewidth=1)

    # 5) Prediction(12) — red
    if prediction_res:
        for pid, arr in prediction_res.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], color='red', linewidth=1.5)

    # 6) Circles (R* and err) + CI text
    if (r_star is not None) and (r_star > 0) and valid_obs_future_true and prediction_res:
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

                    # R* circle: light gray fill, no border
                    ax.add_patch(Circle((p[0], p[1]), r_star,
                                        fill=True, edgecolor='none', facecolor='lightgray', zorder=0.5))
                    # err circle: orange fill with black border
                    ax.add_patch(Circle((p[0], p[1]), err,
                                        fill=True, edgecolor='black', facecolor='orange', linewidth=1, zorder=1.0))
                    # GT marker
                    ax.scatter([g[0]], [g[1]], marker='x', s=18, color='black', zorder=2.0)

                    # CI text
                    if annotate_ci:
                        if (max_ci_annotations_per_step is None) or (ann_counts[s] < max_ci_annotations_per_step):
                            ci = (r_star - err) / r_star
                            if np.isfinite(ci):
                                dx, dy = offsets.get(s, (0.04, 0.04))
                                ax.text(p[0] + dx, p[1] + dy,
                                        f"CI@t+{s}={ci:.{ci_decimals}f}",
                                        fontsize=ci_fontsize, zorder=3.0)
                                ann_counts[s] += 1

        ax.text(0.01, 0.99,
                "Circles @ pred: R* (light gray fill, no border), err (orange fill, black border)",
                transform=ax.transAxes, fontsize=8, va='top', color='black')

    # Legend (history / GT future / prediction)
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

    # Unified R* for CI
    rstar = r_star
    steps_to_eval = (2, 5, 10)  # for control-input CI

    # -----------------------------
    # GLOBAL BUFFERS (Trajectron++)
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
    buffer_pos_x_result = []  # position_x for each experiments
    buffer_pos_y_result = []  # position_y for each experiments

    buffer_intermediate = []
    buffer_terminal = []
    buffer_control = []

    # --------------------------------------
    # GLOBAL BUFFERS (Oracle ground truth)
    # --------------------------------------
    buffer_collision_rate_oracle = []
    buffer_infeasible_rate_oracle = []
    buffer_avg_minimal_cost_oracle = []
    buffer_avg_intermediate_cost_oracle = []
    buffer_avg_terminal_cost_oracle = []
    buffer_avg_control_cost_oracle = []
    buffer_prediction_times_oracle = []
    buffer_travel_times_oracle = []
    success_count_oracle = 0
    buffer_pos_x_result_oracle = []
    buffer_pos_y_result_oracle = []

    buffer_intermediate_oracle = []
    buffer_terminal_oracle = []
    buffer_control_oracle = []

    for times in range(num_iter):
        # =========================================================
        # BLOCK A: predictor-driven MPC (cache plan per frame)
        # =========================================================
        print("==================================")
        print("BLOCK A Started")
        frame = 0
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        buffer_timestamp = []
        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []  # per-frame x within this run
        buffer_pos_y = []

        goal = np.array([goal_x, goal_y])

        data_dir = "/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_17_Mar_2025_22_52_52lobby_data_ar3"
        # obj_predictor = LinearPredictor(prediction_len=prediction_len, history_len=8, smoothing_factor=0.75, dt=dt)
        # obj_predictor = KoopmanPredictor(prediction_len=prediction_len, data_dir=data_dir, min_samples=100, dt=dt, pattern=r'^.*\d{2}\.npy$')
        obj_predictor = TrajectronPredictor(prediction_len=prediction_len, model_dir=data_dir, device='cpu')  # Trajectron++ predictor
        controller = GridMPC(n_steps=prediction_len, dt=dt)

        max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1)
        offline_calibration_set = {i: [] for i in range(prediction_len)}

        cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                      step_size=0.05,
                                                      n_scores=prediction_len,
                                                      max_interval_lengths=max_interval_lengths,
                                                      sample_size=20,
                                                      offline_calibration_set=offline_calibration_set)

        begin = time.time()
        print("NEED TO IMPLEMENT INITIAL POSITION OF ROBOT LATER")
        init_robot_pose = np.array([0, 0, np.pi / 2.])
        t_begin = 40
        t_step = t_begin
        t_end = 2000
        velocity = np.array([0., 0., ])
        buffer_vel = []
        done = False

        # iteration output dir for viz
        iter_out_dir = pathlib.Path("viz_blockA") / f"iter_{times+1:03d}"
        iter_out_dir.mkdir(parents=True, exist_ok=True)

        # Cache of Block A plans per frame (for Block B comparison)
        plans_pred_by_frame = {}

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

            # record robot trajectory for viz
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # Observations
            observation = environment._get_obs()
            observation_future_true = environment._get_obs_future()

            # Filter valid histories and GT futures (finite & correct shape)
            valid_obs = {}
            valid_obs_future_true = {}
            for pid, traj in observation.items():
                # history
                try:
                    arr_hist = np.asarray(traj, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if not (arr_hist.ndim == 2 and arr_hist.shape[0] == 8 and arr_hist.shape[1] >= 2 and np.isfinite(arr_hist[:, :2]).all()):
                    continue
                valid_obs[pid] = arr_hist

                # GT future
                fut = observation_future_true.get(pid, None)
                if fut is None:
                    continue
                arr_fut = np.asarray(fut, dtype=np.float64)
                if not (arr_fut.ndim == 2 and arr_fut.shape[1] >= 2 and arr_fut.shape[0] >= prediction_len and np.isfinite(arr_fut[:prediction_len, :2]).all()):
                    continue
                valid_obs_future_true[pid] = arr_fut[:prediction_len, :2]

            # Collision check (simple proximity)
            if valid_obs:
                dynamic_obs = valid_obs
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
                robot_pos = np.array([position_x, position_y])
                distances = np.sqrt(np.sum((initial_positions - robot_pos) ** 2, axis=1))
                collisions = distances <= 0.7
                if np.any(collisions):
                    print("Collision!")
                    collision_count += int(np.sum(collisions))

            # Prediction → controller
            pred_start = time.time()
            prediction_res = obj_predictor(dynamic_obs if valid_obs else {})
            pred_time = time.time() - pred_start
            buffer_prediction_times.append(pred_time)
            confidence_intervals = cp_module.update(dynamic_obs if valid_obs else {}, prediction_res if isinstance(prediction_res, dict) else {})

            # Control (MPC with predictions)
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

            # Cache Block A plan at this frame (for Block B comparison)
            try:
                u_seq_pred = np.asarray(velocity, dtype=np.float64)  # (H,2) [v,w]
            except Exception:
                u_seq_pred = None
            J_pred = float(minimal)
            Jcomp_pred = float(intermediate + terminal + control)
            plans_pred_by_frame[frame] = {
                'u_seq': u_seq_pred,
                'J': J_pred,
                'Jcomp': Jcomp_pred
            }

            # Save visualization (history/GT/pred + circles/CI labels)
            try:
                save_blockA_frame_png(
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
                    annotate_ci=True,
                    ci_decimals=2,
                    ci_fontsize=7,
                    max_ci_annotations_per_step=None
                )
            except Exception as e:
                print(f"[WARN] viz save failed at frame {frame}: {e}")

            # Feasibility / stopping logic
            buffer_infeasibility.append(info['feasible'])
            if not info['feasible']:
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

            # Goal check
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

            # Apply first control to environment
            cmd_linear_x, cmd_angular_z = velocity[0]
            robot_pose, done = environment.step([cmd_linear_x, cmd_angular_z])
            position_x, position_y, orientation_z = robot_pose

            # Accumulate costs
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            frame += 1
            buffer_vel = velocity

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

        # =========================================================
        # BLOCK B: oracle-driven MPC (control-test CI 3 types)
        # =========================================================
        print("==================================")
        print("BLOCK B Started")
        frame = 0
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        buffer_timestamp = []
        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []
        buffer_pos_y = []

        # CI buffers (control-test)
        iter_ci_obj = []       # per-frame objective-diff CI
        iter_ci_cost = []      # per-frame component-cost-sum diff CI
        iter_ci_u_records = [] # per-frame, selected step input-diff CI

        data_dir = "/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_17_Mar_2025_22_52_52lobby_data_ar3"
        obj_predictor = TrajectronPredictor(prediction_len=prediction_len, model_dir=data_dir, device='cpu')  # not used here
        controller = GridMPC(n_steps=prediction_len, dt=dt)

        max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1)
        offline_calibration_set = {i: [] for i in range(prediction_len)}
        cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                      step_size=0.05,
                                                      n_scores=prediction_len,
                                                      max_interval_lengths=max_interval_lengths,
                                                      sample_size=20,
                                                      offline_calibration_set=offline_calibration_set)

        begin = time.time()
        print("NEED TO IMPLEMENT INITIAL POSITION OF ROBOT LATER")
        init_robot_pose = np.array([0, 0, np.pi / 2.])
        t_begin = 40
        t_step = t_begin
        t_end = 2000
        velocity = np.array([0., 0., ])
        buffer_vel = []
        done = False

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

            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            observation = environment._get_obs()
            observation_future_true = environment._get_obs_future()

            # Filter valid histories & GT futures
            valid_obs = {}
            valid_obs_future_true = {}
            for pid, traj in observation.items():
                try:
                    arr_hist = np.asarray(traj, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if not (arr_hist.ndim == 2 and arr_hist.shape[0] == 8 and arr_hist.shape[1] >= 2 and np.isfinite(arr_hist[:, :2]).all()):
                    continue
                valid_obs[pid] = arr_hist

                fut = observation_future_true.get(pid, None)
                if fut is None:
                    continue
                arr_fut = np.asarray(fut, dtype=np.float64)
                if not (arr_fut.ndim == 2 and arr_fut.shape[1] >= 2 and arr_fut.shape[0] >= prediction_len and np.isfinite(arr_fut[:prediction_len, :2]).all()):
                    continue
                valid_obs_future_true[pid] = arr_fut[:prediction_len, :2]

            if valid_obs:
                dynamic_obs = valid_obs
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
                robot_pos = np.array([position_x, position_y])
                distances = np.sqrt(np.sum((initial_positions - robot_pos) ** 2, axis=1))
                if np.any(distances <= 0.7):
                    print("Collision!")
                    collision_count += int(np.sum(distances <= 0.7))

            # Oracle prediction = GT futures
            pred_start = time.time()
            prediction_res = valid_obs_future_true  # oracle
            pred_time = time.time() - pred_start
            buffer_prediction_times_oracle.append(pred_time)
            confidence_intervals = cp_module.update(dynamic_obs if valid_obs else {}, prediction_res if prediction_res else {})

            # Control (MPC with oracle predictions)
            velocity, info, minimum, intermediate, terminal, control, minimal = controller(
                pos_x=position_x,
                pos_y=position_y,
                orientation_z=orientation_z,
                linear_x=linear_x,
                angular_z=angular_z,
                boxes=persistent_static_boxes,
                predictions=prediction_res if prediction_res else {},
                confidence_intervals=confidence_intervals,
                goal=goal
            )

            # ---- Control-test CI (compare with Block A's plan of same frame) ----
            try:
                u_seq_gt = np.asarray(velocity, dtype=np.float64)  # (H,2)
            except Exception:
                u_seq_gt = None
            J_gt = float(minimal)
            Jcomp_gt = float(intermediate + terminal + control)

            pred_plan = plans_pred_by_frame.get(frame, None)
            if pred_plan is not None:
                # 1) Objective-diff CI
                err_obj = abs(pred_plan['J'] - J_gt)
                if np.isfinite(err_obj) and rstar > 0:
                    ci_obj = (rstar - err_obj) / rstar
                    if np.isfinite(ci_obj):
                        iter_ci_obj.append(ci_obj)

                # 2) Cost-diff CI (approx: sum of components)
                err_cost = abs(pred_plan['Jcomp'] - Jcomp_gt)
                if np.isfinite(err_cost) and rstar > 0:
                    ci_cost = (rstar - err_cost) / rstar
                    if np.isfinite(ci_cost):
                        iter_ci_cost.append(ci_cost)

                # 3) Input-diff CI on selected steps
                u_seq_pred = pred_plan['u_seq']
                if (u_seq_pred is not None) and (u_seq_gt is not None) and \
                   (u_seq_pred.ndim == 2) and (u_seq_gt.ndim == 2) and \
                   (u_seq_pred.shape[1] >= 2) and (u_seq_gt.shape[1] >= 2):
                    for S in steps_to_eval:
                        j = S - 1
                        if (u_seq_pred.shape[0] > j) and (u_seq_gt.shape[0] > j):
                            p = u_seq_pred[j, :2]; g = u_seq_gt[j, :2]
                            if np.isfinite(p).all() and np.isfinite(g).all():
                                err_u = float(np.linalg.norm(p - g))
                                if np.isfinite(err_u) and rstar > 0:
                                    ci_u = (rstar - err_u) / rstar
                                    if np.isfinite(ci_u):
                                        iter_ci_u_records.append({
                                            'frame': frame,
                                            'step': S,
                                            'err_u': err_u,
                                            'ci_u': ci_u
                                        })

            # Feasibility & progress
            buffer_infeasibility.append(info['feasible'])
            if not info['feasible']:
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

            # Goal check
            if np.abs(position_x - goal[0]) < 0.3 and np.abs(position_y - goal[1]) < 0.3:
                print(frame, 'Goal reached!')
                environment.step([0, 0])
                is_success = True
                success_count_oracle += 1
                travel_time = time.time() - begin
                buffer_travel_times_oracle.append(travel_time)
                buffer_pos_x_result_oracle.append(buffer_pos_x)
                buffer_pos_y_result_oracle.append(buffer_pos_y)
                buffer_avg_minimal_cost_oracle.append(np.sum(minimum_cost) / len(minimum_cost) if minimum_cost else np.nan)
                buffer_avg_intermediate_cost_oracle.append(np.sum(buffer_intermediate_oracle) / len(buffer_intermediate_oracle) if buffer_intermediate_oracle else np.nan)
                buffer_avg_terminal_cost_oracle.append(np.sum(buffer_terminal_oracle) / len(buffer_terminal_oracle) if buffer_terminal_oracle else np.nan)
                buffer_avg_control_cost_oracle.append(np.sum(buffer_control_oracle) / len(buffer_control_oracle) if buffer_control_oracle else np.nan)
                break

            # Apply first control to environment
            cmd_linear_x, cmd_angular_z = velocity[0]
            robot_pose, done = environment.step([cmd_linear_x, cmd_angular_z])
            position_x, position_y, orientation_z = robot_pose

            # Accumulate costs
            minimum_cost.append(minimal)
            buffer_intermediate_oracle.append(intermediate)
            buffer_terminal_oracle.append(terminal)
            buffer_control_oracle.append(control)

            frame += 1
            buffer_vel = velocity

        buffer_collision_rate_oracle.append(collision_count / max(1, frame))
        buffer_infeasible_rate_oracle.append(infeasible_count / max(1, frame))

        print("Next : #{}_scenario_oracle".format(times + 1))
        print("Collision_rate_oracle: ", collision_count / max(1, frame))
        print("Infeasible_rate_oracle: ", infeasible_count / max(1, frame))
        if buffer_prediction_times_oracle:
            print("Avg_prediction_time_oracle: ", np.sum(buffer_prediction_times_oracle) / len(buffer_prediction_times_oracle))
            print('Variance prediction time_oracle', np.var(buffer_prediction_times_oracle))
        if is_success and minimum_cost:
            print("Avg_minimal_cost_oracle: ", np.sum(minimum_cost) / len(minimum_cost))
            print("Avg_intermediate_cost_oracle: ", np.sum(buffer_intermediate_oracle) / len(buffer_intermediate_oracle) if buffer_intermediate_oracle else np.nan)
            print("Avg_terminal_cost_oracle: ", np.sum(buffer_terminal_oracle) / len(buffer_terminal_oracle) if buffer_terminal_oracle else np.nan)
            print("Avg_control_cost_oracle: ", np.sum(buffer_control_oracle) / len(buffer_control_oracle) if buffer_control_oracle else np.nan)
            print("Travel_time_oracle: ", travel_time)

        # ---- Print control-test CI summaries for this iteration ----
        # Objective-based
        if iter_ci_obj:
            c = np.array(iter_ci_obj, dtype=np.float64)
            c = c[np.isfinite(c)]
            if c.size:
                print(f"[Iter {times+1}][Control] CI_obj_mean={c.mean():.3f} (N={c.size})")
            else:
                print(f"[Iter {times+1}][Control] CI_obj_mean: no finite")
        else:
            print(f"[Iter {times+1}][Control] CI_obj_mean: no records")

        # Cost-based (component-sum)
        if iter_ci_cost:
            c = np.array(iter_ci_cost, dtype=np.float64)
            c = c[np.isfinite(c)]
            if c.size:
                print(f"[Iter {times+1}][Control] CI_cost_mean={c.mean():.3f} (N={c.size})")
            else:
                print(f"[Iter {times+1}][Control] CI_cost_mean: no finite")
        else:
            print(f"[Iter {times+1}][Control] CI_cost_mean: no records")

        # Control-input-based
        if iter_ci_u_records:
            cs = np.array([r['ci_u'] for r in iter_ci_u_records], dtype=np.float64)
            cs = cs[np.isfinite(cs)]
            if cs.size:
                print(f"[Iter {times+1}][Control] CI_u_mean(all steps)={cs.mean():.3f} (N={cs.size})")
                for S in steps_to_eval:
                    vals = np.array([r['ci_u'] for r in iter_ci_u_records if r['step']==S], dtype=np.float64)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        print(f"  - t+{S}: CI_u_mean={vals.mean():.3f} (N={vals.size})")
            else:
                print(f"[Iter {times+1}][Control] CI_u_mean: no finite")
        else:
            print(f"[Iter {times+1}][Control] CI_u_mean: no records")

    # End for num_iter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_x', type=float, default=8.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=0.2)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--r_star', type=float, default=0.5)
    args = parser.parse_args()
    main(args.goal_x, args.goal_y, args.num_iter, args.r_star)
