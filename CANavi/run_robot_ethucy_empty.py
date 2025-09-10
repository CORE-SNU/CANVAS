import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib
import numpy as np
import yaml
import cv2
import pathlib
import os
import subprocess
import random
import pickle

import sys
sys.path.append('/home/snowhan1021/tools_paper/CANavi')
from env_empty import Environment
from detection.detection_utils import Box
from control.grid_solver import GridMPC
# from control.sampling_based_mpc import SamplingBasedMPC
from conformal_prediction.adaptive_cp import AdaptiveConformalPredictionModule
from predictor import Predictors

matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

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
                x=x_center,
                y=y_center,
                w=width,
                h=height,
                deg=0,
                rad=0,
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
                          xlim=(-2.5, 10.0),  #(-7.5, 13.5)
                          ylim=(-10.0, 2.0)): #(-12.5, 5.5)
    """
    valid_obs: dict[pid -> (8, >=2)]
    valid_obs_future_true: dict[pid -> (12, >=2)]
    prediction_res: dict[pid -> (12, >=2)]
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1) static boxes frame
    if static_boxes:
        for b in static_boxes:
            # b.vertices: (4,2)
            poly = Polygon(b.vertices, closed=True, facecolor='gray', edgecolor='gray', linewidth=1, zorder=0.1)
            ax.add_patch(poly)

    # 2) robot trajectory + current position + goal
    px, py = robot_traj_xy
    if len(px) and len(py):
        ax.plot(px, py, linewidth=2)
    ax.scatter([robot_xy[0]], [robot_xy[1]], marker='o', s=30)  # current
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker='*', s=80)

    # 3) pedestrian history(8)  — narrow line
    if valid_obs:
        for pid, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)

    # 4) ground_truth for future(12) — dotted line
    if valid_obs_future_true:
        for pid, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], linestyle='--', color='black', linewidth=1)

    # 5) prediction(12) — line
    if prediction_res:
        for pid, arr in prediction_res.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], color='red', linewidth=1.5)

    # 6) circle for visualize err & R* (center = predicted position)
    if (r_star is not None) and valid_obs_future_true and prediction_res:
        common = set(prediction_res.keys()) & set(valid_obs_future_true.keys())
        
        offsets = {2: (0.05, 0.05), 5: (-0.05, 0.05), 10: (0.05, -0.05)}
        ann_counts = {s: 0 for s in steps_to_annotate}
        
        for s in steps_to_annotate:
            j = s - 1  # 0-index
            for pid in common:
                pred = np.asarray(prediction_res[pid], dtype=np.float64)
                gt   = np.asarray(valid_obs_future_true[pid], dtype=np.float64)
                if (pred.ndim == 2 and gt.ndim == 2 and pred.shape[1] >= 2 and gt.shape[1] >= 2
                    and len(pred) > j and len(gt) > j):
                    p = pred[j, :2]; g = gt[j, :2]
                    if np.isfinite(p).all() and np.isfinite(g).all():
                        err = float(np.linalg.norm(p - g))
                        # R* (circle with dotted line)
                        ax.add_patch(Circle((p[0], p[1]), r_star, fill=True, edgecolor='none', facecolor='lightgray', zorder=0.5))
                        # err (circle with line)
                        ax.add_patch(Circle((p[0], p[1]), err, fill=True, edgecolor='black', facecolor='orange', linewidth=1, zorder=1.0))
                        # ground truth position (×)
                        #ax.scatter([g[0]], [g[1]], marker='x', s=18)

                        # CI annotation
                        if annotate_ci:
                            if (max_ci_annotations_per_step is None) or (ann_counts[s] < max_ci_annotations_per_step):
                                ci = (r_star - err) / r_star
                                if np.isfinite(ci):
                                    dx, dy = offsets.get(s, (0.04, 0.04))
                                    ax.text(p[0] + dx, p[1] + dy,
                                            f"CI@t+{s}={ci:.{ci_decimals}f}",
                                            fontsize=ci_fontsize)
                                    ann_counts[s] += 1

        #ax.text(0.01, 0.99, "Circles @ pred: solid=err, dotted=R*; text=CI", transform=ax.transAxes, fontsize=8, va='top')

    # --- legend: History / GT future / Prediction ---
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

def collect_pred_errors_and_ci(prediction_res, valid_obs_future_true, base_time, dt, rstar, steps_to_eval=(2,5,10)):
    recs = []
    if not isinstance(prediction_res, dict) or not isinstance(valid_obs_future_true, dict):
        return recs
    # r* cap
    if not np.isfinite(rstar) or rstar <= 0:
        rstar = 1e-6

    common = set(prediction_res.keys()) & set(valid_obs_future_true.keys())
    for pid in common:
        P = np.asarray(prediction_res[pid], dtype=np.float64)
        G = np.asarray(valid_obs_future_true[pid], dtype=np.float64)
        if not (P.ndim == 2 and G.ndim == 2 and P.shape[1] >= 2 and G.shape[1] >= 2):
            continue

        # evaluate only for selected steps (1-indexed → 0-index)
        for S in steps_to_eval:
            j = S - 1
            if len(P) <= j or len(G) <= j:
                continue
            p = P[j, :2]; g = G[j, :2]
            if not (np.isfinite(p).all() and np.isfinite(g).all()):
                continue
            err = float(np.linalg.norm(p - g))
            if not np.isfinite(err):
                continue
            ci = (rstar - err) / rstar
            if not np.isfinite(ci):
                continue
            recs.append({
                'pid': pid,
                'step': S,                          # 2,5,10th
                't_pred': base_time + S * dt,       # future timestep
                'err': err,
                'ci': ci
            })
    return recs

def main(goal_x, goal_y, num_iter, r_star,data_dir,files):
    # odometry/filtered rate : 50Hz / ouster/points rate : 10Hz
    dt = 0.10

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
        # BLOCK A: ORIGINAL LOGIC — Trajectron++ drives controller
        # =========================================================
        frame = 0
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        buffer_timestamp = []
        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []  # position_x for each frame in current experiment
        buffer_pos_y = []  # position_y for each frame in current experiment
        steps_to_eval = (2, 5, 10)

        goal = np.array([goal_x, goal_y])

        prediction_len = 12
        koopman_dir = "/home/snowhan1021/tools_paper/CANavi/koopman"
        #obj_predictor = LinearPredictor(prediction_len=prediction_len, history_len=8, smoothing_factor=0.75, dt=dt)                            # Linear predictor
        #obj_predictor = KoopmanPredictor(prediction_len=prediction_len, data_dir=data_dir, min_samples=100, dt=dt, pattern=r'^.*\d{2}\.npy$')  # Koopman predictor
        #obj_predictor = TrajectronPredictor(prediction_len=prediction_len, model_dir=data_dir, device='cpu')                                    # Trajectron++ predictor
        obj_predictor = Predictors(name="linear")                                                                                                 # EigenTrajectory predictor
        controller = GridMPC(n_steps=prediction_len, dt=dt)

        max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1)
        offline_calibration_set = {i: [] for i in range(prediction_len)}

        cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                      step_size=0.05,
                                                      n_scores=prediction_len,
                                                      max_interval_lengths=max_interval_lengths,
                                                      sample_size=20,
                                                      offline_calibration_set=offline_calibration_set
                                                      )

        begin = time.time()
        goal = np.array([goal_x, goal_y])
        t_begin = 40
        t_step=t_begin
        t_end=200
        velocity = np.array([0., 0., ])
        buffer_vel = []
        done = False
        environment = Environment(
                filepath=os.path.join(files+'.npy'),
                dt=dt,
                init_robot_pose=0,#NOT USED
                n_pedestrians=0,#NOT USED
                t_begin=t_begin,
                t_end=t_end
            )
        environment.reset()
        while not done:
            detect_time = time.time()
            # (i) Detecting : Pointcloud to rectangles (bounding boxes)
            # (ii) tracking : object tracking → return (trajectories, object_types)
            observation = environment._get_obs()
            # Keep only 8-step, all-finite (no NaN/Inf/None) trajectories with at least (x,y)
            valid_obs = {}
            valid_obs_future_true = {}
            for pid, traj in observation.items():
                try:
                    arr = np.asarray(traj, dtype=np.float64)   # fails if there are None's
                except (TypeError, ValueError):
                    continue
                if arr.shape[0] == 8 and arr.ndim == 2 and arr.shape[1] >= 2 and np.isfinite(arr).all():
                    valid_obs[pid] = arr
            dynamic_obs={}
            # (iii) Proceed only if there is at least one valid 8-step trajectory
            if valid_obs:
                dynamic_obs = valid_obs  # your dataset is pedestrians only
                # Create array with the latest (x,y) of each trajectory (last row of the 8)
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
            else:
                # No valid 8-step observations this step; skip or handle as you like
                pass

            # (iv) predicting : with dynamic object & predictor -> get pedestrian trajectories
            
            try:
                pred_start = time.time()
                prediction_res = obj_predictor(dynamic_obs)
                pred_time = time.time() - pred_start
                buffer_prediction_times.append(pred_time)
            except:
                print(valid_obs)
                print(dynamic_obs)
                print(f"[WARN] prediction failed at {files}")
                break
            

            
            # Save visualization

            
            _,done=environment.step()
            # Publish the control inputs
            #print("Minimum, intermediate, terminal, control, minimum_idx : ", minimal, intermediate, terminal, control, minimum)
            #print("Percentages - intermediate, terminal, control : ", intermediate/minimal, terminal/minimal, control/minimal)
            frame += 1
            buffer_vel = velocity


        print("Next : #{}_scenario".format(times + 1))
        print("Avg_prediction_time: ", np.sum(buffer_prediction_times) / len(buffer_prediction_times))
        print('Variance prediction time', np.var(buffer_prediction_times))

    # (Your old commented-out pickle/save block stays as-is if you want to re-enable it.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_x', type=float, default=8.0) # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=0.2) # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--r_star', type=float, default=0.5)
    args = parser.parse_args()
    datasets = ['biwi_eth', 'biwi_hotel', 'students001','students003', 'crowds_zara01', 'crowds_zara02']
    data_directories=[
    '/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_11_Feb_2025_10_01_22eth_vel_ar3',
    '/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_10_Feb_2025_21_00_50hotel_vel_ar3',
    '/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_10_Feb_2025_15_01_26_univ_vel_ar3',
    '/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_10_Feb_2025_15_01_26_univ_vel_ar3',
    '/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_10_Feb_2025_11_19_14_zara01_vel_ar3',
    '/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_03_Feb_2025_14_11_39_zara02_vel_ar3']
    for i, j in zip(datasets, data_directories):
        main(args.goal_x, args.goal_y, args.num_iter, args.r_star,j,i)
