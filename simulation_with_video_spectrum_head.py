import time
import argparse
import numpy as np
import cv2
import pathlib
import os
import subprocess
import random
import pickle
import csv

import sys
_DATA_DIR = os.path.dirname(__file__)

sys.path.append(_DATA_DIR)
from canvas.datasets.dataset_loader import get_dataset_spec, _load_background_image
from canvas import Environment, Box, GridMPC, \
    AdaptiveConformalPredictionModule, Predictors, CompetencyIndex, Predictor_CI
from save_ci import save_ci_traj_positions_csv, save_ci_ctrl_local_csv, project_ctrl_step_to_local_xy, save_ci_iteration_csv, save_frame_png,save_frame_png_spectrum,save_frame_png_spectrum_video
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from math import radians, cos, sin
from sim_raw_overlay import RawVideoOverlay

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""

# -----------------------------
# Static map geometry → boxes
# -----------------------------
#persistent_static_boxes = []    # Save static object as bounding box

'''
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
'''
def region_to_box(region: dict, default_deg: float = 0.0, resolution: float = 1e-3) -> Box:
    xmin, xmax = region["xmin"], region["xmax"]
    ymin, ymax = region["ymin"], region["ymax"]
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    deg = float(region.get("deg", default_deg))
    rad = radians(deg)
    
    corners = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2],
    ], dtype=float)
    
    c, s = cos(rad), sin(rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    rot_corners = (corners @ R.T) + np.array([x_center, y_center])
    return Box(
        x=x_center, y=y_center, w=w, h=h,
        deg=deg, rad=rad, area=w*h,
        vertices=rot_corners,
        resolution=resolution,
        pos=np.array([x_center, y_center], dtype=float)
    )


# -----------------------------
# Main
# -----------------------------
def main(goal_x, goal_y, num_iter, r_star, dataset, predictor, video_fps, save_video,
         overlay, frame_offset, extracted_fps, output_fps):
    # Simulation rates
    #dt = 0.10
    dt = 1/2.5

    persistent_static_boxes = [region_to_box(r) for r in get_dataset_spec(dataset).static_regions]

    # Predictor horizon
    prediction_len = 12
    history_len = 8

    # Unified R* (kept for future score/CI; not used in controller)
    rstar = r_star
    #steps_to_eval = (2, 5, 10)

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
        it_ci_traj_pos_rows = []   # rows: {frame, pid, step, x, y, ci}  (global pedestrian positions)
        it_ci_ctrl_local_rows = [] # rows: {frame, step, x, y, ci}       (robot-centered local)
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

        spec = get_dataset_spec(dataset)
        bg_img = _load_background_image(spec.bg.path, spec.bg.rotate90)
        bg_extent = spec.bg.extent
        bg_alpha = spec.bg.alpha

        # ---- Choose predictor ----
        #data_dir = "/home/snowhan1021/tools_paper/CANavi/prediction/trajectron/models_17_Mar_2025_22_52_52lobby_data_ar3"
        obj_predictor = Predictors(chosen_predictor=predictor,prediction_len=prediction_len,history_len=history_len,dt=dt,dataset=dataset,device='cpu')                                    # Trajectron++ predictor

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
        cp_module_gt = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
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

        datasets_dir = os.path.join(_DATA_DIR, "src", "canvas", "datasets")
        fname_map = {
            "Lobby":  "0.npy",
            "ETH":    "biwi_eth.npy",
            "Hotel":  "biwi_hotel.npy",
            "Zara01": "crowds_zara01.npy",
            "Zara02": "crowds_zara02.npy",
            "Univ":   "students003.npy",
        }
        npy_path = os.path.join(datasets_dir, fname_map[dataset])

        # iteration output dir for viz
        iter_out_dir = pathlib.Path("viz") / f"iter_{times+1:03d}"
        iter_out_dir.mkdir(parents=True, exist_ok=True)

        environment = Environment(
            filepath=npy_path,
            dt=dt,
            init_robot_pose=init_robot_pose,
            t_begin=t_begin,
            t_end=t_end
        )
        position_x, position_y, orientation_z = environment.reset()

        video_writer = None
        video_path = iter_out_dir / f"sim_iter_{times+1:03d}_head.mp4"

        overlay_result = None
        if overlay:
            out_mp4 = iter_out_dir / f"sim_iter_{times+1:03d}_raw_overlay.mp4"
            overlay_result = RawVideoOverlay(
                dataset=dataset,
                out_video_path=str(out_mp4),
                frame_offset=frame_offset,
                sim_dt=dt,
                extracted_fps=extracted_fps,
                output_fps=output_fps
            )

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
            confidence_intervals_gt=cp_module_gt.update(dynamic_obs, valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {})

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
                confidence_intervals=confidence_intervals_gt,
                goal=goal
            )
            prediction_competency=Predictor_CI()
            prediction_comptency_score=prediction_competency.CI_default(confidence_intervals)
            
            # 1) traj CI (series)
            ci_traj_series = ci_traj(
                prediction_res=prediction_res if isinstance(prediction_res, dict) else {},
                gt_future=valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {},
            )
            buffer_ci_traj_series.append(ci_traj_series)
            it_ci_traj_series.append(ci_traj_series)

            traj_anchor = "pred"  # or "gt"
            if isinstance(prediction_res, dict) and isinstance(valid_obs_future_true, dict):
                common_pids = set(prediction_res.keys()) & set(valid_obs_future_true.keys())
                for pid in common_pids:
                    p = np.asarray(prediction_res[pid], dtype=np.float64)      # (~T, >=2)
                    g = np.asarray(valid_obs_future_true[pid], dtype=np.float64)
                    if p.ndim >= 2 and g.ndim >= 2 and p.shape[1] >= 2 and g.shape[1] >= 2:
                        T = min(len(p), len(g), prediction_len)
                        if T > 0:
                            err = np.linalg.norm(p[:T, :2] - g[:T, :2], axis=1)   # (T,)
                            ci  = (rstar - err) / rstar
                            ci  = np.minimum(ci, 1.0)                            # clip to 1.0
                            xy_src = p if traj_anchor == "pred" else g
                            for j in range(T):
                                x, y = float(xy_src[j, 0]), float(xy_src[j, 1])
                                cij  = float(ci[j])
                                if np.isfinite(x) and np.isfinite(y) and np.isfinite(cij):
                                    it_ci_traj_pos_rows.append({
                                        "frame": int(frame),
                                        "pid": int(pid),
                                        "step": int(j + 1),  # 1-based
                                        "x": x,
                                        "y": y,
                                        "ci": cij
                                    })

            # 2) control CI (series)
            ci_ctrl_series = ci_ctrl(
                ctrl_pred=velocity if velocity is not None else [],
                ctrl_gt=velocity_gt if velocity_gt is not None else [],
            )
            buffer_ci_ctrl_series.append(ci_ctrl_series)
            it_ci_ctrl_series.append(ci_ctrl_series)

            ctrl_mode = "unicycle"  # or "cartesian"
            T_ctrl = min(len(ci_ctrl_series), len(velocity) if velocity is not None else 0)
            for k in range(T_ctrl):
                dx, dy = project_ctrl_step_to_local_xy(velocity[k], dt, mode=ctrl_mode)
                ci_k   = float(ci_ctrl_series[k])
                if np.isfinite(dx) and np.isfinite(dy) and np.isfinite(ci_k):
                    it_ci_ctrl_local_rows.append({
                        "frame": int(frame),
                        "step": int(k + 1),  # 1-based
                        "x": float(dx),      # local displacement for this step
                        "y": float(dy),
                        "ci": ci_k
                    })

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

            # --------- Visualization (CI labels disabled by default) ---------
            try:
                bg_img = _load_background_image(overlay_result._frame_path_for_current(), spec.bg.rotate90)
                frame_png=save_frame_png_spectrum_video(
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
                    annotate_ci=True,  # keep False here; enable later if needed
                    background_image=bg_img,
                    background_extent=bg_extent,
                    background_alpha=bg_alpha
                )
                if save_video:
                    img = cv2.imread(frame_png)
                    if img is not None:
                        if video_writer is None:
                            h, w = img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))
                        video_writer.write(img)
            except Exception as e:
                print(f"[WARN] viz save failed at frame {frame}: {e}")

            if overlay_result is not None:
                overlay_result.step(valid_obs, valid_obs_future_true, prediction_res)
            
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

            '''
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
            print(f"[frame {frame}] CI_Prediction  avg={prediction_comptency_score:.3f}")
            '''
            # --------- Accumulate costs ---------
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            frame += 1
            buffer_vel = velocity

        '''
        # ===== Write per-iteration CI CSV =====
        ci_pos_csv_path = save_ci_traj_per_agent_csv(
            iter_out_dir=iter_out_dir,
            iteration_index=times + 1,
            it_ci_traj_per_agent=it_ci_traj_per_agent,
            prediction_len=prediction_len
        )
        print(f"[iter {times+1}] per-agent traj CI CSV saved to: {ci_pos_csv_path}")
        
        ci_csv_path = save_ci_iteration_csv(
            iter_out_dir=iter_out_dir,
            iteration_index=times + 1,
            it_ci_traj_series=it_ci_traj_series,
            it_ci_ctrl_series=it_ci_ctrl_series,
            it_ci_obj=it_ci_obj,
            it_ci_ctrl_cost=it_ci_ctrl_cost,
            prediction_len=prediction_len
        )
        print(f"[iter {times+1}] CI CSV saved to: {ci_csv_path}")
        '''
        # --- (d) heatmap-ready CSVs ---
        traj_pos_csv = save_ci_traj_positions_csv(
            iter_out_dir=iter_out_dir,
            iteration_index=times + 1,
            rows=it_ci_traj_pos_rows
        )
        print(f"[iter {times+1}] CI(traj) positions CSV saved: {traj_pos_csv}")

        ctrl_local_csv = save_ci_ctrl_local_csv(
            iter_out_dir=iter_out_dir,
            iteration_index=times + 1,
            rows=it_ci_ctrl_local_rows
        )
        print(f"[iter {times+1}] CI(ctrl) local CSV saved: {ctrl_local_csv}")
        
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

        if video_writer is not None:
            video_writer.release()
            print(f"[iter {times+1}] wrote video: {video_path}")
        
        if overlay_result is not None:
            overlay_result.close()

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
    print("===================================")
    print("Enter the variables : --goal_x, --goal_y, --num_iter, --r_star, --dataset, --predictor")
    print("--dataset : ETH, Hotel, Univ, Zara01, Zara02, Lobby")
    print("--predictor : linear, gp, eigen, traj, koopcast, SocialVAE, Social-STGCNN")
    print("===================================")
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_x', type=float, default=8.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=0.2)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--r_star', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default="Zara01")
    parser.add_argument('--predictor', type=str, default="SocialVAE")
    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument('--video_fps', type=float, default=2.5)
    #============================================================
    parser.add_argument("--overlay", type=bool, default=True,
                        help="Use homography to draw history/GT/prediction on extracted real frames")
    parser.add_argument("--frame_offset", type=int, default=40,
                        help="Align sim time to real frames (index shift)")
    parser.add_argument("--extracted_fps", type=float, default=2.5,
                        help="FPS used by video_parser.py to extract frames")
    parser.add_argument("--output_fps", type=float, default=10.0,
                        help="Output MP4 FPS; defaults to extracted_fps")
    args = parser.parse_args()

    main(args.goal_x, args.goal_y, args.num_iter, args.r_star, args.dataset, args.predictor, video_fps=args.video_fps, save_video=args.save_video,
         overlay=args.overlay, frame_offset=args.frame_offset, extracted_fps=args.extracted_fps, output_fps=args.output_fps)


