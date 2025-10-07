import time
import argparse
import numpy as np
import cv2
import pathlib
import os
import sys
_DATA_DIR = os.path.dirname(__file__)

sys.path.append(_DATA_DIR)
from canvas.datasets import get_dataset_spec, _load_background_image
from canvas import Environment, GridMPC, \
    AdaptiveConformalPredictionModule, Predictors, CompetencyIndex, Predictor_CI
from save_ci import save_ci_traj_positions_csv, save_ci_ctrl_local_csv, project_ctrl_step_to_local_xy, save_frame_png

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""
class Simulation():
    def __init__(self, environment, predictor, controller, cp_module, goal, max_pedestrian, persistent_static_boxes, dataset, prediction_len):
        self.env = environment
        self.predictor = predictor
        self.controller = controller
        self.cp_module = cp_module
        self.goal = goal
        self.max_ped = max_pedestrian
        self.persistent_static_boxes = persistent_static_boxes
        self.dataset = dataset
        self.prediction_len = prediction_len

        self.buffer_collision_rate = []
        self.buffer_infeasible_rate = []
        self.buffer_avg_minimal_cost = []
        self.buffer_avg_intermediate_cost = []
        self.buffer_avg_terminal_cost = []
        self.buffer_avg_control_cost = []
        self.buffer_prediction_times = []
        self.buffer_travel_times = []
        self.success_count = 0
        self.buffer_pos_x_result = []
        self.buffer_pos_y_result = []

    def set_buffer(self):
        self.buffer_collision_rate = []
        self.buffer_infeasible_rate = []
        self.buffer_avg_minimal_cost = []
        self.buffer_avg_intermediate_cost = []
        self.buffer_avg_terminal_cost = []
        self.buffer_avg_control_cost = []
        self.buffer_prediction_times = []
        self.buffer_travel_times = []
        self.success_count = 0
        self.buffer_pos_x_result = []
        self.buffer_pos_y_result = []

    def save_video(self):
        return

    def run(self, times):
        print("==================================")
        print("SIMULATION PIPELINE Started")
        print("==================================")
        frame = 0
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []  # per-frame x within this run
        buffer_pos_y = []
        buffer_intermediate = []
        buffer_terminal = []
        buffer_control = []

        spec = get_dataset_spec(self.dataset)
        bg_img = _load_background_image(spec.bg.path, spec.bg.rotate90)

        cp_module = self.cp_module
        cp_module_gt = self.cp_module
        buffer_vel = []
        done = False

        # iteration output dir for 'viz'
        iter_out_dir = pathlib.Path("viz") / f"iter_{times+1:03d}"
        iter_out_dir.mkdir(parents=True, exist_ok=True)

        position_x, position_y, orientation_z = self.env.reset()
        begin = time.time()

        self.set_buffer()

        video_writer = None
        video_path = iter_out_dir / f"sim_iter_{times+1:03d}.mp4"

        while not done:
            detect_time = time.time()
            linear_x, angular_z = self.env.get_velocity()

            # record robot trajectory
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # --------- Observations (history & GT futures) ---------
            observation = self.env._get_obs()
            observation_future_true = self.env._get_obs_future()

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
                    valid_obs_future_true[pid] = fut[:self.prediction_len, :2]

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
            prediction_res = self.predictor(dynamic_obs if dynamic_obs else {})
            pred_time = time.time() - pred_start
            self.buffer_prediction_times.append(pred_time)

            # --------- CP update (once per frame) ---------
            confidence_intervals = cp_module.update(dynamic_obs, prediction_res if isinstance(prediction_res, dict) else {})
            confidence_intervals_gt=cp_module_gt.update(dynamic_obs, valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {})

            # --------- Controller (once per frame, with predictions) ---------
            velocity, info, minimum, intermediate, terminal, control, minimal = self.controller(
                pos_x=position_x,
                pos_y=position_y,
                orientation_z=orientation_z,
                linear_x=linear_x,
                angular_z=angular_z,
                boxes=self.persistent_static_boxes,
                predictions=prediction_res if isinstance(prediction_res, dict) else {},
                confidence_intervals=confidence_intervals,
                goal=self.goal
            )

            # For GT(Oracle based) : no status update here, just for get controller input for GT
            velocity_gt, info_gt, minimum_gt, intermediate_gt, terminal_gt, control_gt, minimal_gt = self.controller(
                pos_x=position_x,
                pos_y=position_y,
                orientation_z=orientation_z,
                linear_x=linear_x,
                angular_z=angular_z,
                boxes=self.persistent_static_boxes,
                predictions=valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {},
                confidence_intervals=confidence_intervals_gt,
                goal=self.goal
            )
            prediction_competency=Predictor_CI()
            prediction_comptency_score=prediction_competency.CI_default(confidence_intervals)
            
            # --------- Feasibility handling ---------
            buffer_infeasibility.append(info.get('feasible', True))
            if not info.get('feasible', True):
                if infeasible_streak < len(buffer_vel):
                    v, w = buffer_vel[infeasible_streak]
                else:
                    v, w = 0.0, 0.0
                print(frame, 'No safe paths found, stopping robot movement for this frame.',
                      position_x, position_y, v, w, time.time() - detect_time)
                self.env.step([v, w])
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
            if np.abs(position_x - self.goal[0]) < 0.3 and np.abs(position_y - self.goal[1]) < 0.3:
                print(frame, 'Goal reached!')
                self.env.step([0, 0])
                is_success = True
                success_count += 1
                travel_time = time.time() - begin
                break

            # --------- Apply first control step ---------
            if velocity is not None and len(velocity) > 0:
                cmd_linear_x, cmd_angular_z = velocity[0]
            else:
                cmd_linear_x, cmd_angular_z = 0.0, 0.0
            robot_pose, done = self.env.step([cmd_linear_x, cmd_angular_z])
            position_x, position_y, orientation_z = robot_pose
            print(frame, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, time.time() - detect_time)

            # --------- Accumulate costs ---------
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            frame += 1
            buffer_vel = velocity
        
        # ---- Iteration-level rates and summaries ----
        self.buffer_collision_rate.append(collision_count / max(1, frame))
        self.buffer_infeasible_rate.append(infeasible_count / max(1, frame))

        print("Next : #{}_scenario".format(times + 1))
        print("Collision_rate: ", collision_count / max(1, frame))
        print("Infeasible_rate: ", infeasible_count / max(1, frame))
        if self.buffer_prediction_times:
            print("Avg_prediction_time: ", np.sum(self.buffer_prediction_times) / len(self.buffer_prediction_times))
            print('Variance prediction time', np.var(self.buffer_prediction_times))
        if is_success and minimum_cost:
            print("Avg_minimal_cost: ", np.sum(minimum_cost) / len(minimum_cost))
            print("Avg_intermediate_cost: ", np.sum(buffer_intermediate) / len(buffer_intermediate) if buffer_intermediate else np.nan)
            print("Avg_terminal_cost: ", np.sum(buffer_terminal) / len(buffer_terminal) if buffer_terminal else np.nan)
            print("Avg_control_cost: ", np.sum(buffer_control) / len(buffer_control) if buffer_control else np.nan)
            print("Travel_time: ", travel_time)





