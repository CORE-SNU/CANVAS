import time
import numpy as np
import pathlib
import os
import sys
_DATA_DIR = os.path.dirname(__file__)

sys.path.append(_DATA_DIR)
from canvas.datasets import get_dataset_spec
from canvas import dynamic_observation_filter

"""
Simulation pipeline (per frame):

  input (history/GT-future) -> predictor -> prediction output -> controller ->
  control output (apply first timestep) / cost function (minimal, inter, term, ctrl)

- Predictor is called ONCE per frame
- Controller (MPC) is called ONCE per frame with predictor outputs
- CP (adaptive conformal) is updated ONCE per frame using current observations & predictions
"""
class Simulation():
    def __init__(self, environment, predictor, controller, cp_module, goal, max_pedestrian, persistent_static_boxes, dataset, 
                 prediction_len, history_len, dt, save_video=True, video_fps=2.5, use_overlay=True, frame_offset=40, extracted_fps=2.5, output_fps=10.0):
        self.env = environment
        self.predictor = predictor
        self.controller = controller
        self.cp_module = cp_module
        self.goal = goal
        self.max_ped = max_pedestrian
        self.persistent_static_boxes = persistent_static_boxes
        self.dataset_obj = dataset
        self.dataset_name = dataset.name
        self.prediction_len = prediction_len
        self.history_len = history_len
        self.dt = dt
        self.save_video = save_video
        self.video_fps = video_fps
        self.use_overlay = use_overlay
        self.frame_offset = frame_offset
        self.extracted_fps = extracted_fps
        self.output_fps = output_fps

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

    def run(self, times: int):
        print("==================================")
        print("SIMULATION PIPELINE Started")
        print("==================================")
        frame = 0
        infeasible_count = 0
        infeasible_streak = 0
        max_infeasible_streak = 10
        collision_count = 0
        is_success = False

        rstar = 0.24
        CI_t = 3

        buffer_infeasibility = []
        minimum_cost = []
        buffer_pos_x = []  # per-frame x within this run
        buffer_pos_y = []
        buffer_intermediate = []
        buffer_terminal = []
        buffer_control = []
        ci_data = []

        spec = get_dataset_spec(self.dataset_name)

        cp_module = self.cp_module
        cp_module_gt = self.cp_module
        buffer_vel = []
        done = False

        obs, side = self.env.reset()
        ego = obs['ego']
        position_x, position_y, orientation_z = ego['position_x'], ego['position_y'], ego['orientation_z']
        last_cmd = (0.0, 0.0)
        begin = time.time()

        self.set_buffer()       

        while not done:
            detect_time = time.time()
            linear_x, angular_z = last_cmd

            # record robot trajectory
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # --------- Observations (history & GT futures) ---------
            observation = obs
            observation_future_true = side.get('future', {})
            valid_obs, valid_obs_future_true = dynamic_observation_filter(
                observation, position_x, position_y, self.prediction_len, observation_future_true, self.max_ped
            )

            # --------- Predictor (once per frame) ---------
            pred_start = time.time()
            prediction_res = self.predictor(valid_obs if valid_obs else {})
            pred_time = time.time() - pred_start
            self.buffer_prediction_times.append(pred_time)

            # --------- CP update (once per frame, pred vs gt) ---------
            confidence_intervals   = cp_module.update(valid_obs, prediction_res if isinstance(prediction_res, dict) else {})
            confidence_intervals_gt= cp_module_gt.update(valid_obs, valid_obs_future_true if isinstance(valid_obs_future_true, dict) else {})

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
                goal=self.goal,
                history = valid_obs
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
                goal=self.goal,
                history = valid_obs
            )

            # --------- Visualization ---------
            ci_data.append(rstar/(rstar + confidence_intervals[CI_t]))
            if self.save_video:
                try:
                    try:
                        self.env.render(ci_series=ci_data, cbar_label='CI Control Prediction')
                    except TypeError:
                        self.env.render()
                except Exception as e:
                    print(f"[WARN] Render failed at frame {frame}: {e}")
            
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
                self.success_count += 1
                travel_time = time.time() - begin
                break

            # --------- Apply first control step ---------
            if velocity is not None and len(velocity) > 0:
                cmd_linear_x, cmd_angular_z = velocity[0]
            else:
                cmd_linear_x, cmd_angular_z = 0.0, 0.0
            obs, terminated, truncated, info = self.env.step([cmd_linear_x, cmd_angular_z])
            ego = obs['ego']
            position_x, position_y, orientation_z = ego['position_x'], ego['position_y'], ego['orientation_z']
            done = terminated or truncated
            side = info
            last_cmd = (cmd_linear_x, cmd_angular_z)
            print(frame, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, time.time() - detect_time)

            # --------- Accumulate costs ---------
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            frame += 1
            buffer_vel = velocity

        #if ci_data:
        #    save_ci_traj_positions_csv(iter_out_dir=iter_out_dir, iteration_index=times+1, rows=ci_data)

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





