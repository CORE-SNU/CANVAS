import time
import argparse
import matplotlib
import numpy as np
import os

import sys
_DATA_DIR = os.path.dirname(__file__)

sys.path.append(_DATA_DIR)
from canvas import Environment, Box, GridMPC, AdaptiveConformalPredictionModule, Predictors
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

def collect_pred_errors_and_ci(prediction_res, valid_obs_future_true, base_time, dt, R_star_vec):
    recs = []
    if not prediction_res or not valid_obs_future_true:
        return recs
    common = set(prediction_res.keys()) & set(valid_obs_future_true.keys())
    H = len(R_star_vec)
    for pid in common:
        pred = np.asarray(prediction_res[pid], dtype=np.float64)[:, :2]
        gt   = np.asarray(valid_obs_future_true[pid], dtype=np.float64)[:, :2]
        L = min(len(pred), len(gt), H)
        for j in range(L):                 # j=0..H-1
            err = float(np.linalg.norm(pred[j] - gt[j]))
            rj  = float(R_star_vec[j]) if R_star_vec[j] > 0 else 1e-6
            ci  = (rj - err) / rj          # instantaneous competency idx
            recs.append({
                'pid': pid,
                'step': j + 1,                      # 1..H
                't_pred': base_time + (j + 1) * dt, # future timestamp
                'err': err,
                'ci': ci
            })
    return recs

def main(goal_x, goal_y, num_iter, r_star):
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
        iter_error_ci_records = [] # list of dicts with keys: pid, step(1..H), t_pred(sec), err(m), CI

        goal = np.array([goal_x, goal_y])

        prediction_len = 12
        history_len=8
        R_star_vec = np.full(prediction_len, r_star, dtype=np.float64)
        data_dir = "/home/snowhan1021/tools_paper/CANavi"
        koopman_dir = "/home/snowhan1021/tools_paper/CANavi/koopman"
        obj_predictor = Predictors(chosen_predictor='Linear',prediction_len=prediction_len,history_len=history_len, device='cpu')                                    # Trajectron++ predictor
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
        print("NEED TO IMPLEMENT INITIAL POSITION OF ROBOT LATER")
        init_robot_pose = np.array([0, 0, np.pi / 2.])
        goal = np.array([goal_x, goal_y])
        t_begin = 40
        t_step=t_begin
        t_end=2000
        velocity = np.array([0., 0., ])
        buffer_vel = []
        done = False
        environment = Environment(
                filepath='eth',
                dt=dt,
                init_robot_pose=init_robot_pose,
                history_len=history_len,
                prediction_len=prediction_len,
                t_begin=t_begin,
                t_end=t_end
            )
        position_x, position_y, orientation_z=environment.reset()
        while not done:
            detect_time = time.time()
            linear_x, angular_z = environment.get_velocity()

            # position recording
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # (i) Detecting : Pointcloud to rectangles (bounding boxes)
            # (ii) tracking : object tracking → return (trajectories, object_types)
            observation = environment._get_obs()
            observation_future_true=environment._get_obs_future()
            # Keep only 8-step, all-finite (no NaN/Inf/None) trajectories with at least (x,y)
            valid_obs = {}
            valid_obs_future_true = {}
            dynamic_obs={}
            for pid, traj in observation.items():
                try:
                    arr = np.asarray(traj, dtype=np.float64)   # fails if there are None's
                except (TypeError, ValueError):
                    continue
                if arr.shape[0] == 8 and arr.ndim == 2 and arr.shape[1] >= 2 and np.isfinite(arr).all():
                    valid_obs[pid] = arr
                    valid_obs_future_true[pid] = np.asarray(observation_future_true[pid], dtype=np.float64)

            # (iii) Proceed only if there is at least one valid 8-step trajectory
            if valid_obs:
                dynamic_obs = valid_obs  # your dataset is pedestrians only
                # Create array with the latest (x,y) of each trajectory (last row of the 8)
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
                robot_pos = np.array([position_x, position_y])

                # Euclidean distance to robot
                distances = np.sqrt(np.sum((initial_positions - robot_pos) ** 2, axis=1))

                # Collision if <= 0.7
                collisions = distances <= 0.7
                if np.any(collisions):
                    print("Collision!")
                    collision_count += int(np.sum(collisions))
            else:
                # No valid 8-step observations this step; skip or handle as you like
                pass

            # (iv) predicting : with dynamic object & predictor -> get pedestrian trajectories
            pred_start = time.time()
            prediction_res = obj_predictor(dynamic_obs)
            pred_time = time.time() - pred_start
            buffer_prediction_times.append(pred_time)
            confidence_intervals = cp_module.update(dynamic_obs, prediction_res)

            # (v) Compute competency idx for prediction accuracy
            # record prediction error (pid·timestamp alignment)
            # 'prediction base time' for this frame = (t_begin + frame)*dt
            base_time = (t_begin + frame) * dt
            iter_error_ci_records.extend(
                collect_pred_errors_and_ci(
                    prediction_res, valid_obs_future_true, base_time, dt, R_star_vec
                )
            )


            # For GridMPC
            velocity, info, minimum, intermediate, terminal, control, minimal = controller(pos_x=position_x,
                                        pos_y=position_y,
                                        orientation_z=orientation_z,
                                        linear_x=linear_x,
                                        angular_z=angular_z,
                                        boxes=persistent_static_boxes,
                                        predictions=prediction_res,
                                        confidence_intervals=confidence_intervals,
                                        goal=goal
                                        )
            minimum_cost.append(minimal)
            buffer_intermediate.append(intermediate)
            buffer_terminal.append(terminal)
            buffer_control.append(control)

            buffer_infeasibility.append(info['feasible'])
            if not info['feasible']:
                v, w = buffer_vel[infeasible_streak]
                print(frame, 'No safe paths found, stopping robot movement for this frame.', position_x, position_y, v, w, time.time() - detect_time)
                #robot.sim(v, w)
                environment.step([v,w])
                frame += 1
                infeasible_count += 1
                infeasible_streak += 1

                if infeasible_streak >= max_infeasible_streak:
                    print(frame, 'Infeasible state lasted too long, failed')
                    break
                continue  # Skip to the next frame
            else:
                infeasible_streak = 0

            # Check that if goal is reached(-> stop and exit)
            if np.abs(position_x - goal[0]) < 0.3 and np.abs(position_y - goal[1]) < 0.3:
                print(frame, 'Goal reached!')
                #robot.sim(.0, .0)
                environment.step([0,0])
                is_success = True
                success_count += 1
                travel_time = time.time() - begin
                buffer_travel_times.append(travel_time)
                buffer_pos_x_result.append(buffer_pos_x)
                buffer_pos_y_result.append(buffer_pos_y)
                buffer_avg_minimal_cost.append(np.sum(minimum_cost) / len(minimum_cost))
                buffer_avg_intermediate_cost.append(np.sum(buffer_intermediate) / len(buffer_intermediate))
                buffer_avg_terminal_cost.append(np.sum(buffer_terminal) / len(buffer_terminal))
                buffer_avg_control_cost.append(np.sum(buffer_control) / len(buffer_control))
                break

            cmd_linear_x, cmd_angular_z = velocity[0]
            robot_pose, done=environment.step([cmd_linear_x, cmd_angular_z])
            position_x, position_y, orientation_z=robot_pose
            # Publish the control inputs
            #print("Minimum, intermediate, terminal, control, minimum_idx : ", minimal, intermediate, terminal, control, minimum)
            #print("Percentages - intermediate, terminal, control : ", intermediate/minimal, terminal/minimal, control/minimal)
            frame += 1
            buffer_vel = velocity

        buffer_collision_rate.append(collision_count / frame)
        buffer_infeasible_rate.append(infeasible_count / frame)

        print("Next : #{}_scenario".format(times + 1))
        print("Collision_rate: ", collision_count / frame)
        print("Infeasible_rate: ", infeasible_count / frame)
        print("Avg_prediction_time: ", np.sum(buffer_prediction_times) / len(buffer_prediction_times))
        print('Variance prediction time', np.var(buffer_prediction_times))
        if is_success:
            print("Avg_minimal_cost: ", np.sum(minimum_cost) / len(minimum_cost))
            print("Avg_intermediate_cost: ", np.sum(buffer_intermediate) / len(buffer_intermediate))
            print("Avg_terminal_cost: ", np.sum(buffer_terminal) / len(buffer_terminal))
            print("Avg_control_cost: ", np.sum(buffer_control) / len(buffer_control))
            print("Travel_time: ", travel_time)

        # Print the summary of prediction error for this simulation iteration
        if iter_error_ci_records:
            cis = np.array([r['ci'] for r in iter_error_ci_records], dtype=np.float64)
            ci_mean = float(cis.mean())
            print(f"[Iter {times+1}] CI_mean={ci_mean:.3f} (N={cis.size})")
        else:
            print(f"[Iter {times+1}] CI_mean: no valid records.")

        # =========================================================
        # BLOCK B: IDENTICAL COPY — controller gets GROUND TRUTH
        #         (only change: predictions = valid_obs_future_true)
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
        buffer_pos_x = []
        buffer_pos_y = []

        goal = np.array([goal_x, goal_y])

        prediction_len = 12
        history_len=8
        data_dir = "/home/snowhan1021/tools_paper/CANavi"
        koopman_dir = "/home/snowhan1021/tools_paper/CANavi/koopman"
        obj_predictor = Predictors(chosen_predictor='Linear',prediction_len=prediction_len,history_len=history_len, device='cpu')                                    # Trajectron++ predictor

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
        print("NEED TO IMPLEMENT INITIAL POSITION OF ROBOT LATER")
        init_robot_pose = np.array([0, 0, np.pi / 2.])
        goal = np.array([goal_x, goal_y])
        t_begin = 40
        t_step=t_begin
        t_end=2000
        velocity = np.array([0., 0., ])
        buffer_vel = []
        done = False
        environment = Environment(
                filepath=os.path.join('0.npy'),
                dt=dt,
                init_robot_pose=init_robot_pose,
                t_begin=t_begin,
                t_end=t_end
            )
        position_x, position_y, orientation_z=environment.reset()
        while not done:
            detect_time = time.time()
            linear_x, angular_z = environment.get_velocity()

            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            observation = environment._get_obs()
            observation_future_true=environment._get_obs_future()
            valid_obs = {}
            valid_obs_future_true = {}
            for pid, traj in observation.items():
                try:
                    arr = np.asarray(traj, dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if arr.shape[0] == 8 and arr.ndim == 2 and arr.shape[1] >= 2 and np.isfinite(arr).all():
                    valid_obs[pid] = arr
                    valid_obs_future_true[pid] = np.asarray(observation_future_true[pid], dtype=np.float64)

            if valid_obs:
                dynamic_obs = valid_obs
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
                robot_pos = np.array([position_x, position_y])
                distances = np.sqrt(np.sum((initial_positions - robot_pos) ** 2, axis=1))
                collisions = distances <= 0.7
                if np.any(collisions):
                    print("Collision!")
                    collision_count += int(np.sum(collisions))
            else:
                pass

            # (iv) predicting : HERE WE USE GROUND TRUTH FUTURE
            pred_start = time.time()
            prediction_res = valid_obs_future_true
            pred_time = time.time() - pred_start
            buffer_prediction_times_oracle.append(pred_time)
            confidence_intervals = cp_module.update(dynamic_obs, prediction_res)

            velocity, info, minimum, intermediate, terminal, control, minimal = controller(pos_x=position_x,
                                        pos_y=position_y,
                                        orientation_z=orientation_z,
                                        linear_x=linear_x,
                                        angular_z=angular_z,
                                        boxes=persistent_static_boxes,
                                        predictions=prediction_res,
                                        confidence_intervals=confidence_intervals,
                                        goal=goal
                                        )
            minimum_cost.append(minimal)
            buffer_intermediate_oracle.append(intermediate)
            buffer_terminal_oracle.append(terminal)
            buffer_control_oracle.append(control)

            buffer_infeasibility.append(info['feasible'])
            if not info['feasible']:
                v, w = buffer_vel[infeasible_streak]
                print(frame, 'No safe paths found, stopping robot movement for this frame.', position_x, position_y, v, w, time.time() - detect_time)
                environment.step([v,w])
                frame += 1
                infeasible_count += 1
                infeasible_streak += 1

                if infeasible_streak >= max_infeasible_streak:
                    print(frame, 'Infeasible state lasted too long, failed')
                    break
                continue
            else:
                infeasible_streak = 0

            if np.abs(position_x - goal[0]) < 0.3 and np.abs(position_y - goal[1]) < 0.3:
                print(frame, 'Goal reached!')
                environment.step([0,0])
                is_success = True
                success_count_oracle += 1
                travel_time = time.time() - begin
                buffer_travel_times_oracle.append(travel_time)
                buffer_pos_x_result_oracle.append(buffer_pos_x)
                buffer_pos_y_result_oracle.append(buffer_pos_y)
                buffer_avg_minimal_cost_oracle.append(np.sum(minimum_cost) / len(minimum_cost))
                buffer_avg_intermediate_cost_oracle.append(np.sum(buffer_intermediate_oracle) / len(buffer_intermediate_oracle))
                buffer_avg_terminal_cost_oracle.append(np.sum(buffer_terminal_oracle) / len(buffer_terminal_oracle))
                buffer_avg_control_cost_oracle.append(np.sum(buffer_control_oracle) / len(buffer_control_oracle))
                break

            cmd_linear_x, cmd_angular_z = velocity[0]
            robot_pose, done=environment.step([cmd_linear_x, cmd_angular_z])
            position_x, position_y, orientation_z=robot_pose
            #print("Minimum, intermediate, terminal, control, minimum_idx : ", minimal, intermediate, terminal, control, minimum)
            #print("Percentages - intermediate, terminal, control : ", intermediate/minimal, terminal/minimal, control/minimal)
            frame += 1
            buffer_vel = velocity

        buffer_collision_rate_oracle.append(collision_count / frame)
        buffer_infeasible_rate_oracle.append(infeasible_count / frame)

        print("Next : #{}_scenario_oracle".format(times + 1))
        print("Collision_rate_oracle: ", collision_count / frame)
        print("Infeasible_rate_oracle: ", infeasible_count / frame)
        print("Avg_prediction_time_oracle: ", np.sum(buffer_prediction_times_oracle) / len(buffer_prediction_times_oracle))
        print('Variance prediction time_oracle', np.var(buffer_prediction_times_oracle))
        if is_success:
            print("Avg_minimal_cost_oracle: ", np.sum(minimum_cost) / len(minimum_cost))
            print("Avg_intermediate_cost_oracle: ", np.sum(buffer_intermediate_oracle) / len(buffer_intermediate_oracle))
            print("Avg_terminal_cost_oracle: ", np.sum(buffer_terminal_oracle) / len(buffer_terminal_oracle))
            print("Avg_control_cost_oracle: ", np.sum(buffer_control_oracle) / len(buffer_control_oracle))
            print("Travel_time_oracle: ", travel_time)

    # (Your old commented-out pickle/save block stays as-is if you want to re-enable it.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_x', type=float, default=8.0) # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=0.2) # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=5)
    parser.add_argument('--r_star', type=float, default=0.5)
    args = parser.parse_args()
    main(args.goal_x, args.goal_y, args.num_iter, args.r_star)
