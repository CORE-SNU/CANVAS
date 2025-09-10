import time
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import yaml
import cv2
import pathlib
import rospy
import os
import subprocess
import random
import pickle

import sys
sys.path.append('/home/snowhan1021/tools_paper/CANavi')

from robot_sim import Robot, terminate_dlio_launch
from detection.detector import Detector
from detection.detection_utils import Box
from tracking.tracker import Tracker
from control.grid_solver import GridMPC
# from control.sampling_based_mpc import SamplingBasedMPC
from conformal_prediction.adaptive_cp import AdaptiveConformalPredictionModule
from std_srvs.srv import Empty
from prediction.linear_predictor import LinearPredictor
from trajectron_predictor import TrajectronPredictor
from koopman.koopy_predictor_justmul import KoopmanPredictor
#from koopman.koopman_predictor_clu_geo import KoopmanPredictor
from prediction.eigen.eigen_predictor import eigen_predictor

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


def reset_simulation():
    # Waiting for the service that '/gazebo/reset_simulation' is ready.
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_sim()  # Simulation reset for service called
        rospy.loginfo("Gazebo simulation reset")
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def launch_sim_lobby():
    rospy.loginfo("Starting sim_lobby.launch...")
    cmd = ["roslaunch", "/home/snowhan1021/tools_paper/CANavi/jackal_ws/src/gazebo-rossim/launch/sim_lobby.launch"]
    # need to get jackal_ws/src/gazebo-rossim/launch/sim_lobby.launch
    process = subprocess.Popen(cmd)
    rospy.loginfo("sim_lobby.launch started with PID: %d", process.pid)
    return process

def shutdown_simulation(process):
    if process is not None:
        rospy.loginfo("Shutting down sim_lobby.launch (PID: %d)...", process.pid)
        process.terminate()
        process.wait()
        rospy.loginfo("sim_lobby.launch terminated.")

def main(goal_x, goal_y, num_iter):
    # odometry/filtered rate : 50Hz / ouster/points rate : 10Hz
    dt = 0.10
    r = rospy.Rate(1.0 / dt)

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

    for times in range(num_iter):
        print("--------------------------------")
        print("SIM_LOBBY_{}".format(times))
        print("--------------------------------")
        sim_lobby_process = launch_sim_lobby()
        time.sleep(10)

        robot = Robot()
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

        goal = np.array([goal_x, goal_y])

        prediction_len = 16
        data_dir = "/home/snowhan1021/tools_paper/CANavi"
        koopman_dir = "/home/snowhan1021/tools_paper/CANavi/koopman"
        #user paser to find this.
        result_path = ('/home/snowhan1021/tools_paper/CANavi/scenario/result_linear.pkl')
        # modify this code section to have both ground truth dataset and data_dir
        # basically for one side the controller recieve truth
        # while the other recieves the obj_preictor as shown below.
        obj_detector = Detector()
        obj_tracker = Tracker()
        #obj_predictor = LinearPredictor(prediction_len=prediction_len, history_len=8, smoothing_factor=0.75, dt=dt)                            # Linear predictor
        obj_predictor = KoopmanPredictor(prediction_len=prediction_len, data_dir=data_dir, min_samples=100, dt=dt, pattern=r'^.*\d{2}\.npy$')  # Koopman predictor
        #obj_predictor = TrajectronPredictor(prediction_len=prediction_len, model_dir=data_dir, device='cpu')                                    # Trajectron++ predictor
        #obj_predictor = eigen_predictor                                                                                                        # EigenTrajectory predictor
        # issue solveable utilizing parser and a slight bit of compiler/text(?) language
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

        sim_time_target = rospy.Time(0) + rospy.Duration(7.5)
        while rospy.Time.now() < sim_time_target:
            rospy.sleep(0.05)

        begin = time.time()

        buffer_vel = []
        while not rospy.is_shutdown():
            if robot.pcd is None or robot.ordered_state is None or robot.velocity is None:
                rospy.loginfo("Waiting for sensor data...")
                r.sleep()
                continue

            detect_time = time.time()
            position_x, position_y, orientation_z, timestamp = robot.ordered_state
            buffer_timestamp.append(timestamp)
            linear_x, angular_z = robot.velocity

            # position recording
            buffer_pos_x.append(position_x)
            buffer_pos_y.append(position_y)

            # (i) Detecting : Pointcloud to rectangles (bounding boxes)
            detection_res = obj_detector(robot.pcd,
                                         transformation={'position_x': position_x,
                                                         'position_y': position_y,
                                                         'orientation_z': orientation_z}
                                         )

            # (ii) tracking : object tracking → return (trajectories, object_types)
            tracking_trajectories, object_types = obj_tracker(detection_res)

            # (iii) object type classification : get only dynamic objects
            dynamic_obs = {obj_id: traj for obj_id, traj in tracking_trajectories.items() if object_types.get(obj_id) == 'dynamic'}

            if dynamic_obs:
                # Create array with the first (x,y) of each trajectory
                initial_positions = np.array([traj[-1, :2] for traj in dynamic_obs.values()])
                robot_pos = np.array([position_x, position_y])
                distances_sq = (np.sum((initial_positions - robot_pos) ** 2, axis=1)) ** (1 / 2)
                # Collision if <= 0.7 
                collisions = distances_sq <= 0.7
                if np.any(collisions):
                    print("Collision!")
                    collision_count += int(np.sum(collisions))
            # how to make this simple. depending on program it may have different history length as a start
            # 
            # (iv) predicting : with dynamic object & predictor -> get pedestrian trajectories
            pred_start = time.time()
            prediction_res = obj_predictor(dynamic_obs)
            pred_time = time.time() - pred_start
            buffer_prediction_times.append(pred_time)
            confidence_intervals = cp_module.update(dynamic_obs, prediction_res)

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
                robot.sim(v, w)
                r.sleep()
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
                robot.sim(.0, .0)
                r.sleep()
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

            # Publish the control inputs
            robot.sim(cmd_linear_x, cmd_angular_z)
            print(frame, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, time.time() - detect_time)
            print("Minimum, intermediate, terminal, control, minimum_idx : ", minimal, intermediate, terminal, control, minimum)
            print("Percentages - intermediate, terminal, control : ", intermediate/minimal, terminal/minimal, control/minimal)

            r.sleep()
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

        rospy.loginfo("Iteration %d: Shutting down sim_lobby.launch", times + 1)

        # Shut down exisiting 'sim_lobby.launch'
        shutdown_simulation(sim_lobby_process)
        time.sleep(30)

        rospy.loginfo("Iteration %d: Shut down sim_lobby.launch!", times + 1)

    success_rate = success_count / 50

    # Save as pickle file
    save_arrays = {
        'collision_rate': buffer_collision_rate,
        'infeasible_rate': buffer_infeasible_rate,
        'avg_minimal_cost': buffer_avg_minimal_cost,
        'avg_intermediate_cost': buffer_avg_intermediate_cost,
        'avg_terminal_cost': buffer_avg_terminal_cost,
        'avg_control_cost': buffer_avg_control_cost,
        'prediction_time': buffer_prediction_times,
        'travel_time': buffer_travel_times,
        'success_rate': [success_rate],
        'pos_x_result': buffer_pos_x_result,
        'pos_y_result': buffer_pos_y_result
    }

    # Load exisiting data if the file exist
    if os.path.exists(result_path):
        with open(result_path, 'rb') as file:
            loaded = pickle.load(file)

        # If the loaded data is 'list', convert it to 'dict'
        if isinstance(loaded, list):
            data = {}
            for entry in loaded:
                for key, value in entry.items():
                    if key in data:
                        if isinstance(data[key], list):
                            if isinstance(value, list):
                                data[key].extend(value)
                            else:
                                data[key].append(value)
                        else:
                            data[key] = [data[key]]
                            data[key].append(value)
                    else:
                        data[key] = value if isinstance(value, list) else [value]
        # If the loaded data is 'dict', just use it
        elif isinstance(loaded, dict):
            data = loaded
            for key in data:
                if not isinstance(data[key], list):
                    data[key] = [data[key]]
        else:
            data = {key: [] for key in save_arrays}
    else:
        data = {key: [] for key in save_arrays}

    for key, new_val in save_arrays.items():
        if not isinstance(new_val, list):
            new_val = [new_val]
        if key in data:
            data[key].extend(new_val)
        else:
            data[key] = new_val

    with open(result_path, 'wb') as file:
        pickle.dump(data, file)

    print("Iteration & simulation finished! Saved result.pkl!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_x', type=float, default=8.0) # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=0.2) # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=5)
    rospy.init_node("navigation", anonymous=True)
    args = parser.parse_args()
    main(args.goal_x, args.goal_y, args.num_iter)
