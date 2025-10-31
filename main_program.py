from __future__ import annotations
import argparse
import numpy as np
import math, time
import os
import sys
_DATA_DIR = os.path.dirname(__file__)
sys.path.append(_DATA_DIR)
from canvas.datasets import Dataset, get_dataset_spec, RegisteredDatasets
from canvas.envs.env import Environment
from canvas import Predictors, region_to_box
from simulation import Simulation

# -----------------------------
# Main
# -----------------------------
def main(dataset, predictor, controller, 
         prediction_len, history_len, start_x, start_y, goal_x, goal_y, t_begin, t_end,
         num_iter, save_video, ci_mode):
    # Predictor horizon
    prediction_len = prediction_len
    history_len = history_len
    # Environment setting
    t_begin = t_begin # time step to begin environment in dataset
    t_end   = t_end   # time step to end environment in dataset
    dataset_obj = RegisteredDatasets[dataset]
    init_robot_pose = {"position_x": start_x, "position_y": start_y, "orientation_z": np.pi/2.} # Start position for control test
    goal = np.array([goal_x, goal_y]) # Goal position for control test
    persistent_static_boxes = [region_to_box(r) for r in get_dataset_spec(dataset).static_regions]
    env = Environment(
            dataset=dataset_obj,
            init_robot_state=init_robot_pose,
            goal_pos=goal,
            t_begin=t_begin,
            t_end=t_end,
            history_len=history_len,
            prediction_horizon=prediction_len,
            path_to_frames='/home/core/Documents/CANVAS/assets/frames',
            path_to_save='./viz_example'
        ) 
    # Control test simulation setting
    sim = Simulation(environment=env, 
                     predictor=predictor,
                     controller=controller,
                     goal=goal,
                     persistent_static_boxes=persistent_static_boxes,
                     dataset=dataset_obj,
                     prediction_len=prediction_len,
                     history_len=history_len,
                     dt=env.dt,
                     t_begin=t_begin,
                     t_end=t_end,
                     save_video=save_video,
                     ci_mode=ci_mode
                    )
    
    for times in range(num_iter):
        sim.run(times=times)

if __name__ == "__main__":
    print("===================================")
    print("Enter the variables : --goal_x, --goal_y, --num_iter, --dataset, --predictor")
    print("--dataset : eth, hotel, univ, zara1, zara2, snu-asri, snu-asri-ood")
    print("--predictor : linear, eigen, traj, koopcast, socialvae, stgcnn")
    print("--controller : grid, conformal, sampling, ecp_mpc, mpc, mppi")
    print("===================================")
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_x', type=float, default=0.0)
    parser.add_argument('--start_y', type=float, default=4.0)
    parser.add_argument('--goal_x', type=float, default=13.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=9.2)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="zara1")
    parser.add_argument('--predictor', type=str, default="eigen")
    parser.add_argument('--controller', type=str, default="mppi")
    parser.add_argument('--prediction_len', type=int, default=12)
    parser.add_argument('--history_len', type=int, default=8)
    parser.add_argument('--t_begin', type=int, default=40)
    parser.add_argument('--t_end', type=int, default=200)
    parser.add_argument('--ci_mode', type=str, default='pos', choices=['pos','act','reg'])
    parser.add_argument('--save_video', type=bool, default=True)
    args = parser.parse_args()

    main(args.dataset, args.predictor, args.controller, 
         args.prediction_len, args.history_len, args.start_x, args.start_y, args.goal_x, args.goal_y, args.t_begin, args.t_end,
         args.num_iter, args.save_video, args.ci_mode)

