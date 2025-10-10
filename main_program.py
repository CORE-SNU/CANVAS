import time
import argparse
import numpy as np
import cv2
import pathlib
import os
import csv
import sys
_DATA_DIR = os.path.dirname(__file__)
sys.path.append(_DATA_DIR)
from canvas.datasets import get_dataset_spec, _load_background_image
from canvas.controllers.controller import controllers
from canvas import Environment, AdaptiveConformalPredictionModule, Predictors,\
        CompetencyIndex, Predictor_CI, region_to_box,dynamic_observation_filter
from simulation import Simulation

# -----------------------------
# Main
# -----------------------------
def main(dataset, predictor, controller, 
         prediction_len, history_len, start_x, start_y, dt, goal_x, goal_y, max_ped, t_begin, t_end,
         num_iter, video_fps, save_video, frame_offset, extracted_fps, output_fps):
    
    # Predictor horizon
    prediction_len = prediction_len
    history_len = history_len
    # Simulation period
    dt = dt 
    # Choose predictor
    obj_predictor = Predictors(chosen_predictor=predictor,prediction_len=prediction_len,history_len=history_len,dt=dt,dataset=dataset,device='cpu')                                    # Trajectron++ predictor
    # Unified R* (kept for future score/CI; not used in controller)
    rstar = 0.5
    # CI Instance
    ci_traj     = CompetencyIndex(case="traj",      r_star=rstar, return_type="series")
    ci_ctrl     = CompetencyIndex(case="control",   r_star=rstar, return_type="series")
    ci_obj      = CompetencyIndex(case="obj",       r_star=rstar)            # scalar
    ci_ctrlcost = CompetencyIndex(case="ctrl_cost", r_star=rstar)            # scalar
    # Environment setting
    t_begin = t_begin # time step to begin environment in dataset
    t_end   = t_end   # time step to end environment in dataset
    dataset = dataset
    datasets_dir = os.path.join(_DATA_DIR, "canvas", "datasets")
    fname_map = {
        "Lobby":  "0.npy",
        "ETH":    "biwi_eth.npy",
        "Hotel":  "biwi_hotel.npy",
        "Zara01": "crowds_zara01.npy",
        "Zara02": "crowds_zara02.npy",
        "Univ":   "students003.npy",
    }
    npy_path = os.path.join(datasets_dir, fname_map[dataset])
    init_robot_pose = np.array([start_x, start_y, np.pi / 2.]) # Start position for control test
    goal = np.array([goal_x, goal_y]) # Goal position for control test
    persistent_static_boxes = [region_to_box(r) for r in get_dataset_spec(dataset).static_regions]
    env = Environment(
            filepath=npy_path,
            dt=dt,
            init_robot_pose=init_robot_pose,
            t_begin=t_begin,
            t_end=t_end
        )
    # CP module setting (use ACP)
    max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1) # Maximum interval length setting
    offline_calibration_set = {i: [] for i in range(prediction_len)}
    cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                  step_size=0.05,
                                                  n_scores=prediction_len,
                                                  max_interval_lengths=max_interval_lengths,
                                                  sample_size=20,
                                                  offline_calibration_set=offline_calibration_set)
    # Choose controller for control test
    controller = controllers(chosen_controller=controller,prediction_len=prediction_len,dt=dt)
    # Control test simulation setting
    sim = Simulation(environment=env, 
                     predictor=obj_predictor,
                     controller=controller,
                     cp_module=cp_module,
                     goal=goal,
                     max_pedestrian=max_ped,
                     persistent_static_boxes=persistent_static_boxes,
                     dataset=dataset,
                     prediction_len=prediction_len,
                     history_len=history_len,
                     dt=dt,
                     save_video=save_video,
                     video_fps=video_fps,
                     use_overlay=True,
                     frame_offset=frame_offset,
                     extracted_fps=extracted_fps,
                     output_fps=output_fps
                    )
    
    for times in range(num_iter):
        sim.run(times=times)

if __name__ == "__main__":
    print("===================================")
    print("Enter the variables : --goal_x, --goal_y, --num_iter, --r_star, --dataset, --predictor")
    print("--dataset : ETH, Hotel, Univ, Zara01, Zara02, Lobby")
    print("--predictor : linear, gp, eigen, traj, koopcast")
    print("--controller : grid, congromal, sampling, ecp")
    print("===================================")
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_x', type=float, default=0.0)
    parser.add_argument('--start_y', type=float, default=4.0)
    parser.add_argument('--goal_x', type=float, default=8.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=4.2)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="Zara01")
    parser.add_argument('--predictor', type=str, default="traj")
    parser.add_argument('--controller', type=str, default="grid")
    parser.add_argument('--prediction_len', type=int, default=12)
    parser.add_argument('--history_len', type=int, default=8)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--t_begin', type=int, default=40)
    parser.add_argument('--t_end', type=int, default=2000)
    #============================================================
    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument('--video_fps', type=float, default=2.5)
    parser.add_argument("--frame_offset", type=int, default=40,
                        help="Align sim time to real frames (index shift)")
    parser.add_argument("--extracted_fps", type=float, default=2.5,
                        help="FPS used by video_parser.py to extract frames")
    parser.add_argument("--output_fps", type=float, default=10.0,
                        help="Output MP4 FPS; defaults to extracted_fps")
    parser.add_argument("--max_ped", type=float, default=4.0,
                    help="Max pedestrians to consider (others ignored)")
    args = parser.parse_args()

    main(args.dataset, args.predictor, args.controller, 
         args.prediction_len, args.history_len, args.start_x, args.start_y, args.dt, args.goal_x, args.goal_y, args.max_ped, args.t_begin, args.t_end,
         args.num_iter, args.video_fps, args.save_video, args.frame_offset, args.extracted_fps, args.output_fps)

