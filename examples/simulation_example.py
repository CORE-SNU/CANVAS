import time
import argparse
import numpy as np

import os
import matplotlib.pyplot as plt

from canvas.datasets import get_dataset_spec
from canvas.datasets import RegisteredDatasets
from canvas.controllers import GridMPC
from canvas.envs.env_new import Environment
from canvas import AdaptiveConformalPredictionModule, Predictors, region_to_box

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


# -----------------------------
# Main
# -----------------------------
def main(goal_x, goal_y, num_iter, dataset_name, predictor):
    path_to_save = 'viz_example'
    _DATA_DIR = os.path.dirname(os.path.dirname(__file__))
    # TODO: fix this
    persistent_static_boxes = [region_to_box(r) for r in get_dataset_spec(dataset_name).static_regions]

    init_robot_pose = {
        'position_x': 12.,
        'position_y': 5.,
        'orientation_z': np.pi
    }
    goal_pos = np.array([goal_x, goal_y])
    t_begin = 40
    t_end = 200

    dataset = RegisteredDatasets[dataset_name]

    # Predictor horizon
    prediction_horizon = 12
    history_len = 8

    env = Environment(
        dataset=dataset,
        init_robot_state=init_robot_pose,
        goal_pos=goal_pos,
        t_begin=t_begin,
        t_end=t_end,
        history_len=history_len,
        prediction_horizon=prediction_horizon,
        path_to_frames='/media/sju5379/F6340D35340CF9FF/euped_assets/frames',
        # directory from which the parsed frames are loaded
        path_to_save='./viz_example'  # directory to save the visualization result
    )

    # -----------------------------
    # GLOBAL BUFFERS (logging)
    # -----------------------------
    # TODO: add a logger to manage these


    for times in range(num_iter):
        print("==================================")
        print("SIMULATION PIPELINE Started")


        # ---- Choose predictor ----
        obj_predictor = Predictors(
            chosen_predictor=predictor,
            prediction_len=prediction_horizon,
            history_len=history_len,
            dt=env.dt,
            dataset=dataset_name,
            device='cpu'
        )  # Trajectron++ predictor

        controller = GridMPC(n_steps=prediction_horizon, dt=env.dt)
        # controller_gt = GridMPC(n_steps=prediction_horizon, dt=dt)  # for GT/oracle control input

        # ---- CP module (updated once per frame) ----
        max_interval_lengths = 0.3 * env.dt * np.arange(1, prediction_horizon + 1)
        offline_calibration_set = {i: [] for i in range(prediction_horizon)}
        cp_module = AdaptiveConformalPredictionModule(
            target_miscoverage_level=0.2,
            step_size=0.05,
            n_scores=prediction_horizon,
            max_interval_lengths=max_interval_lengths,
            sample_size=20,
            offline_calibration_set=offline_calibration_set
        )

        obs, simulation_info = env.reset()
        truncated = False

        frame = 0

        while not truncated:
            # simulation loop

            # record robot trajectory
            position_x = obs['ego']['position_x']
            position_y = obs['ego']['position_y']

            # --------- Predictor (once per frame) ---------
            prediction_res = obj_predictor(obs['non-ego'])

            # --------- CP update (once per frame) ---------
            confidence_intervals = cp_module.update(
                obs['non-ego'],
                prediction_res
            )

            # --------- Controller (once per frame, with predictions) ---------
            velocity, controller_info, minimum, intermediate, terminal, control, minimal = controller(
                **obs['ego'],
                boxes=persistent_static_boxes,
                predictions=prediction_res,
                confidence_intervals=confidence_intervals,
                goal=goal_pos
            )

            # For GT(Oracle based) : no status update here, just for get controller input for GT
            # TODO: implement the lagged ACI; this is cheating
            '''
            velocity_gt, controller_info_gt, minimum_gt, intermediate_gt, terminal_gt, control_gt, minimal_gt = controller_gt(
                **scene_obs['ego'],
                boxes=persistent_static_boxes,
                predictions=valid_obs_future_true,
                confidence_intervals=np.zeros(prediction_len),
                goal=goal
            )
            '''

            # --------- Feasibility handling ---------

            if not controller_info['feasible']:
                cmd_linear_x, cmd_angular_z = 0., 0.
            else:
                cmd_linear_x, cmd_angular_z = velocity[0]

            # forward the env
            action = np.array([cmd_linear_x, cmd_angular_z])
            obs, terminated, truncated, simulation_info = env.step(action)

            fig, ax = env.render()

            fig.savefig(os.path.join(path_to_save, '{:03d}.png'.format(env.timestep)), bbox_inches='tight', pad_inches=0)
            plt.close()

            # --------- Goal check ---------
            if terminated:
                print('[frame {}] Goal reached!'.format(frame))
                break

            frame += 1

    return

if __name__ == "__main__":
    # TODO: write -h?
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="zara1")
    parser.add_argument('--predictor', type=str, default="traj")

    parser.add_argument('--goal_x', type=float, default=3.0)  # 8.0 , 6.0
    parser.add_argument('--goal_y', type=float, default=6.0)  # 0.2 , -6.0
    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument('--video_fps', type=float, default=2.5)

    parser.add_argument("--overlay", type=bool, default=True,
                        help="Use homography to draw history/GT/prediction on extracted real frames")
    parser.add_argument("--frame_offset", type=int, default=40,
                        help="Align sim time to real frames (index shift)")
    parser.add_argument("--extracted_fps", type=float, default=2.5,
                        help="FPS used by video_parser.py to extract frames")
    parser.add_argument("--output_fps", type=float, default=10.0,
                        help="Output MP4 FPS; defaults to extracted_fps")
    parser.add_argument("--max_ped", type=float, default=3.0,
                        help="Max pedestrians to consider (others ignored)")
    parser.add_argument("--cont_CI", type=str, default="traj",
                        help="Continuous CI type: traj, control, obj, ctrl_cost to map on video.")
    parser.add_argument("--CI_t", type=int, default=3,
                        help="CI to use for pred.")
    args = parser.parse_args()
    main(args.goal_x, args.goal_y, args.num_iter, args.dataset, args.predictor)
