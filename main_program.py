from __future__ import annotations
import argparse
import numpy as np
import math, time
import os
import sys
_DATA_DIR = os.path.dirname(__file__)
sys.path.append(_DATA_DIR)
from canvas.datasets import Dataset, get_dataset_spec, RegisteredDatasets
from canvas.controllers.controller import controllers
from canvas.envs.env import Environment
from canvas import Predictors, region_to_box
from simulation import Simulation

# -----------------------------
# Main
# -----------------------------
def main(dataset, predictor, controller, 
         prediction_len, history_len, start_x, start_y, goal_x, goal_y, max_ped, t_begin, t_end,
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
                     max_pedestrian=max_ped,
                     persistent_static_boxes=persistent_static_boxes,
                     dataset=dataset_obj,
                     prediction_len=prediction_len,
                     history_len=history_len,
                     dt=env.dt,
                     t_begin=t_begin,
                     t_end=t_end,
                     save_video=save_video,
                     ci_mode=ci_mode,
                     verbose=(num_iter == 1)
                    )
    
    for times in range(num_iter):
        if num_iter !=1:
            init_pose, goal = sample_start_goal_random(
                dataset_name=dataset,                       # args.dataset
                persistent_static_boxes=persistent_static_boxes,
                min_goal_dist=2.0, margin=0.5,
                rng=1234 + times 
            )

            sim.goal = goal
            try:
                sim.env._goal = goal
            except Exception:
                pass
            try:
                sim.env._init_state = init_pose
            except Exception:
                pass

        sim.run(times=times)
        print(
            f"[FINAL] dataset={dataset}, predictor={predictor}, "
            f"run={times+1}/{num_iter}, "
            f"mean_of_frame_means={sim.overall_frame_mean_ci():.6f}"
        )

# -----------------------------
# Temporary utility : for randomize start/goal positions
# -----------------------------

def _normalize_extent(extent):
    xmin, xmax, ymin, ymax = map(float, extent)
    if xmin > xmax: xmin, xmax = xmax, xmin
    if ymin > ymax: ymin, ymax = ymax, ymin
    return xmin, xmax, ymin, ymax

def _inside_extent(x, y, extent, margin=0.0):
    xmin, xmax, ymin, ymax = _normalize_extent(extent)
    return (xmin + margin) <= x <= (xmax - margin) and (ymin + margin) <= y <= (ymax - margin)

def _point_in_rotbox(x, y, box, eps=0.0):
    if hasattr(box, "x") and hasattr(box, "y") and hasattr(box, "w") and hasattr(box, "h"):
        cx, cy = float(box.x), float(box.y)
        w, h   = float(box.w), float(box.h)  
        rad    = float(getattr(box, "rad", 0.0))
        
        cosr, sinr = math.cos(-rad), math.sin(-rad)
        dx, dy = float(x) - cx, float(y) - cy
        lx = cosr * dx - sinr * dy
        ly = sinr * dx + cosr * dy
        halfw = w * 0.5 + eps
        halfh = h * 0.5 + eps
        return (-halfw <= lx <= halfw) and (-halfh <= ly <= halfh)

    if isinstance(box, dict):
        for keys in [
            ("xmin", "xmax", "ymin", "ymax"),
            ("x_min", "x_max", "y_min", "y_max"),
            ("min_x", "max_x", "min_y", "max_y"),
            ("left", "right", "bottom", "top"),
        ]:
            if all(k in box for k in keys):
                bxmin = float(box[keys[0]]); bxmax = float(box[keys[1]])
                bymin = float(box[keys[2]]); bymax = float(box[keys[3]])
                return (bxmin - eps) <= x <= (bxmax + eps) and (bymin - eps) <= y <= (bymax + eps)

    if isinstance(box, (tuple, list)) and len(box) == 4:
        a, b, c, d = map(float, box)
        xmin, xmax = (min(a, c), max(a, c))
        ymin, ymax = (min(b, d), max(b, d))
        return (xmin - eps) <= x <= (xmax + eps) and (ymin - eps) <= y <= (ymax + eps)

    if hasattr(box, "vertices"):
        verts = np.asarray(box.vertices, dtype=float)
        bxmin, bymin = float(verts[:,0].min()), float(verts[:,1].min())
        bxmax, bymax = float(verts[:,0].max()), float(verts[:,1].max())
        return (bxmin - eps) <= x <= (bxmax + eps) and (bymin - eps) <= y <= (bymax + eps)

    return False

def _inside_any_forbidden(x, y, boxes, margin=0.0):
    if not boxes: return False
    for b in boxes:
        if _point_in_rotbox(x, y, b, eps=margin):
            return True
    return False

def sample_start_goal_random(dataset_name, persistent_static_boxes,
                             min_goal_dist=2.0, margin=0.05, max_tries=4000,
                             rng=None, orientation=np.pi/2):
    spec = get_dataset_spec(dataset_name)
    extent = spec.bg.extent  # (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = _normalize_extent(extent)

    if rng is None:
        rng = np.random.default_rng((time.time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFF)
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))

    def sample_point():
        for _ in range(max_tries):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if _inside_extent(x, y, extent, margin) and not _inside_any_forbidden(x, y, persistent_static_boxes, margin):
                return float(x), float(y)
        raise RuntimeError("Failed to sample a valid point (extent/forbidden). Try smaller margin or fewer boxes.")

    sx, sy = sample_point()
    for _ in range(max_tries):
        gx, gy = sample_point()
        if ((gx - sx)**2 + (gy - sy)**2) ** 0.5 >= float(min_goal_dist):
            break
    else:
        raise RuntimeError("Failed to sample goal respecting min_goal_dist.")

    init_pose = {"position_x": sx, "position_y": sy, "orientation_z": float(orientation)}
    goal = np.array([gx, gy], dtype=float)
    return init_pose, goal

# -----------------------------

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
    #============================================================
    parser.add_argument('--save_video', type=bool, default=True)
    parser.add_argument("--max_ped", type=int, default=4,
                    help="Max pedestrians to consider (others ignored)")
    args = parser.parse_args()

    main(args.dataset, args.predictor, args.controller, 
         args.prediction_len, args.history_len, args.start_x, args.start_y, args.goal_x, args.goal_y, args.max_ped, args.t_begin, args.t_end,
         args.num_iter, args.save_video, args.ci_mode)

