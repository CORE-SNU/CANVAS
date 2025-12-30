



# CANVAS: Competency-Aware Navigation Assessment for Pedestrian Trajectory Forecasting

## Installation
Run
```
git clone --recurse-submodules https://github.com/CORE-SNU/CANVAS.git
```
To use MPPI controllers as baselines, run 
```
cd CANVAS
pip install -e canvas/controllers/pytorch_mppi
```
This installs the MPPI implementation forked from [pytorch-mppi][pytorch-mppi-link]. Note that the requires pytorch to be installed in advance.


Below is a sample run code for the bare minimum imports required to call the predictors and competency index implemented into our code.




```python
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
            path_to_frames='~/CANVAS/assets/frames',
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
```

---

## Running the simulation

You can run the simulation with "main_program.py" with some dedicated variables

* --start_x, --start_y : Start position for control test (default : (0.0, 4.0))
* --goal_x, --goal_y : Goal position for control test (default : (8.0, 4.2))
* --num_iter : Number of iterations for simulation (default : 1)
* --dataset : must be chosen from the below options (default : zara1)
    * ETH datasets: `eth`, `hotel`
    * UCY datasets: `zara1`, `zara2`, `univ`
    * SNU-ASRI datasets: `snu-asri`, `snu-asri-ood`
* --predictor : Select the predictor (default : traj)
    * **Linear predictor** `linear`
    * **[EigenTrajectory][eigentraj-link]** `eigen`
    * **[Trajectron++][trajectronpp-link]** `traj`
    * **[Social-STGCNN][socialstgcnn-link]** `socialstgcnn`
    * **[Social-VAE][socialvae-link]** `socialvae`
    * **[KoopCast][koopcast-link]** `koopcast` 
* --controller : Select the controller for control test (default : grid)
    * **Grid MPC** `grid`
    * **Sampling-base MPC** `sampling`
    * **Conformal Prediction MPC** `conformal`
    * **Egocentric Conformal Prediction MPC** `ecp_mpc`
    * **MPPI** `mppi`
* --prediction_len : Length of the predicted trajectory (default : 12, unit : frame)
* --history_len : Length of the ground truth trajectory (default : 8)
* --t_begin : Start time of the dataset (default : 40)
* --t_end : End time of the data (default: 2000)
* --save_video : Save the result to video (default : True)




## To-do-list
- [x] MPPI implementation
- [ ] ECP-MPC migration
- [x] visualization: direction-indicators, complete pedestrian histories, robot figure, linewidth
- [x] benchmark tests: training in `SNU-ASRI`
- [ ] OOD definition & evaluation
- [ ] intuitive scenarios
- [x] exclude `.npy` files from the repository; they are too large to keep inside the repository; need to be downloaded from an external source
- [ ] exclude all pytorch model weights
- [ ] path to frame directory as an environment variable instead of manually modifying the associated variables inside the codes
- [ ] manage configurations as config files
- [ ] add a visualizer class to control all visualization
- [ ] homography matrix for `snu-asri` (unified visualization code)


[trajectronpp-link]: https://github.com/StanfordASL/Trajectron-plus-plus
[eigentraj-link]: https://github.com/InhwanBae/EigenTrajectory
[socialstgcnn-link]: https://github.com/abduallahmohamed/Social-STGCNN
[socialvae-link]: https://github.com/xupei0610/SocialVAE
[koopcast-link]: https://github.com/Koopcast/Koopcast
[pytorch-mppi-link]: https://github.com/UM-ARM-Lab/pytorch_mppi

# Update Log
[2025-10-02 19:45] move src/canvas to canvas; move all simulation*.py files into examples

[2025-10-02 20:48] Dataset class added; defined in `canvas.datasets.dataset`

[2025-10-10 11:00] Changed tool's main code that accompany with "example" python code - 'main_program.py' (split existing code to 'main_program.py' & 'simulation.py')

[2025-10-16] MPPI integrated into our codebase

# Troubleshooting
Run the following if QT platform plugin is messed up:
```
export QT_QPA_PLATFORM=offscreen
```
