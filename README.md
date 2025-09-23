# CANVAS: Competency-Aware Navigation Assessment for Pedestrian Trajectory Forecasting
Below is a sample run code for the bare minimum imports required to call the predictors and competency index implemented into our code.

```python
from src.canvas.datasets.dataset_loader import get_dataset_spec, _load_background_image
from src.canvas import Environment, GridMPC, AdaptiveConformalPredictionModule, Predictors, CompetencyIndex, Predictor_CI
from save_ci import save_ci_traj_positions_csv, save_ci_ctrl_local_csv, project_ctrl_step_to_local_xy, save_ci_iteration_csv, save_frame_png

# setup: dataset, predictor, simulation environment, controller, competency index
prediction_len = 12
history_len = 8
dt = 0.10

obj_predictor = Predictors(chosen_predictor="linear",prediction_len=prediction_len,history_len=history_len,dt=dt,dataset="ETH",device='cpu')         
ci_traj     = CompetencyIndex(case="traj",r_star=0.5, return_type="series")
t_begin=40 # time step to begin environment in dataset
t_end= 300 # time step to end environment in dataset
dt= 0.4
init_robot_pose=np.array([0, 0, np.pi / 2.])
max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1)
offline_calibration_set = {i: [] for i in range(prediction_len)}
env = Environment(
            filepath=npy_path,
            dt=dt,
            init_robot_pose=init_robot_pose,
            t_begin=t_begin,
            t_end=t_end
        )
cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                      step_size=0.05,
                                                      n_scores=prediction_len,
                                                      max_interval_lengths=max_interval_lengths,
                                                      sample_size=20,
                                                      offline_calibration_set=offline_calibration_set)
controller = GridMPC(n_steps=prediction_len, dt=dt)

# simulation loop
obs = env.reset()
# make module for filtering valid history and future predictions.
dynamic obs= # module to be implemented.
for t in range(200):
            #make all-in-one module for the processes below?
            prediction_res = predictor(obs)
            confidence_intervals = cp_module.update(dynamic_obs, prediction_res if isinstance(prediction_res, dict) else {})
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
            #update the below section a bit more
            index.update({'obs': obs, 'pred': prediction_res, 'action': action}, step=t)
            idx = index.compute_index()
            obs = env.sim(action)

# logging
index.save_res('competency_example.npy')
```

---

## Running the simulation

You can run the simulation with "simulation.py" with some variables

* --goal_x, --goal_y : Goal position for control test (default : (8.0, 0.2))
* --num_iter : Number of iterations for simulation (default : 1)
* --r_star : Threshold value $R^*$ of computating the Competency Index (CI) (default : 0.5)
* --dataset : Select the dataset (default : Lobby)
    * ETH
    * Hotel
    * Univ
    * Zara01
    * Zara02
    * Lobby
* --predictor : Select the predictor (default : linear)
    * linear (Linear predictor)
    * gp (GP predictor)
    * eigen (EigenTrajectory)
    * traj (Trajectron++)
    * koopcast (KoopCast)
* --save_video : Save the result to video (default : False)
* --video_fps : Configure FPS for saving video. Maybe better to fit with 'dt' (default : 10.0)

