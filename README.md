# CANVAS: Competency-Aware Navigation Assessment for Pedestrian Trajectory Forecasting
Below is a sample run code for the bare minimum imports required to call the predictors and competency index implemented into our code.

```python
from src.canvas.datasets.dataset_loader import get_dataset_spec, _load_background_image
from src.canvas import Environment, GridMPC, AdaptiveConformalPredictionModule, Predictors, CompetencyIndex, Predictor_CI
from save_ci import save_ci_traj_positions_csv, save_ci_ctrl_local_csv, project_ctrl_step_to_local_xy, save_ci_iteration_csv, save_frame_png

# setup: dataset, predictor, simulation environment, controller, competency index
obj_predictor = Predictors(chosen_predictor=predictor,prediction_len=prediction_len,history_len=history_len,dt=dt,dataset=dataset,device='cpu')         
ci_traj     = CompetencyIndex(case="traj",      r_star=rstar, return_type="series")
t_begin=40 # time step to begin environment in dataset
t_end= 300 # time step to end environment in dataset
environment = Environment(
            filepath=npy_path,
            dt=dt,
            init_robot_pose=init_robot_pose,
            t_begin=t_begin,
            t_end=t_end
        )
cost_function = L2Euclidean(goal=env.goal, terminal_weight=10.)
controller = MPPI(kinematic_model='differential_drive', cost_function=cost_function)
index = CostCompetencyIndex(conformal_predictor=conformal_predictor, controller=controller)

# simulation loop
obs = env.reset()

for t in range(200):
    prediction_res = predictor(obs)
    action, controller_info = controller(obs, prediction_res)
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
