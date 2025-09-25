from src.canvas.datasets.dataset_loader import get_dataset_spec, _load_background_image
from src.canvas import Environment, GridMPC, AdaptiveConformalPredictionModule,\
 Predictors, CompetencyIndex, Predictor_CI,dynamic_observation_filter ,region_to_box
import numpy as np

# setup: dataset, predictor, simulation environment, controller, competency index
prediction_len = 12
history_len = 8
dt = 0.10
obj_predictor = Predictors(chosen_predictor="linear",prediction_len=prediction_len,history_len=history_len,dt=dt,dataset="ETH",device='cpu')         
ci_traj     = CompetencyIndex(case="traj",r_star=0.5, return_type="series")
t_begin=40 # time step to begin environment in dataset
t_end= 300 # time step to end environment in dataset
dt= 0.4
goal=np.array([8.0, .2])
init_robot_pose=np.array([0, 0, np.pi / 2.])
max_interval_lengths = 0.3 * dt * np.arange(1, prediction_len + 1)
offline_calibration_set = {i: [] for i in range(prediction_len)}
dataset="zara1"
env = Environment(
            filepath=dataset,
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
persistent_static_boxes = [region_to_box(r) for r in get_dataset_spec(dataset).static_regions]
# simulation loop
position_x, position_y, orientation_z = env.reset()
done = False
while not done:
    obs = env._get_obs()
    linear_x, angular_z = env.get_velocity()
    dynamic_obs= dynamic_observation_filter(obs, position_x, position_y, prediction_len)
    #make all-in-one module for the processes below?
    prediction_res = obj_predictor(dynamic_obs)
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
    #index.update({'obs': obs, 'pred': prediction_res, 'action': action}, step=t)
    #idx = index.compute_index()
    if velocity is not None and len(velocity) > 0:
        cmd_linear_x, cmd_angular_z = velocity[0]
    else:
        cmd_linear_x, cmd_angular_z = 0.0, 0.0
    robot_pose, done = env.step([cmd_linear_x, cmd_angular_z])
    position_x, position_y, orientation_z = robot_pose
#index.save_res('competency_example.npy')