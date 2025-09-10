# CANVAS: Competency-Aware Navigation Assessment for Pedestrian Trajectory Forecasting


```python
from canvas.datasets import load_dataset, Dataset
from canvas.predictors import TrajectronPlusPlus
from canvas.controllers import MPPI
from canvas.envs import SimulationEnv
from canvas.conformal_predictors import ACI
from canvas.conformal_predictors.score_functions import CostDiscrepancy
from canvas.competency_indices import CostCompetencyIndex
from canvas.controllers.cost_functions import L2Euclidean

# setup: dataset, predictor, simulation environment, controller, competency index
dataset: Dataset = load_dataset(id='ETH')
predictor = TrajectronPlusPlus()
predictor.load('eth_pretrained.pth')
conformal_predictor = ACI(score_function=ActionDiscrepancy, stepsize=1e-3, target_miscoverage=0.1, threshold=1.)

env = SimulationEnv(dataset)
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
