import torch
from .mpc import BaseMPC
from .mppi import KernelMPPI
from .conformal_controller import ConformalController
from .grid_solver import GridMPC
from .sampling_based_mpc import SamplingBasedMPC
from .ecp_mpc import EgocentricCPMPC
import numpy as np
class controllers:
    def __init__(self, chosen_controller='mppi', prediction_len=12, dt=0.1, goal=None, d_min=None, mppi_params=None):
        """Simple access class for different controllers.

        Args:
            chosen_controller: One of {"mpc", "mppi"}.
            prediction_len: Number of future steps to predict.
            dt: Timestep used by some controllers.
            goal: goal position
            d_min: safety region (margin)
            mppi_params: for mppi
        """
        self._dt = dt
        name = str(chosen_controller).strip().lower()
        ROBOT_RAD = .4
        d_min = ROBOT_RAD + .1 / np.sqrt(2.)

        if name in ("mpc", "basempc"):
            # Base MPC            
            self.ControllerModel = BaseMPC(prediction_horizon=prediction_len, dt=dt, goal=goal, d_min=d_min)
        elif name == "mppi":
            # MPPI
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mppi_params = {
                "num_samples": 500,
                "noise_mu": torch.zeros(2, dtype=torch.float, device=device),
                "noise_sigma": torch.diag(torch.tensor([1., 1.], dtype=torch.float, device=device)),
                "u_max": torch.tensor([.8, .7], dtype=torch.float, device=device),
                "lambda_": 1,
                "device": device
            }
            self.ControllerModel = KernelMPPI(prediction_horizon=prediction_len, dt=dt, goal=goal, d_min=d_min, mppi_params=mppi_params)
        else:
            raise ValueError(
                f"Unknown controller '{chosen_controller}'. "
                "Available options are 'mpc' and 'mppi'."
            )
        '''
        if name in ("conformal", "conf"):
            # Conformal Controller
            self.ControllerModel = ConformalController(n_steps=prediction_len, dt=dt)
        elif name in ("grid", "gridmpc"):
            # Grid-based MPC
            self.ControllerModel = GridMPC(n_steps=prediction_len, dt=dt)
        elif name in ("sampling", "samplingmpc"):
            # Sampling-based MPC
            self.ControllerModel = SamplingBasedMPC(n_steps=prediction_len, dt=dt)
        elif name in ("ecp", "ecpmpc"):
            # Egocentric Convex Polytope MPC
            self.ControllerModel = EgocentricCPMPC(n_steps=prediction_len, dt=dt)
        '''
        
    def __call__(self, obs, prediction_res, change_controller_sate=False):
        return self.ControllerModel(obs=obs, prediction_res=prediction_res, change_controller_state=change_controller_sate)
    def controller(self):
        return self.ControllerModel