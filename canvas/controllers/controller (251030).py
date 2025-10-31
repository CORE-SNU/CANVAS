import torch
from .conformal_controller import ConformalController
from .grid_solver import GridMPC
from .sampling_based_mpc import SamplingBasedMPC
from .ecp_mpc import EgocentricCPMPC
from .mppi import KernelMPPI
from .mpc import BaseMPC
import numpy as np
class controllers:
    def __init__(self, chosen_controller='conformal', prediction_len=12, dt=0.1):
        """Simple access class for different controllers.

        Args:
            chosen_controller: One of {"conformal", "mpc", "grid", "sampling", "ecp_mpc", "mppi"}.
            prediction_len: Number of future steps to predict.
            dt: Timestep used by some controllers.
        """
        self._dt = dt
        name = str(chosen_controller).strip().lower()

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
        elif name in ("mpc", "basempc"):
            # Base MPC
            ROBOT_RAD = .4
            d_min = ROBOT_RAD + .1 / np.sqrt(2.)
            self.ControllerModel = BaseMPC(prediction_horizon=prediction_len, dt=dt, d_min=d_min)
        elif name == "mppi":
            # MPPI
            ROBOT_RAD = .4
            d_min = ROBOT_RAD + .1 / np.sqrt(2.)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mppi_params = {
                "num_samples": 500,
                "noise_mu": torch.zeros(2, dtype=torch.float, device=device),
                "noise_sigma": torch.diag(torch.tensor([1., 1.], dtype=torch.float, device=device)),
                "u_max": torch.tensor([.8, .7], dtype=torch.float, device=device),
                "lambda_": 1,
                "device": device
            }
            self.ControllerModel = KernelMPPI(prediction_horizon=prediction_len, dt=dt, mppi_params=mppi_params, d_min=d_min)
        else:
            raise ValueError(
                f"Unknown controller '{chosen_controller}'. "
                "Available options are 'conformal', 'mpc', 'grid', 'sampling', 'ecp', and 'mppi'."
            )
    def __call__(self, position_x, position_y, orientation_z, linear_x, angular_z, boxes, predictions, confidence_intervals, goal,history=None, **__):
        return self.ControllerModel(position_x=position_x, position_y=position_y, orientation_z=orientation_z, 
                                    linear_x=linear_x, angular_z=angular_z, boxes=boxes, 
                                    predictions=predictions, confidence_intervals=confidence_intervals, goal=goal, history=history)
    def controller(self):
        return self.ControllerModel