#finish creating code for controller.py module
from .conformal_controller import ConformalController
from .grid_solver import GridMPC
from .sampling_based_mpc import SamplingBasedMPC
from .ecp_mpc import EgocentricCPMPC
import numpy as np
class controllers:
    def __init__(self, chosen_controller='conformal', prediction_len=12, dt=0.1):
        """Simple access class for different controllers.

        Args:
            chosen_controller: One of {"conformal", "grid", "sampling","ecp_mpc"}.
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

        else:
            raise ValueError(
                f"Unknown controller '{chosen_controller}'. "
                "Available options are 'conformal', 'grid', 'sampling', and 'ecp'."
            )
    def __call__(self, position_x, position_y, orientation_z, cmd_linear_x, cmd_angular_z, boxes, predictions, confidence_intervals, goal,history=None, **__):
        return self.ControllerModel(position_x=position_x, position_y=position_y, orientation_z=orientation_z, 
                                    linear_x=cmd_linear_x, angular_z=cmd_angular_z, boxes=boxes, 
                                    predictions=predictions, confidence_intervals=confidence_intervals, goal=goal,history=history)
    def controller(self):
        return self.ControllerModel