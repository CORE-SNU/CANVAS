# gaussian_process_predictor.py

from typing import Dict, Any
import numpy as np
import george
from george import kernels
from scipy.optimize import minimize

from .wrapper_predictor import BasePredictors


class GaussianProcessPredictor(BasePredictors):
    def __init__(
        self,
        prediction_len: int = 12,
        history_len: int = 8,
        dt: float = 0.1,
        device: str = "cpu",
    ):
        # Initialize common fields (history_len, prediction_len, dt, device)
        super().__init__(
            prediction_len=prediction_len,
            history_len=history_len,
            dt=dt,
            device=device,
        )
        # Keep original attribute names so the call body remains unchanged
        assert self.prediction_len > 0
        self._prediction_len = self.prediction_len
        self._history_len = self.history_len
        self._dt = self._dt  # already set by BasePredictors

    def __call__(self, tracking_result: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        H = self._history_len
        P = self._prediction_len
        dt = self._dt  # (currently not used; see comment in rollout)

        pred: Dict[Any, np.ndarray] = {}
        for object_id, traj in tracking_result.items():
            traj = np.asarray(traj)
            if traj.shape[0] >= H + 1:
                # --- build training data ---
                disp = traj[1:] - traj[:-1]           # per-step displacement
                Yx = disp[-H:, 0]                     # targets (x)
                Yy = disp[-H:, 1]                     # targets (y)
                X = traj[-H:]                         # inputs: positions (ndim=2)

                # --- kernels & GPs ---
                kx = kernels.ExpSquaredKernel(metric=1.0, ndim=2)
                ky = kernels.ExpSquaredKernel(metric=1.0, ndim=2)
                gp_x = george.GP(kx, solver=george.HODLRSolver)
                gp_y = george.GP(ky, solver=george.HODLRSolver)

                # use small observation noise (do NOT pass Yx/Yy here)
                yerr = 1e-3 * np.ones(H)
                gp_x.compute(X, yerr)
                gp_y.compute(X, yerr)

                # --- hyperparameter opt (optional but kept) ---
                self._optimize_hyperparameters(gp_x, gp_y, Yx, Yy)
                # recompute with the same X, yerr after params changed
                gp_x.compute(X, yerr)
                gp_y.compute(X, yerr)

                # --- rollout ---
                ps = []
                state = traj[-1].copy()
                for _ in range(P):
                    mu_x, _ = gp_x.predict(Yx, t=state[None, :], return_cov=False)
                    mu_y, _ = gp_y.predict(Yy, t=state[None, :], return_cov=False)

                    # single-integrator: treat targets as displacement per step
                    # if you prefer velocity, set disp=(traj[1:]-traj[:-1])/dt and add (mu*dt)
                    state = state + np.array([mu_x.item(), mu_y.item()])
                    ps.append(state.copy())
                pred[object_id] = np.stack(ps, axis=0).astype(np.float32)
            else:
                pred[object_id] = np.repeat(traj[-1:, :], P, axis=0).astype(np.float32)
        return pred

    @staticmethod
    def _optimize_hyperparameters(gp_x, gp_y, Yx, Yy):
        def nll_x(p):
            gp_x.set_parameter_vector(p)
            return -gp_x.log_likelihood(Yx)
        def grad_nll_x(p):
            gp_x.set_parameter_vector(p)
            return -gp_x.grad_log_likelihood(Yx)
        res_x = minimize(nll_x, gp_x.get_parameter_vector(), jac=grad_nll_x, method="L-BFGS-B")
        gp_x.set_parameter_vector(res_x.x)

        def nll_y(p):
            gp_y.set_parameter_vector(p)
            return -gp_y.log_likelihood(Yy)
        def grad_nll_y(p):
            gp_y.set_parameter_vector(p)
            return -gp_y.grad_log_likelihood(Yy)
        res_y = minimize(nll_y, gp_y.get_parameter_vector(), jac=grad_nll_y, method="L-BFGS-B")
        gp_y.set_parameter_vector(res_y.x)
