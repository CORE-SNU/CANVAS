import os, sys, time
import importlib
import torch
import numpy as np
import sys
_DATA_DIR_VAE = os.path.dirname(__file__)
from .social_vae import SocialVAE
from .utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state
from ..wrapper_predictor import BasePredictors
class Social_VAE_Predictor(BasePredictors):
    def __init__(
    self,
    prediction_len: int = 12,
    history_len: int = 8,
    dt: float = 0.4,
    device: str = "cpu",
    *,
    cfg: str = "/config/eth.py",
    model_path: str = "/models/eth",
    seed_: int = 1,
    ):
        super().__init__(
        prediction_len=prediction_len,
        history_len=history_len,
        dt=dt,
        device=device,
        )
        spec = importlib.util.spec_from_file_location("config", _DATA_DIR_VAE+cfg)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        #seems neccesary to run program
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dt=dt
        seed(seed_)
        self.rng_state = get_rng_state(self.device)
        self.model = SocialVAE(horizon=prediction_len, ob_radius=cfg.OB_RADIUS, hidden_dim=cfg.RNN_HIDDEN_DIM)
        self.model.to(self.device)
        model_path=_DATA_DIR_VAE+model_path
        if model_path:
            ckpt = os.path.join(model_path, "ckpt-best")
            if os.path.exists(ckpt):
                state_dict = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state_dict["model"])
            else:
                print("model_path not found: {}".format(ckpt))
                raise FileNotFoundError
        
    def xy_to_state6(self,xy, dt=0.4, frameskip=1):
        xy = np.asarray(xy, dtype=np.float32)
        step = float(frameskip) * float(dt)
        vxvy = np.gradient(xy, step, axis=0, edge_order=1).astype(np.float32)
        axay = np.gradient(vxvy, step, axis=0, edge_order=1).astype(np.float32)
        return np.concatenate([xy, vxvy, axay], axis=1).astype(np.float32)  # (L,6)

    def build_x(self,tracking_results, device="cpu", dt=0.4, frameskip=1):
        # prediction: {pid: (L,2)}
        pids = list(tracking_results.keys())
        states = [self.xy_to_state6(tracking_results[pid], dt, frameskip) for pid in pids]  # [(L,6)]*N
        x = np.stack(states, axis=1).astype(np.float32)  # (L, N, 6)  N = len(pids)
        return torch.from_numpy(x).to(device), pids

    def __call__(self, tracking_results, frameskip=1):
        if not tracking_results:
            return {}
        self.model.eval()
        set_rng_state(self.rng_state, self.device)
        with torch.no_grad():
            x, pids = self.build_x(tracking_results, device=self.device, dt=self.dt, frameskip=frameskip)
            L, N, D = x.shape
            out_dict = {}
            for i, pid in enumerate(pids):
                # (1) target agent stream: keep batch N=1
                x_main = x[:, i:i+1, :] 
                if N > 1:
                    neigh = torch.cat([x[:, :i, :], x[:, i+1:, :]], dim=1)   # [L, N-1, 6]
                    neigh = neigh.unsqueeze(1)                               # [L, 1, N-1, 6]
                else:
                    neigh = x.new_empty((L, 1, 0, D))                        # [L, 1, 0, 6] no neighbors
                y = self.model(x, None, n_predictions=0)  # expected (H, N, 2)
                if y.dim() == 2 and y.shape[1] == 2:
                    traj = y  # (H,2)
                elif y.dim() == 3 and y.shape[2] == 2:
                    # (H,S,2) where S can be 1 or 1+Nn. By construction, ego is first.
                    ego_idx = 0

                    # Safety: if ordering ever changes, re-identify ego by proximity to last observed xy.
                    try:
                        last_xy = x_main[-1, 0, :2]                      # (2,)
                        # distance between y[0, j] and last_xy
                        dists = ((y[0, :, :] - last_xy).pow(2).sum(-1))  # (S,)
                        ego_idx = int(dists.argmin().item())
                    except Exception:
                        pass  # keep default 0

                    traj = y[:, ego_idx, :]  # (H,2)
                else:
                    raise RuntimeError(f"Unexpected output shape {tuple(y.shape)}; wanted (H,2) or (H,S,2).")

                out_dict[pid] = traj.detach().cpu().numpy()
            return out_dict
