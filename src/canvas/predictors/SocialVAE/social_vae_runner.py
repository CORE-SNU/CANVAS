import os, sys, time
import importlib
import torch
import numpy as np

from social_vae import SocialVAE
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state
from ..wrapper_predictor import BasePredictors
class Social_VAE_Predictor(BasePredictors):
    def __init__(
    self,
    prediction_len: int = 12,
    history_len: int = 8,
    dt: float = 0.1,
    device: str = "cpu",
    *,
    cfg: str = "./",
    model_path: str = None,
    seed_: int = 1,
    ):
        super().__init__(
        prediction_len=prediction_len,
        history_len=history_len,
        dt=dt,
        device=device,
        )
        spec = importlib.util.spec_from_file_location("config", cfg)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        #seems neccesary to run program
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        seed(seed_)
        self.rng_state = get_rng_state(self.device)
        self.model = SocialVAE(horizon=prediction_len, ob_radius=cfg.OB_RADIUS, hidden_dim=cfg.RNN_HIDDEN_DIM)
        self.model.to(self.device)

        if model_path:
            ckpt = os.path.join(model_path, "ckpt-best")
            if os.path.exists(ckpt):
                state_dict = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state_dict["model"])
            else:
                print("model_path not found: {}".format(ckpt))
                raise FileNotFoundError
        
    def xy_to_state6(xy, dt=0.4, frameskip=1):
        xy = np.asarray(xy, dtype=np.float32)
        step = float(frameskip) * float(dt)
        vxvy = np.gradient(xy, step, axis=0, edge_order=1).astype(np.float32)
        axay = np.gradient(vxvy, step, axis=0, edge_order=1).astype(np.float32)
        return np.concatenate([xy, vxvy, axay], axis=1).astype(np.float32)  # (L,6)

    def build_x(prediction, device="cpu", dt=0.4, frameskip=1):
        # prediction: {pid: (L,2)}
        pids = list(prediction.keys())
        states = [xy_to_state6(prediction[pid], dt, frameskip) for pid in pids]  # [(L,6)]*N
        x = np.stack(states, axis=1).astype(np.float32)  # (L, N, 6)  N = len(pids)
        return torch.from_numpy(x).to(device), pids

    def test(self, prediction, dt=0.4, frameskip=1, return_numpy=True):
        self.model.eval()
        set_rng_state(self.rng_state, self.device)
        with torch.no_grad():
            x, pids = build_x(prediction, device=self.device, dt=dt, frameskip=frameskip)
            out = self.model(x, None, n_predictions=0)  # expected (H, N, 2)
            arr = out.detach().cpu().numpy() 
            return {pid: arr[:, i, :] for i, pid in enumerate(pids)}
