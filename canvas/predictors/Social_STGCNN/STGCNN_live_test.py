import math
import numpy as np
import torch
import networkx as nx
from .model import social_stgcnn
from ..wrapper_predictor import BasePredictors
import sys
import os
import pickle
_DATA_DIR_STGCNN = os.path.dirname(__file__)
# --- helpers ---------------------------------------------------------------

def _anorm(p1, p2):
    dx, dy = (p1[0] - p2[0]), (p1[1] - p2[1])
    nrm = math.sqrt(dx * dx + dy * dy)
    return 0.0 if nrm == 0.0 else 1.0 / nrm

@torch.no_grad()
def _dict8_to_graph(
    obs_dict, obs_len=8, norm_lap_matr=True,
    device="cuda", dtype=torch.float32
):
    """
    Convert {pid: (T=8,2)} -> (V_obs[T,N,2], A_obs[T,N,N], pid_order, abs_seq[N,2,T])
    - V_obs uses relative displacements (∆x, ∆y), like the original repo
    - A_obs uses inverse L2 between per-step ∆’s (normalized Laplacian if requested)
    """
    items = []
    for pid, traj in obs_dict.items():
        arr = np.asarray(traj, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < obs_len:
            continue
        arr = arr[-obs_len:, :2]
        if not np.isfinite(arr).all():
            continue
        items.append((pid, arr))
    if not items:
        raise ValueError("No valid agents with >= obs_len finite (x,y) history.")

    # fixed node order for determinism
    items.sort(key=lambda kv: str(kv[0]))
    pid_order = [pid for pid, _ in items]
    N, T = len(items), obs_len

    # absolute seq (N,2,T) and relative (N,2,T)
    seq_abs = np.stack([np.stack([a[:,0], a[:,1]], axis=0) for _, a in items], axis=0)  # (N,2,T)
    seq_rel = np.zeros_like(seq_abs)
    seq_rel[:, :, 1:] = seq_abs[:, :, 1:] - seq_abs[:, :, :-1]

    # build V (T,N,2) and A (T,N,N)
    V = np.zeros((T, N, 2), dtype=np.float64)
    A = np.zeros((T, N, N), dtype=np.float64)
    for t in range(T):
        step_rel = seq_rel[:, :, t]  # (N,2)
        V[t] = step_rel
        for i in range(N):
            A[t, i, i] = 1.0
            for j in range(i + 1, N):
                w = _anorm(step_rel[i], step_rel[j])
                A[t, i, j] = w
                A[t, j, i] = w
        if norm_lap_matr:
            G = nx.from_numpy_array(A[t])
            A[t] = nx.normalized_laplacian_matrix(G).toarray()

    V_obs  = torch.tensor(V, dtype=dtype, device=device)        # (T,N,2)
    A_obs  = torch.tensor(A, dtype=dtype, device=device)        # (T,N,N)
    absseq = torch.tensor(seq_abs, dtype=dtype, device=device)  # (N,2,T)
    return V_obs, A_obs, pid_order, absseq

def _rel_to_abs(rel_seq, last_abs):
    """
    rel_seq: (T_pred, N, 2)  | last_abs: (N, 2)
    returns: (T_pred, N, 2) absolute by cumulative sum starting from last_abs
    """
    out = torch.cumsum(rel_seq, dim=0) + last_abs.unsqueeze(0)
    return out

# --- main deterministic predictor ------------------------------------------
class STGCNN_Predictor(BasePredictors):
    def __init__(
        self,
        prediction_len: int = 12,
        history_len: int = 8,
        dt: float = 0.1,
        device: str = "cpu",
        *,
        model_dir: str = "/checkpoint/social-stgcnn-eth",
    ):
        # Common interface fields: prediction_len, history_len, dt, device
        super().__init__(
            prediction_len=prediction_len,
            history_len=history_len,
            dt=dt,
            device=device,
        )
        model_path = _DATA_DIR_STGCNN+model_dir+'/val_best.pth'
        args_path = _DATA_DIR_STGCNN+model_dir+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)
        self.model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)

        self.obs_len=history_len
        self.pred_len=prediction_len
        self.norm_lap_matr=True
        self.device=device
        self.dtype=torch.float32
        self.model_dir = model_dir
    @torch.no_grad()
    def __call__(
        self,
        tracking_results,
    ):
        """
        Deterministic Social-STGCNN rollout:
        {pid: (8,2)} -> {pid: (12,2)} using the predicted means (μx, μy).
        """
        if not tracking_results:
            return {}
        self.model.eval()
        V_obs, A_obs, pid_order, absseq = _dict8_to_graph(
            tracking_results, obs_len=self.obs_len, norm_lap_matr=self.norm_lap_matr, device=self.device, dtype=self.dtype
        )
        # model expects (B, C=2, T_obs, N); A passed without batch (as in repo test)
        B = 1
        T_obs, N, _ = V_obs.shape
        assert T_obs == self.obs_len, "history length mismatch"

        V_in = V_obs.unsqueeze(0).permute(0, 3, 1, 2)  # (1,2,T_obs,N)
        A_in = A_obs  # (T_obs,N,N)

        # forward
        V_pred, _ = self.model(V_in, A_in)                  # usually (1, F, T_pred, N)
        V_pred = V_pred.permute(0, 2, 3, 1).contiguous()  # (1,T_pred,N,F)

        # take deterministic means (μx, μy), not samples
        mu_rel = V_pred[..., 0:2]                      # (1,T_pred,N,2)
        mu_rel = mu_rel[0]                             # (T_pred,N,2)

        # convert to absolute using last observed absolute position
        last_abs = absseq[:, :, -1]                    # (N,2)
        pred_abs = _rel_to_abs(mu_rel, last_abs)       # (T_pred,N,2)

        # map back to {pid: (T_pred,2)} with the fixed node order
        out = {pid_order[i]: pred_abs[:, i, :].detach().cpu().numpy() for i in range(N)}
        return out

# --- example ---------------------------------------------------------------

# live_obs = {42: np.random.rand(8,2), 7: np.random.rand(8,2)}
# model = ...  # your loaded social_stgcnn(...).to(device)
# preds = predict_live_dict_det(model, live_obs, obs_len=8, pred_len=12, device="cuda")
# print({k: v.shape for k,v in preds.items()})  # {pid: (12,2)}
