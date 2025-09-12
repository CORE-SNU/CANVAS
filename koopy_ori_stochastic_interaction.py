#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDN + Koopman (stochastic one-shot target) trajectory predictor — Multi-Agent MDN (instant neighbors only)

Pipeline (single file):
  1) Train an MDN p(g_t | h_t^{(i)}, m_t^{(i)}) for the P-step-ahead target g_t^{(i)} = x_{t+P}^{(i)}
     where:
       - h_t^{(i)} = (x_{t-H+1}^{(i)}, ..., x_t^{(i)})  (agent i history)
       - m_t^{(i)} = Positions of neighbors at time s within radius R (relative to agent i).
  2) Estimate a global Koopman-like linear operator K on lifted state
     psi_t = [x_t, x_{t-1}, ..., x_{t-H+1}, g_t, 1]
     so that psi_{t+1} ≈ K psi_t with g_{t+1} = x_{t+1+P} (unchanged).
  3) Evaluate on a test file: use MDN mean and best-of-S MDN samples for g_t,
     roll forward P steps via K, and report ADE / FDE.

Author: ChatGPT (Jungjin's assistant)
"""

import os
import math
import glob
import argparse
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------- Hyperparameters & Defaults ----------------------
HISTORY_LENGTH_DEFAULT = 8
PRED_LENGTH_DEFAULT    = 12
MDN_COMPONENTS_DEFAULT = 6   # mixture components
MDN_HIDDEN_DEFAULT     = 128
MDN_EPOCHS_DEFAULT     = 20
MDN_BATCH_DEFAULT      = 512
MDN_LR_DEFAULT         = 1e-3
SIGMA_MIN_DEFAULT      = 0.05  # meters
RIDGE_LAMBDA_DEFAULT   = 1e-3  # Koopman ridge
SAMPLES_DEFAULT        = 20     # best-of-S samples for metrics

# Multi-agent context defaults
MAX_NEIGHBORS_DEFAULT  = 6      # M (0 => disable)
NEIGHBOR_RADIUS_DEFAULT = 5.0   # meters
NEIGHBOR_RELATIVE_DEFAULT = True  # frame-wise relative coordinates

# ---------------------- Utility ----------------------

def valid_windows(traj: np.ndarray, H: int, P: int):
    T = traj.shape[0]
    mask = np.all(np.isfinite(traj), axis=1)
    for s in range(H-1, T - P):
        if np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+1]) and mask[s+P]:
            yield s

def build_multiagent_context_instant(
    data: np.ndarray,
    s: int,
    i: int,
    neighbor_radius: float,
    max_neighbors: int,
    relative: bool = True
) -> np.ndarray:
    """
    Use ONLY neighbor positions at time s.
    Returns vector shape (2*max_neighbors,)
    """
    if max_neighbors <= 0:
        return np.zeros((0,), dtype=np.float32)

    T, N, _ = data.shape
    pos_i_s = data[s, i, :]  # (2,)

    candidates = []
    for j in range(N):
        if j == i:
            continue
        pos_j_s = data[s, j, :]
        if not np.all(np.isfinite(pos_j_s)):
            continue
        if np.linalg.norm(pos_j_s - pos_i_s) <= neighbor_radius:
            candidates.append((j, pos_j_s))

    candidates.sort(key=lambda item: np.linalg.norm(item[1] - pos_i_s))
    neighbors = candidates[:max_neighbors]

    pieces = []
    for _, pos_j_s in neighbors:
        if relative:
            vec = pos_j_s - pos_i_s
        else:
            vec = pos_j_s
        pieces.append(vec.astype(np.float32))

    while len(pieces) < max_neighbors:
        pieces.append(np.zeros((2,), dtype=np.float32))

    return np.concatenate(pieces, axis=0)  # (2*max_neighbors,)

def collect_mdn_dataset_multiagent(
    train_files: List[str], H: int, P: int,
    max_neighbors: int, neighbor_radius: float, neighbor_relative: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (z, g) for MDN training:
      z = [h_t^{(i)}, m_t^{(i)}], h_t^{(i)} ∈ R^{2H}, m_t^{(i)} ∈ R^{2*max_neighbors}
      g = x_{t+P}^{(i)} ∈ R^2
    """
    H2 = 2 * H
    Mdim = 2 * max_neighbors
    X_list, Y_list = [], []

    for fpath in train_files:
        data = np.load(fpath)  # (T, N, 2)
        T, N, _ = data.shape
        for i in range(N):
            traj_i = data[:, i, :]
            for s in valid_windows(traj_i, H, P):
                hist_i = traj_i[s-H+1:s+1]                 # (H,2)
                g_i    = traj_i[s+P]                       # (2,)
                hvec   = hist_i[::-1].reshape(H2)          # latest-first
                mvec   = build_multiagent_context_instant(
                    data, s, i,
                    neighbor_radius=neighbor_radius,
                    max_neighbors=max_neighbors,
                    relative=neighbor_relative
                ) if max_neighbors > 0 else np.zeros((0,), dtype=np.float32)
                z = np.concatenate([hvec.astype(np.float32), mvec.astype(np.float32)], axis=0)
                X_list.append(z)
                Y_list.append(g_i.astype(np.float32))

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y

def collect_koopman_pairs(train_files: List[str], H: int, P: int, use_bias: bool=True):
    H2 = 2 * H
    D  = H2 + 2 + (1 if use_bias else 0)
    P_cols, F_cols = [], []
    for fpath in train_files:
        data = np.load(fpath)  # (T, N, 2)
        T, N, _ = data.shape
        for ag in range(N):
            traj = data[:, ag, :]
            mask = np.all(np.isfinite(traj), axis=1)
            for s in range(H-1, T - P - 1):
                if not (np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+2])):
                    continue
                hist  = traj[s-H+1:s+1]
                histp = traj[s-H+2:s+2]
                gt_g    = traj[s+P]
                gt_gp   = traj[s+1+P]
                hvec  = hist[::-1].reshape(H2)
                hvecp = histp[::-1].reshape(H2)
                psi   = np.concatenate([hvec, gt_g], axis=0)
                psip  = np.concatenate([hvecp, gt_gp], axis=0)
                if use_bias:
                    psi  = np.concatenate([psi,  [1.0]], axis=0)
                    psip = np.concatenate([psip, [1.0]], axis=0)
                P_cols.append(psi)
                F_cols.append(psip)
    if len(P_cols) == 0:
        raise RuntimeError("No Koopman pairs collected. Check data and masks.")
    Pmat = np.stack(P_cols, axis=1)  # (D,M)
    Fmat = np.stack(F_cols, axis=1)  # (D,M)
    return Pmat.astype(np.float64), Fmat.astype(np.float64)

# ---------------------- MDN Model ----------------------

LOG2PI = math.log(2.0 * math.pi)

class MDN(nn.Module):
    def __init__(self, in_dim: int, n_components: int = MDN_COMPONENTS_DEFAULT,
                 hidden: int = MDN_HIDDEN_DEFAULT, sigma_min: float = SIGMA_MIN_DEFAULT):
        super().__init__()
        self.K = n_components
        self.sigma_min = sigma_min
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_pi   = nn.Linear(hidden, self.K)
        self.head_mu   = nn.Linear(hidden, self.K * 2)
        self.head_logS = nn.Linear(hidden, self.K * 2)

    def forward(self, x):
        h = self.net(x)
        pi_logits = self.head_pi(h)
        mu        = self.head_mu(h).view(-1, self.K, 2)
        log_sigma = self.head_logS(h).view(-1, self.K, 2)
        min_log = math.log(self.sigma_min)
        log_sigma = torch.clamp(log_sigma, min=min_log)
        log_pi = torch.log_softmax(pi_logits, dim=1)
        return log_pi, mu, log_sigma

    @torch.no_grad()
    def mixture_mean(self, x):
        log_pi, mu, _ = self.forward(x)
        pi = torch.exp(log_pi)
        mean = torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        return mean

    @torch.no_grad()
    def sample(self, x, n_samples: int = 1):
        B = x.shape[0]
        log_pi, mu, log_sigma = self.forward(x)
        pi = torch.exp(log_pi)
        comps = torch.multinomial(pi, num_samples=n_samples, replacement=True)
        mu_g = torch.gather(mu, 1, comps.unsqueeze(-1).expand(B, n_samples, 2))
        sigma = torch.exp(torch.gather(log_sigma, 1, comps.unsqueeze(-1).expand(B, n_samples, 2)))
        eps = torch.randn_like(mu_g)
        return mu_g + sigma * eps

def mdn_nll(log_pi, mu, log_sigma, y):
    B, K, _ = mu.shape
    y_exp = y.unsqueeze(1).expand(B, K, 2)
    inv_var = torch.exp(-2.0 * log_sigma)
    log_norm = -0.5 * (
        torch.sum((y_exp - mu)**2 * inv_var, dim=2) +
        2.0 * torch.sum(log_sigma, dim=2) +
        2.0 * LOG2PI
    )
    log_mix = torch.logsumexp(log_pi + log_norm, dim=1)
    return -torch.mean(log_mix)

# ---------------------- Koopman Estimation ----------------------

def estimate_K(Pmat: np.ndarray, Fmat: np.ndarray, ridge: float = RIDGE_LAMBDA_DEFAULT) -> np.ndarray:
    D, M = Pmat.shape
    A = Pmat @ Pmat.T
    if ridge > 0:
        A = A + ridge * np.eye(D, dtype=Pmat.dtype)
    B = Fmat @ Pmat.T
    K = np.linalg.solve(A, B.T).T
    return K

# ---------------------- Build psi and rollout ----------------------

def build_psi_from_hist_g(hist: np.ndarray, g: np.ndarray, use_bias: bool=True) -> np.ndarray:
    H = hist.shape[0]
    hvec = hist[::-1].reshape(2*H)
    psi = np.concatenate([hvec, g], axis=0)
    if use_bias:
        psi = np.concatenate([psi, [1.0]], axis=0)
    return psi

def rollout_with_K(hist: np.ndarray, g: np.ndarray, K: np.ndarray, P: int, use_bias: bool=True) -> np.ndarray:
    psi = build_psi_from_hist_g(hist, g, use_bias)
    preds = []
    for _ in range(P):
        psi = K @ psi
        preds.append(psi[:2].copy())
    return np.stack(preds, axis=0)

# ---------------------- Dataset iterators ----------------------

def iter_test_windows(data: np.ndarray, H: int, P: int):
    T, N, _ = data.shape
    for i in range(N):
        traj = data[:, i, :]
        mask = np.all(np.isfinite(traj), axis=1)
        for s in range(H-1, T - P):
            if not (np.all(mask[s-H+1:s+1]) and np.all(mask[s+1:s+P+1])):
                continue
            hist = traj[s-H+1:s+1]
            gt   = traj[s+1:s+P+1]
            yield i, s, hist, gt

# ---------------------- Training / Evaluation ----------------------

def train_mdn(X: np.ndarray, Y: np.ndarray, in_dim: int, K: int, hidden: int,
              epochs: int, batch_size: int, lr: float, device: str='cpu') -> MDN:
    model = MDN(in_dim, n_components=K, hidden=hidden).to(device)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            log_pi, mu, log_sigma = model(xb)
            loss = mdn_nll(log_pi, mu, log_sigma, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(ds)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[MDN] Epoch {ep}/{epochs}  NLL: {avg:.6f}")
    return model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import time

# ---------------------- Evaluate ----------------------
import numpy as np
import torch

# ---------- helpers that accept EITHER {id: [H,D]} or {id: {'past':[H,D], 'mask':[H]}} ----------
def _past_array(entry, H: int) -> np.ndarray:
    """Normalize an obs entry to [H, D] float32 (left-pad by repeating first row if needed)."""
    if isinstance(entry, dict) and "past" in entry:
        arr = np.asarray(entry["past"], dtype=np.float32)
    else:
        arr = np.asarray(entry, dtype=np.float32)  # your case: [H,D]
    if arr.shape[0] < H:
        pad = np.repeat(arr[:1], H - arr.shape[0], axis=0)
        arr = np.concatenate([pad, arr], axis=0)
    return arr[-H:, :]  # ensure exactly H

def _neighbor_ctx_all_from_obs(
    obs: dict,
    H: int,
    max_neighbors: int,
    neighbor_radius: float,
    relative: bool = True,
) -> tuple[list[int], np.ndarray]:
    """
    Returns:
      ids  : [M]
      mvec : [M, 2*max_neighbors] float32
    """
    ids = sorted(obs.keys())
    M = len(ids)
    if max_neighbors <= 0 or M == 0:
        return ids, np.zeros((M, 0), dtype=np.float32)

    pos = np.stack([_past_array(obs[k], H)[-1, :2] for k in ids], axis=0).astype(np.float32)  # [M,2]

    mvec = np.zeros((M, 2 * max_neighbors), dtype=np.float32)
    for a in range(M):
        # pairwise distances at current time
        d = np.linalg.norm(pos - pos[a], axis=1)  # [M]
        order = np.argsort(d)
        pieces, taken = [], 0
        for idx in order:
            if idx == a:
                continue
            if d[idx] > neighbor_radius:
                break
            vec = (pos[idx] - pos[a]) if relative else pos[idx]
            pieces.append(vec.astype(np.float32))
            taken += 1
            if taken == max_neighbors:
                break
        if taken < max_neighbors:
            pieces += [np.zeros((2,), dtype=np.float32)] * (max_neighbors - taken)
        if pieces:
            mvec[a] = np.concatenate(pieces, axis=0)
    return ids, mvec

def _features_from_obs_all(
    obs: dict,
    H: int,
    max_neighbors: int,
    neighbor_radius: float,
    neighbor_relative: bool,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Returns:
      ids  : [M]
      hist : [M, H, 2] float32   (xy history for rollout)
      z    : [M, 2*H + 2*max_neighbors] float32  (MDN input)
    """
    ids = sorted(obs.keys())
    if len(ids) == 0:
        return [], np.zeros((0, H, 2), np.float32), np.zeros((0, 2 * H + 2 * max_neighbors), np.float32)

    hist = np.stack([_past_array(obs[k], H)[:, :2] for k in ids], axis=0).astype(np.float32)  # [M,H,2]
    hvec = hist[:, ::-1, :].reshape(len(ids), 2 * H).astype(np.float32)                       # [M,2H]

    _, mvec = _neighbor_ctx_all_from_obs(
        obs, H, max_neighbors=max_neighbors, neighbor_radius=neighbor_radius, relative=neighbor_relative
    )                                                                                          # [M,2*max_neighbors]

    z = np.concatenate([hvec, mvec], axis=1).astype(np.float32)                                # [M, 2H + 2*max_neighbors]
    return ids, hist, z

# ---------- evaluate ALL agents directly from your obs dict ----------
def evaluate_from_obs_all(
    obs: dict,            # your {id: np.ndarray[H,D]}
    H: int,
    P: int,
    mdn,                  # MDN module; mixture_mean([M,D_in]) -> [M,2]
    K: np.ndarray,
    device: str = "cpu",
    max_neighbors: int = 0,
    neighbor_radius: float = 5.0,
    neighbor_relative: bool = True,
):
    """
    Returns:
      preds: dict[id] -> rollout_with_K(hist[a], g_mean[a], K, P)
    """
    ids, hist, z = _features_from_obs_all(
        obs, H,
        max_neighbors=max_neighbors,
        neighbor_radius=neighbor_radius,
        neighbor_relative=neighbor_relative,
    )
    if len(ids) == 0:
        return {}

    hx = torch.from_numpy(z).to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    with torch.no_grad():
        g_mean = mdn.mixture_mean(hx).detach().cpu().numpy().reshape(len(ids), 2)  # [M,2]

    preds = {agent_id: rollout_with_K(hist[a], g_mean[a], K, P) for a, agent_id in enumerate(ids)}
    return preds

# ---------------------- Main ----------------------

def create():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/home/snowhan/conformal_prediction/lobby2/biwi_hotel/train')
    parser.add_argument('--test_file', type=str, default='/home/snowhan/tools_paper/CANavi/biwi_hotel.npy')
    parser.add_argument('--history', type=int, default=HISTORY_LENGTH_DEFAULT)
    parser.add_argument('--pred', type=int, default=PRED_LENGTH_DEFAULT)
    parser.add_argument('--mdn_K', type=int, default=MDN_COMPONENTS_DEFAULT)
    parser.add_argument('--mdn_hidden', type=int, default=MDN_HIDDEN_DEFAULT)
    parser.add_argument('--mdn_epochs', type=int, default=MDN_EPOCHS_DEFAULT)
    parser.add_argument('--mdn_batch', type=int, default=MDN_BATCH_DEFAULT)
    parser.add_argument('--mdn_lr', type=float, default=MDN_LR_DEFAULT)
    parser.add_argument('--sigma_min', type=float, default=SIGMA_MIN_DEFAULT)
    parser.add_argument('--ridge', type=float, default=RIDGE_LAMBDA_DEFAULT)
    parser.add_argument('--samples', type=int, default=SAMPLES_DEFAULT)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_mdn', type=str, default='mdn.pt')
    parser.add_argument('--save_K', type=str, default='koopman_K_1.npy')
    parser.add_argument('--max_neighbors', type=int, default=MAX_NEIGHBORS_DEFAULT)
    parser.add_argument('--neighbor_radius', type=float, default=NEIGHBOR_RADIUS_DEFAULT)
    parser.add_argument('--neighbor_relative', type=int, default=1)

    args = parser.parse_args()

    H, P = args.history, args.pred
    maxN  = max(0, int(args.max_neighbors))
    R     = float(args.neighbor_radius)
    rel   = bool(int(args.neighbor_relative))

    train_files = glob.glob(os.path.join(args.train_dir, '*.npy'))
    if len(train_files) == 0:
        raise FileNotFoundError(f"No .npy files found under {args.train_dir}")
    print(f"Found {len(train_files)} train files.")
    X, Y = collect_mdn_dataset_multiagent(
        train_files, H, P, max_neighbors=maxN, neighbor_radius=R, neighbor_relative=rel
    )
    print(f"MDN dataset: X={X.shape}, Y={Y.shape}")

    in_dim = X.shape[1]
    mdn = train_mdn(
        X, Y, in_dim=in_dim, K=args.mdn_K, hidden=args.mdn_hidden,
        epochs=args.mdn_epochs, batch_size=args.mdn_batch, lr=args.mdn_lr,
        device=args.device
    )
    torch.save(mdn.state_dict(), args.save_mdn)
    print(f"Saved MDN to {args.save_mdn}")

    Pmat, Fmat = collect_koopman_pairs(train_files, H, P, use_bias=True)
    K = estimate_K(Pmat, Fmat, ridge=args.ridge)
    np.save(args.save_K, K)
    print(f"Estimated K with shape {K.shape}, saved to {args.save_K}")
    return H,P,mdn,K,args.samples, args.device, maxN,R,rel


