#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDN + Koopman — Evaluate from OBS (minimal)

Changes vs your previous trimmed version:
  - Restore MDN (with .sample back in).
  - Add MDNKoopmanEvaluator class:
      __init__(cfg_path, mdn_pt_path=None, device=None, ...)
        -> loads cfg, MDN, K
      __call__(obs) -> evaluate_from_obs_all(...)
  - Keep helpers + evaluate_from_obs_all unchanged for minimal diff.
"""

import os
import json
import math
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------- Constants (compatibility / defaults) ----------------------


# ---------------------- MDN (inference; restored .sample) ----------------------

LOG2PI = math.log(2.0 * math.pi)

class MDN(nn.Module):
    def __init__(self, in_dim: int, n_components: int,
                 hidden: int , sigma_min: float ):
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
        # clamp sigma for numerical stability
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
        comps = torch.multinomial(pi, num_samples=n_samples, replacement=True)  # [B, nS]
        mu_g = torch.gather(mu, 1, comps.unsqueeze(-1).expand(B, n_samples, 2))
        sigma = torch.exp(torch.gather(log_sigma, 1, comps.unsqueeze(-1).expand(B, n_samples, 2)))
        eps = torch.randn_like(mu_g)
        return mu_g + sigma * eps

# ---------------------- psi builders + rollout ----------------------

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

# ---------------------- OBS helpers (unchanged) ----------------------

def _past_array(entry, H: int) -> np.ndarray:
    """Normalize an obs entry to [H, D] float32 (left-pad by repeating first row if needed)."""
    if isinstance(entry, dict) and "past" in entry:
        arr = np.asarray(entry["past"], dtype=np.float32)
    else:
        arr = np.asarray(entry, dtype=np.float32)  # typical case: [H,D]
    if arr.shape[0] < H:
        pad = np.repeat(arr[:1], H - arr.shape[0], axis=0)
        arr = np.concatenate([pad, arr], axis=0)
    return arr[-H:, :]  # ensure exactly H

def _neighbor_ctx_all_from_obs(
    obs: Dict[Any, Any],
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
    obs: Dict[Any, Any],
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

# ---------------------- Evaluate from OBS (unchanged function) ----------------------

def evaluate_from_obs_all(
    obs: dict,            # {id: np.ndarray[H,D]} OR {id: {'past':[H,D], ...}}
    H: int,
    P: int,
    mdn: MDN,             # MDN module; mixture_mean([M,D_in]) -> [M,2]
    K: np.ndarray,
    device: str,
    max_neighbors: int,
    neighbor_radius: float,
    neighbor_relative: bool ,
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
    with torch.no_grad():
        g_mean = mdn.mixture_mean(hx).detach().cpu().numpy().reshape(len(ids), 2)  # [M,2]

    preds = {agent_id: rollout_with_K(hist[a], g_mean[a], K, P) for a, agent_id in enumerate(ids)}
    return preds

# ---------------------- Lightweight loaders (used internally by the class) ----------------------

def _resolve_device(device_str: str) -> str:
    if device_str and device_str.startswith("cuda") and torch.cuda.is_available():
        return device_str
    return "cpu"

def _load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    # minimal validation
    required = ["H", "P", "K_path", "feature_in_dim", "max_neighbors", "neighbor_radius", "neighbor_relative"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in cfg: {cfg_path}")
    return cfg

def _load_K(K_path: str) -> np.ndarray:
    K = np.load(K_path)
    if K.ndim != 2:
        raise ValueError(f"K must be 2D, got shape {K.shape}")
    return K

def _load_mdn(mdn_pt_path: str, in_dim: int, device: str,
              mdn_K: int ,
              mdn_hidden: int ,
              sigma_min: float ) -> MDN:
    model = MDN(in_dim, n_components=mdn_K, hidden=mdn_hidden, sigma_min=sigma_min).to(device)
    state = torch.load(mdn_pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
