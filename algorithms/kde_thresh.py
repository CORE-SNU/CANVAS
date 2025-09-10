#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

# summary feature extractor (5‑dim) for mode="feature"
def extract_features(file_dfs, window_size=20, dt_factor=0.1, mode="feature"):
    feats = []
    for df in file_dfs.values():
        for pid, grp in df.groupby('ped_id'):
            grp = grp.sort_values('frame')
            if len(grp) < window_size:
                continue
            window = grp.iloc[-window_size:]
            coords = window[['x','y']].to_numpy()
            frames = window['frame'].to_numpy()
            if np.isnan(coords).any():
                continue
            disps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            total_disp = disps.sum()
            total_dt = (frames[-1] - frames[0]) * dt_factor
            if total_dt == 0:
                continue
            mean_spd = total_disp / total_dt
            std_spd  = np.std(disps / dt_factor)
            spd      = disps / dt_factor
            acc      = np.diff(spd)
            ang      = np.arctan2(np.diff(coords[:,1]), np.diff(coords[:,0]))
            curv     = np.abs(np.diff(ang))
            if mode == "feature":
                feats.append([
                    mean_spd,
                    std_spd,
                    np.nanmean(acc)  if acc.size  else 0.0,
                    np.nanstd(acc)   if acc.size  else 0.0,
                    np.nanmean(curv) if curv.size else 0.0
                ])
            elif mode == "raw":
                feats.append(coords.flatten())
            else:
                raise ValueError(f"unknown mode {mode!r}")
    if not feats:
        n_feat = 5 if mode == "feature" else window_size * 2
        return np.empty((0, n_feat))
    return np.vstack(feats)

# load and normalize raw windows for mode="raw"
def load_trajectory_files(patterns):
    file_dfs = {}
    for pattern in patterns:
        for fn in glob.glob(pattern):
            df = pd.read_csv(
                fn,
                sep=r'\s+', header=None,
                names=['frame','ped_id','x','y'],
                dtype={'frame': int,'ped_id': int,'x': float,'y': float},
                na_values=['','nan']
            ).dropna(subset=['frame','ped_id','x','y'])
            file_dfs[fn] = df
    return file_dfs

def normalize(traj: np.ndarray) -> np.ndarray:
    origin = traj[0]
    traj = traj - origin
    if len(traj) < 2:
        return traj
    direction = traj[1]
    angle = np.arctan2(direction[1], direction[0])
    rot_angle = np.pi/2 - angle
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    R = np.array([[c, -s], [s, c]])
    return traj @ R.T

def extract_normalized_last_windows_txt(
    file_dfs: dict[str, pd.DataFrame],
    window_size: int
) -> np.ndarray:
    rows = []
    for df in file_dfs.values():
        for pid, grp in df.groupby('ped_id'):
            grp = grp.sort_values('frame')
            if len(grp) < window_size:
                continue
            seg = grp.iloc[-window_size:][['x','y']].to_numpy()
            if np.isnan(seg).any():
                continue
            seg_n = normalize(seg)
            rows.append(seg_n.flatten())
    if not rows:
        return np.empty((0, window_size*2))
    return np.vstack(rows)

# combined KDE threshold computation
def compute_kde_thresholds_txt(
    train_dir: str,
    contamination_list: list[float],
    bandwidth: float = 1.0,
    history_length: int = 8,
    prediction_length: int = 12,
    mode: str = "raw",
    dt_factor: float = 0.1
) -> tuple[dict[float, float], KernelDensity]:
    """
    Build a KDE model on either:
      - raw normalized windows (mode="raw"), or
      - 5‑dim summary features (mode="feature"),
    then compute thresholds for each alpha in contamination_list.

    Returns:
      - thresholds: alpha → minimal density for ID
      - kde_model : fitted KernelDensity instance
    """
    window_size = history_length + prediction_length
    patterns    = [os.path.join(train_dir, '*.txt')]
    dfs         = load_trajectory_files(patterns)

    if mode == "raw":
        X_train = extract_normalized_last_windows_txt(dfs, window_size)
    elif mode == "feature":
        X_train = extract_features(
            dfs,
            window_size=window_size,
            dt_factor=dt_factor,
            mode="feature"
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if X_train.size == 0:
        raise ValueError("No valid training windows found")

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_train)
    log_probs = kde.score_samples(X_train)
    probs     = np.exp(log_probs)

    thresholds = {
        alpha: float(np.quantile(probs, 1 - alpha))
        for alpha in contamination_list
    }

    return thresholds, kde
