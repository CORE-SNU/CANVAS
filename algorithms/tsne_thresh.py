import glob
import os
import numpy as np
import pandas as pd

def load_trajectory_files(patterns):
    file_dfs = {}
    for pattern in patterns:
        for fn in glob.glob(pattern):
            df = pd.read_csv(
                fn,
                sep=r'\s+',
                header=None,
                names=['frame','ped_id','x','y'],
                dtype={'frame': int, 'ped_id': int, 'x': float, 'y': float},
                na_values=['','nan']
            ).dropna(subset=['frame','ped_id','x','y'])
            file_dfs[fn] = df
    return file_dfs

def extract_features(file_dfs, window_size, dt_factor=0.1,mode='feature'):
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

            disps      = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            total_disp = disps.sum()
            total_dt   = (frames[-1] - frames[0]) * dt_factor
            if total_dt == 0:
                continue
            mean_spd = total_disp / total_dt
            std_spd  = np.std(disps / dt_factor)

            spd  = disps / dt_factor
            acc  = np.diff(spd)
            ang  = np.arctan2(np.diff(coords[:,1]), np.diff(coords[:,0]))
            curv = np.abs(np.diff(ang))

            if mode=="feature":
                feats.append([
                mean_spd,
                std_spd,
                np.nanmean(acc)  if acc.size else 0.0,
                np.nanstd(acc)   if acc.size else 0.0,
                np.nanmean(curv) if curv.size else 0.0
            ])
            else:
                feats.append(coords.flatten())  
    return np.array(feats)

def compute_perplexity_sigmas(
    X: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50
) -> np.ndarray:
    n = X.shape[0]
    D2 = np.sum((X[:,None,:] - X[None,:,:])**2, axis=-1)
    sigmas = np.zeros(n, dtype=float)
    logP = np.log2(perplexity)

    for i in range(n):
        dist2 = D2[i].copy()
        dist2[i] = np.inf
        lo, hi, sigma = 1e-20, 1e5, 1.0
        for _ in range(max_iter):
            P = np.exp(-dist2 / (2 * sigma*sigma))
            S = P.sum()
            if S == 0:
                break
            P /= S
            H = -np.sum(P[P>0] * np.log2(P[P>0]))
            diff = H - logP
            if abs(diff) < tol:
                break
            if H > logP:
                hi    = sigma
                sigma = 0.5 * (sigma + lo)
            else:
                lo    = sigma
                sigma = 0.5 * (sigma + hi)
        sigmas[i] = sigma

    return sigmas

def compute_tsne_perplexity_thresholds(
    train_dir: str,
    contamination_list: list[float],
    perplexity: float,
    history_length: int = 8,
    prediction_length: int = 12,
    dt_factor: float = 0.1,mode='feature'
) -> tuple[dict[float, float], np.ndarray, np.ndarray]:
    """
    Train‐set perplexity‐based OOD thresholds.

    Args:
      train_dir: directory of train .txt files
      contamination_list: list of alphas
      perplexity: target t-SNE perplexity P
      history_length: H (history frames)
      prediction_length: N (future frames)
      dt_factor: seconds per frame

    Returns:
      thresholds: {alpha → sigma-threshold}
      sigmas:     np.array of all training σ_i
      X_train:    training feature matrix (n_samples×5)
    """
    window_size = history_length + prediction_length

    patterns = [os.path.join(train_dir, '*.txt')]
    dfs      = load_trajectory_files(patterns)

    # build training feature matrix
    X_train = extract_features(dfs, window_size, dt_factor,mode)

    # compute per-window sigmas for target perplexity
    sigmas = compute_perplexity_sigmas(X_train, perplexity)

    thresholds = {
        alpha: float(np.quantile(sigmas, 1 - alpha))
        for alpha in contamination_list
    }

    return thresholds, sigmas, X_train
