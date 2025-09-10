import glob
import os
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance

def load_trajectory_files(patterns):
    """
    Given a list of glob patterns, load all matching .txt trajectory files
    into a dict mapping filename → DataFrame with columns [frame, ped_id, x, y].
    """
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

def extract_features(file_dfs, window_size=20, dt_factor=0.1, mode="feature"):
    """
    For each trajectory file and each pedestrian ID, take the last `window_size`
    frames, compute the 5‑dim feature vector [mean_spd, std_spd, mean_acc,
    std_acc, mean_curv], and return an (n_samples × 5) numpy array.

    mean_spd is now computed as total_distance / total_time over the window.
    """
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

            # total displacement and total time
            disps       = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            total_disp  = disps.sum()
            total_dt    = (frames[-1] - frames[0]) * dt_factor
            if total_dt == 0:
                continue

            mean_spd    = total_disp / total_dt
            std_spd     = np.std(disps / dt_factor)

            # accelerations and curvature computed on per-step speeds
            spd         = disps / dt_factor
            acc         = np.diff(spd)
            ang         = np.arctan2(np.diff(coords[:,1]), np.diff(coords[:,0]))
            curv        = np.abs(np.diff(ang))

            
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

def compute_thresholds(
    train_dir: str,
    contamination_list: list[float],
    window_size: int = 20,
    dt_factor: float = 0.1,
     mode="feature"
) -> tuple[dict[float, float], np.ndarray, np.ndarray]:
    """
    Train a Mahalanobis model on all .txt files in `train_dir` and return:

      - thresholds: dict mapping each contamination → threshold d^2
      - mu        : mean vector (shape (5,))
      - cov_inv   : precision matrix (shape (5,5))
    """
    # load all .txt in the directory
    patterns = [os.path.join(train_dir, '*.txt')]
    dfs      = load_trajectory_files(patterns)

    # extract 5‑dim features
    X = extract_features(dfs, window_size, dt_factor, mode)

    # fit empirical covariance
    cov     = EmpiricalCovariance().fit(X)
    mu      = cov.location_       # shape (5,)
    cov_inv = cov.precision_      # shape (5,5)

    # compute Mahalanobis distances on training set
    d2 = cov.mahalanobis(X)

    # for each contamination, threshold at the (1−c)-quantile
    thresholds = {
        c: float(np.quantile(d2, 1 - c))
        for c in contamination_list
    }

    return thresholds, mu, cov_inv
