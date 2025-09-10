import glob
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def load_trajectory_files(patterns):
    """
    Load all whitespace-delimited .txt files matching each glob pattern
    into a dict of DataFrames keyed by filename.
    """
    file_dfs = {}
    for pattern in patterns:
        for fn in glob.glob(pattern):
            df = pd.read_csv(
                fn,
                sep=r'\s+',
                header=None,
                names=['frame', 'ped_id', 'x', 'y'],
                dtype={'frame': int, 'ped_id': int, 'x': float, 'y': float},
                na_values=['','nan']
            ).dropna(subset=['frame','ped_id','x','y'])
            file_dfs[fn] = df
    return file_dfs


def extract_features(file_dfs, window_size=20, dt_factor=0.1, mode="feature"):
    """
    Extract either 5-d summary features or raw flattened coordinate windows:
      - mode="feature": [mean_spd, std_spd, mean_acc, std_acc, mean_curv]
      - mode="raw": flattened (window_size*2)-D [x1,y1,...,xN,yN]
    Returns an (n_samples x n_features) numpy array.
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

            # compute displacements and time
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


def extract_raw_last_windows(file_dfs, history_length=8, prediction_length=12):
    """
    Shortcut to extract only raw flattened windows, without summary features.
    """
    window_size = history_length + prediction_length
    rows = []
    for df in file_dfs.values():
        for pid, grp in df.groupby('ped_id'):
            grp = grp.sort_values('frame')
            if len(grp) < window_size:
                continue
            coords = grp[['x','y']].to_numpy()[-window_size:]
            if np.isnan(coords).any():
                continue
            rows.append(coords.flatten())
    if not rows:
        return np.empty((0, window_size*2))
    return np.vstack(rows)


def compute_knn_thresholds_raw(
    train_dir: str,
    contamination_list: list[float],
    k: int = 5,
    history_length: int = 8,
    prediction_length: int = 12,
  
    dt_factor: float = 0.1,
      mode: str = "raw"
) -> tuple[dict[float, float], NearestNeighbors, int]:
    """
    Build a KNN model on either raw or feature representations, then return:
      - thresholds: mapping contamination → k-th neighbor distance
      - neigh     : fitted NearestNeighbors instance
      - k         : the neighbor index used
    """
    patterns = [os.path.join(train_dir, '*.txt')]
    dfs      = load_trajectory_files(patterns)

    if mode == "raw":
        X = extract_raw_last_windows(dfs, history_length, prediction_length)
    elif mode == "feature":
        window_size = history_length + prediction_length
        X = extract_features(
            dfs,
            window_size=window_size,
            dt_factor=dt_factor,
            mode="feature"
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if X.shape[0] == 0:
        raise ValueError("No valid samples found in training data")

    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(X)
    dists, _ = neigh.kneighbors(X)
    dk = dists[:, k]

    thresholds = {
        c: float(np.quantile(dk, 1 - c))
        for c in contamination_list
    }

    return thresholds, neigh, k
