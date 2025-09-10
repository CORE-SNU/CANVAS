import os
import numpy as np
from .cp.adaptive_cp import AdaptiveConformalPredictionModule
from .env import Environment

def _compute_sigma_single(x, X_train, perplexity, tol=1e-5, max_iter=50):
    """
    Binary‐search for sigma so that the perplexity of x wrt X_train equals target.
    """
    # squared distances to all training points
    dist2 = np.sum((X_train - x)**2, axis=1)
    logP  = np.log2(perplexity)
    lo, hi, sigma = 1e-20, 1e5, 1.0

    for _ in range(max_iter):
        P = np.exp(-dist2 / (2 * sigma*sigma))
        if P.sum() == 0:
            break
        P /= P.sum()
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

    return sigma

def run_scenario_tsne(
    model: str,
    cp_type: str,
    miscoverage_level: float,
    n_pedestrians: int,
    test_dirpath: str,

    map_size: list[float],
    thresholds: float,
    X_train: np.ndarray,
    perplexity: float,
    history_length: int = 8,
    prediction_length: int = 12,
    r_star: list[float] | None = None,
    t_begin: int = 40,
    t_end: int = 200,
     mode="feature"
) -> tuple[float, float, float, float]:
    """
    TSNE‐perplexity based OOD detection:
      - X_train: (N_train x 5) array of training features
      - thresholds[c]: σ‐threshold for contamination c
      - perplexity: target t‑SNE perplexity
    """
    if r_star is None:
        r_star = [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3]
    window_size = history_length + prediction_length

    # conformal predictor
    dt       = 0.1
    pred_len = len(r_star)
    cp_module = AdaptiveConformalPredictionModule(
        target_miscoverage_level=miscoverage_level,
        step_size=0.0 if cp_type=='split' else 0.05,
        n_scores=pred_len,
        max_interval_lengths=0.5 * dt * np.arange(1, pred_len+1),
        sample_size=20,
        offline_calibration_set={i: [] for i in range(pred_len)}
    )

 
    radii_id, radii_ood = [], []
    numbers, number2, d2_low = 0, 0, 0.0

    for fname in sorted(os.listdir(test_dirpath)):
        if not fname.endswith('.txt'):
            continue

        base   = os.path.splitext(fname)[0]
        y_pred = np.load(os.path.join(test_dirpath, f"{base}_{model}_predictions.npy"))
        y_true = np.load(os.path.join(test_dirpath, f"{base}_targets.npy"), allow_pickle=True)

        env        = Environment(
            filepath=os.path.join(test_dirpath, base + '.npy'),
            dt=dt,
            n_pedestrians=n_pedestrians,
            t_begin=t_begin,
            t_end=t_end
        )
        data       = env._data
        t_step     = t_begin

        valid, detected = [], []
        for pid in range(y_pred.shape[2]):
            if len(valid) >= n_pedestrians:
                break
            if not np.isnan(y_true[t_step,:,pid]).any():
                valid.append(pid)
                detected.append(pid)

        while t_step < t_end:
            obs       = env._get_obs(valid)
            pred_res  = {i: y_pred[t_step,:,i] for i in valid}
            intervals = cp_module.update(obs, pred_res) if valid else np.array([np.nan]*pred_len)

            start = t_step - (history_length - 1)
            endw  = start + window_size

            id_set = set()
            if 0 <= start and endw <= data.shape[0]:
                for pid in valid:
                    window = data[start:endw, pid, :]
                    if window.shape[0]!=window_size or np.isnan(window).any():
                        continue

                    # compute the same 5-D feature vector
                    disp = np.linalg.norm(np.diff(window,axis=0),axis=1)
                    total_disp = disp.sum()
                    total_dt   = (window_size-1)*dt
                    mean_spd   = total_disp/total_dt
                    std_spd    = np.std(disp/dt)
                    spd        = disp/dt
                    acc        = np.diff(spd)
                    ang        = np.arctan2(np.diff(window[:,1]), np.diff(window[:,0]))
                    curv       = np.abs(np.diff(ang))
                    if mode=="feature":
                        feat = np.array([
                                    mean_spd,
                                    std_spd,
                                    acc.mean()  if acc.size  else 0.0,
                                    acc.std()   if acc.size  else 0.0,
                                    curv.mean() if curv.size else 0.0
                                ])
                    else:
                        feat = window.flatten()

                    # compute sigma(x) against X_train
                    sigma = _compute_sigma_single(feat, X_train, perplexity)
                    if sigma > d2_low:
                        d2_low = sigma

                    thresh = thresholds # use same key
                    if sigma <= thresh:
                        id_set.add(pid)
                        numbers += 1
                    else:
                        number2 += 1

            shifts = (np.array(r_star) - intervals) / np.array(r_star)
            radius = np.nanmean(shifts)
            for pid in valid:
                if pid in id_set:
                    radii_id.append(radius)
                else:
                    radii_ood.append(radius)
            _, done = env.step()

            t_step += 1
            new_valid = []
            for pid in range(y_pred.shape[2]):
                if len(new_valid) >= n_pedestrians:
                    break
                all_nan  = np.isnan(y_true[t_step,:,pid]).all()
                had_data = not np.isnan(y_true[:t_step,:,pid]).all()
                any_nan  = np.isnan(y_true[t_step,:,pid]).any()
                if pid in detected:
                    if not all_nan:
                        new_valid.append(pid)
                elif not any_nan and not had_data:
                    new_valid.append(pid)
                    detected.append(pid)
            valid = new_valid

    # debug prints
    print("ID count:", numbers, "OOD count:", number2)
    print("Max sigma seen:", d2_low)

    return (
        np.nanmean(radii_id), np.nanstd(radii_id),
        np.nanmean(radii_ood), np.nanstd(radii_ood)
    )
