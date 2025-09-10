import os
import numpy as np
from sklearn.neighbors import KernelDensity
from .cp.adaptive_cp import AdaptiveConformalPredictionModule
from .env import Environment


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


def run_scenario_kde(
    model: str,
    cp_type: str,
    miscoverage_level: float,
    n_pedestrians: int,
    test_dirpath: str,
    map_size: list[float],
    thresholds: dict[float, float],
    kde: KernelDensity,
    contamination: float,
    history_length: int = 8,
    prediction_length: int = 12,
    r_star: list[float] | None = None,
    t_begin: int = 40,
    t_end: int = 200,
    mode: str = "raw"
) -> tuple[float, float, float, float]:
    """
    Sliding-window OOD detection using KDE densities as scores.
    mode: "raw" → normalize+flatten windows; "feature" → 5-d summary features
    """
    if r_star is None:
        r_star = [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3]
    window_size = history_length + prediction_length

    # density threshold (probability space)
    prob_thr = thresholds[contamination]
    log_thr  = np.log(prob_thr)

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
    count_id, count_ood, logp_low = 0, 0, np.inf

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
        env.reset()

        t_step     = t_begin

        # initial valid set
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
                    if window.shape[0] != window_size or np.isnan(window).any():
                        continue

                    # choose features based on mode
                    if mode == "feature":
                        # summary features
                        disps      = np.linalg.norm(np.diff(window, axis=0), axis=1)
                        total_disp = disps.sum()
                        total_dt   = (window_size-1) * dt
                        mean_spd   = total_disp / total_dt
                        std_spd    = np.std(disps / dt)
                        spd        = disps / dt
                        acc        = np.diff(spd)
                        ang        = np.arctan2(np.diff(window[:,1]), np.diff(window[:,0]))
                        curv       = np.abs(np.diff(ang))
                        feat       = np.array([
                            mean_spd,
                            std_spd,
                            acc.mean()  if acc.size  else 0.0,
                            acc.std()   if acc.size  else 0.0,
                            curv.mean() if curv.size else 0.0
                        ]).reshape(1, -1)
                    else:
                        # normalize + raw
                        seg_norm = normalize(window)
                        feat     = seg_norm.flatten().reshape(1, -1)

                    # log density
                    log_prob = kde.score_samples(feat)[0]
                    if log_prob < logp_low:
                        logp_low = log_prob

                    # classify
                    if log_prob >= log_thr:
                        id_set.add(pid)
                        count_id += 1
                    else:
                        count_ood += 1

            # conformal radii
            shifts = (np.array(r_star) - intervals) / np.array(r_star)
            radius = np.nanmean(shifts)
            for pid in valid:
                (radii_id if pid in id_set else radii_ood).append(radius)

            # step
            _, done = env.step()

            t_step += 1
            # refresh valid
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

    print(f"OOD count: {count_ood}, ID count: {count_id}")
    print(f"Density threshold (prob): {prob_thr}")
    print(f"Lowest log-density seen: {logp_low}")

    return (
        np.nanmean(radii_id), np.nanstd(radii_id),
        np.nanmean(radii_ood), np.nanstd(radii_ood)
    )
