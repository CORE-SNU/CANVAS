import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .cp.adaptive_cp import AdaptiveConformalPredictionModule
from .env import Environment
from .visualization_utils import (
    draw_map3,
    visualize_tracking_result,
    visualize_prediction_result,
    visualize_cp_result2
)

def run_scenario(
    model,
    cp_type,
    miscoverage_level,
    n_pedestrians,
    test_dirpath,
    map_size,
    threshold,
    mu,
    cov_inv,
    history_length=8,
    prediction_length=12,
    r_star=None,
    t_begin=40,
    t_end=200,
    mode="feature",
    bg_img_path='lobby3.png'
):
    # ---------------------------------------------------
    # Helper: visualize an extreme case and save PNG only
    # ---------------------------------------------------
    def visualize_extreme_case(filepath_prefix, info):
        ped_positions = info["ped_positions"]
        ood_pids = info["ood_pids"]
        valid_pids = info["valid_pids"]
        pred_res = info["pred_res"]
        intervals = info["intervals"]

        def draw_scene(ax):
            # Draw trajectories of valid pedestrians and mark current position
            for pid in valid_pids:
                if pid in info["window_positions"]:
                    traj = info["window_positions"][pid]
                    ax.plot(traj[:, 0], traj[:, 1], 'k--', linewidth=1, zorder=70)

                    # Use position at t_step (end of history) instead of start
                    if traj.shape[0] >= history_length:
                        current_pos = traj[history_length - 1]
                    else:
                        current_pos = traj[-1]  # fallback if shorter
                    ax.add_patch(Circle(current_pos, 0.15, color='blue', zorder=95))

            # Show conformal prediction intervals
            selected_steps = [1, 6, 11]
            visualize_cp_result2(intervals, pred_res, selected_steps, ax, rotated_img, extent)

            # r_star visualization
            if 'r_star' in info:
                r_star_vals = np.array(info['r_star'])
            else:
                r_star_vals = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3])

            for pid, pred_traj in pred_res.items():
                if pred_traj is None:
                    continue
                for step_idx in selected_steps:
                    if step_idx < len(pred_traj) and step_idx < len(r_star_vals):
                        pos = pred_traj[step_idx]
                        x0, y0 = pos
                        r_val = r_star_vals[step_idx]
                        i_val = intervals[step_idx]

                        # Draw baseline r_star (blue)
                        ax.add_patch(Circle(
                            pos,
                            radius=r_val,
                            facecolor='blue',
                            alpha=0.15,
                            edgecolor='none',
                            zorder=40
                        ))

                        # Difference line
                        diff = i_val - r_val
                        ax.plot(
                            [x0 + r_val, x0 + i_val],
                            [y0, y0],
                            color='red',
                            linewidth=2,
                            zorder=50
                        )

                        # Annotate difference
                        ax.text(
                            x0 + (r_val + i_val) / 2,
                            y0 + 0.1,
                            f"{diff:+.2f}",
                            color='red',
                            fontsize=8,
                            ha='center',
                            va='bottom',
                            zorder=60
                        )

        # --------- FULL MAP VIEW ---------
        fig, ax, rotated_img, extent = draw_map3(*info["map_size"], bg_img_path=bg_img_path)
        ax.set_title(f"Extreme case (radius={info['radius']:.3f}) at t={info['t_step']}", fontsize=10)
        draw_scene(ax)
        fig.tight_layout()
        plt.savefig(filepath_prefix + "_full.png", dpi=200)
        plt.close(fig)

        # --------- ZOOMED-IN VIEW (if at least 3 pedestrians) ---------
        if len(valid_pids) >= 3:
            fig, ax, rotated_img, extent = draw_map3(*info["map_size"], bg_img_path=bg_img_path)
            ax.set_title(f"Extreme case (zoomed) radius={info['radius']:.3f}", fontsize=10)

            # Zoom around the valid pedestrians
            valid_positions = ped_positions[valid_pids, :]
            x_min, y_min = valid_positions.min(axis=0)
            x_max, y_max = valid_positions.max(axis=0)
            pad = 2.0
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)

            draw_scene(ax)
            fig.tight_layout()
            plt.savefig(filepath_prefix + "_zoom.png", dpi=400)
            plt.close(fig)

    # ---------------------------------------------------
    # Main code
    # ---------------------------------------------------
    if r_star is None:
        r_star = [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3]

    window_size = history_length + prediction_length
    thresh = threshold

    dt = 0.1
    pred_len = len(r_star)
    cp_module = AdaptiveConformalPredictionModule(
        target_miscoverage_level=miscoverage_level,
        step_size=0.0 if cp_type == 'split' else 0.05,
        n_scores=pred_len,
        max_interval_lengths=0.5 * dt * np.arange(1, pred_len + 1),
        sample_size=20,
        offline_calibration_set={i: [] for i in range(pred_len)}
    )
    radii_id, radii_ood = [], []
    d2_low = 0.0

    # Track extremes
    max_radius, min_radius = -np.inf, np.inf
    max_info, min_info = None, None

    for fname in sorted(os.listdir(test_dirpath)):
        if not fname.endswith('.txt'):
            continue

        base = os.path.splitext(fname)[0]
        y_pred = np.load(os.path.join(test_dirpath, f"{base}_{model}_predictions.npy"))
        y_true = np.load(os.path.join(test_dirpath, f"{base}_targets.npy"), allow_pickle=True)

        env = Environment(
            filepath=os.path.join(test_dirpath, base + '.npy'),
            dt=dt,
            n_pedestrians=n_pedestrians,
            t_begin=t_begin,
            t_end=t_end
        )
        data = env._data
        env.reset()
        t_step = t_begin

        valid, detected = [], []
        for pid in range(y_pred.shape[2]):
            if len(valid) >= n_pedestrians:
                break
            if not np.isnan(y_true[t_step, :, pid]).any():
                valid.append(pid)
                detected.append(pid)

        while t_step < t_end:
            obs = env._get_obs(valid)
            pred_res = {i: y_pred[t_step, :, i] for i in valid}
            intervals = cp_module.update(obs, pred_res) if valid else np.array([np.nan] * pred_len)

            start = t_step - (history_length - 1)
            endw = start + window_size

            id_set = set()
            window_dict = {}
            if 0 <= start and endw <= data.shape[0]:
                for pid in valid:
                    window = data[start:endw, pid, :]
                    if window.shape[0] != window_size or np.isnan(window).any():
                        continue

                    disps = np.linalg.norm(np.diff(window, axis=0), axis=1)
                    total_disp = disps.sum()
                    total_dt = (window_size - 1) * dt
                    mean_spd = total_disp / total_dt
                    std_spd = np.std(disps / dt)

                    spd = disps / dt
                    acc = np.diff(spd)
                    ang = np.arctan2(np.diff(window[:, 1]), np.diff(window[:, 0]))
                    curv = np.abs(np.diff(ang))

                    if mode == "feature":
                        feat = np.array([
                            mean_spd,
                            std_spd,
                            acc.mean() if acc.size else 0.0,
                            acc.std() if acc.size else 0.0,
                            curv.mean() if curv.size else 0.0
                        ])
                    else:
                        feat = window.flatten()

                    delta = feat - mu
                    d2 = float(delta @ cov_inv @ delta)
                    d2_low = max(d2_low, d2)

                    if d2 > thresh:
                        id_set.add(pid)
                    window_dict[pid] = window

            shifts = (np.array(r_star) - intervals) / np.array(r_star)
            with np.errstate(invalid='ignore'):
                radius = np.nanmean(shifts)

            # Use the position at t_step (history_length - 1) for snapshot
            if not np.isnan(radius) and len(valid) >= 3:
                ped_positions_snapshot = data[t_step, :, :].copy()
                for pid, window in window_dict.items():
                    if window.shape[0] >= history_length:
                        ped_positions_snapshot[pid] = window[history_length - 1]

                snapshot = {
                    "radius": radius,
                    "t_step": t_step,
                    "map_size": map_size,
                    "ped_positions": ped_positions_snapshot,
                    "window_positions": window_dict.copy(),
                    "ood_pids": list(id_set),
                    "valid_pids": list(valid),
                    "pred_res": pred_res,
                    "intervals": intervals,
                    "r_star": r_star,
                }
                if radius > max_radius:
                    max_radius = radius
                    max_info = snapshot
                if radius < min_radius:
                    min_radius = radius
                    min_info = snapshot

            for pid in valid:
                (radii_ood if pid in id_set else radii_id).append(radius)

            _, done = env.step()
            t_step += 1

            # update valid pedestrians
            new_valid = []
            for pid in range(y_pred.shape[2]):
                if len(new_valid) >= n_pedestrians:
                    break
                all_nan = np.isnan(y_true[t_step, :, pid]).all()
                had_data = not np.isnan(y_true[:t_step, :, pid]).all()
                any_nan = np.isnan(y_true[t_step, :, pid]).any()
                if pid in detected:
                    if not all_nan:
                        new_valid.append(pid)
                elif not any_nan and not had_data:
                    new_valid.append(pid)
                    detected.append(pid)
            valid = new_valid

    # Save final visualizations only
    if max_info is not None:
        print("MAX")
        visualize_extreme_case(os.path.join(test_dirpath, "extreme_max"), max_info)
    if min_info is not None:
        print("MIN")
        visualize_extreme_case(os.path.join(test_dirpath, "extreme_min"), min_info)

    return (
        np.nanmean(radii_id), np.nanstd(radii_id),
        np.nanmean(radii_ood), np.nanstd(radii_ood)
    )
