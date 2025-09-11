# save_ci.py
import os
import csv
import numpy as np
import pathlib

# --- Matplotlib (headless) ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D


# ============================================================
# Save CI to .csv files per iteration 
# ============================================================
def save_ci_iteration_csv(iter_out_dir,
                          iteration_index,
                          it_ci_traj_series,
                          it_ci_ctrl_series,
                          it_ci_obj,
                          it_ci_ctrl_cost,
                          prediction_len):
    """
    Save one CSV per iteration with all per-frame CI results.
    Path: <iter_out_dir>/ci_iter_<iteration_index>.csv

    Columns:
      frame, ci_traj_avg, ci_traj_final, ci_ctrl_avg, ci_ctrl_final, ci_obj, ci_ctrl_cost,
      ci_traj_s0..s{T-1}, ci_ctrl_s0..s{T-1}
    """
    iter_out_dir = pathlib.Path(iter_out_dir)
    iter_out_dir.mkdir(parents=True, exist_ok=True)
    ci_csv_path = iter_out_dir / f"ci_iter_{iteration_index:03d}.csv"

    step_cols_traj = [f"ci_traj_s{j}" for j in range(prediction_len)]
    step_cols_ctrl = [f"ci_ctrl_s{j}" for j in range(prediction_len)]
    fieldnames = (["frame",
                   "ci_traj_avg", "ci_traj_final",
                   "ci_ctrl_avg", "ci_ctrl_final",
                   "ci_obj", "ci_ctrl_cost"]
                  + step_cols_traj + step_cols_ctrl)

    n_frames = max(len(it_ci_traj_series),
                   len(it_ci_ctrl_series),
                   len(it_ci_obj),
                   len(it_ci_ctrl_cost))

    with open(ci_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx in range(n_frames):
            traj_series = it_ci_traj_series[frame_idx] if frame_idx < len(it_ci_traj_series) else np.array([])
            ctrl_series = it_ci_ctrl_series[frame_idx] if frame_idx < len(it_ci_ctrl_series) else np.array([])

            traj_avg   = float(np.nanmean(traj_series)) if getattr(traj_series, "size", 0) else np.nan
            traj_final = float(traj_series[-1])         if getattr(traj_series, "size", 0) else np.nan
            ctrl_avg   = float(np.nanmean(ctrl_series)) if getattr(ctrl_series, "size", 0) else np.nan
            ctrl_final = float(ctrl_series[-1])         if getattr(ctrl_series, "size", 0) else np.nan

            row = {
                "frame":    int(frame_idx),
                "ci_traj_avg":   traj_avg,
                "ci_traj_final": traj_final,
                "ci_ctrl_avg":   ctrl_avg,
                "ci_ctrl_final": ctrl_final,
                "ci_obj":        float(it_ci_obj[frame_idx]) if frame_idx < len(it_ci_obj) else np.nan,
                "ci_ctrl_cost":  float(it_ci_ctrl_cost[frame_idx]) if frame_idx < len(it_ci_ctrl_cost) else np.nan,
            }

            for j in range(prediction_len):
                row[f"ci_traj_s{j}"] = float(traj_series[j]) if getattr(traj_series, "size", 0) > j else np.nan
                row[f"ci_ctrl_s{j}"] = float(ctrl_series[j]) if getattr(ctrl_series, "size", 0) > j else np.nan

            writer.writerow(row)

    return str(ci_csv_path)


# ============================================================
# CI visualizer to .png file per frame
# ============================================================
def save_frame_png(outdir,
                   frame_idx,
                   static_boxes,
                   robot_xy,
                   robot_traj_xy,
                   goal_xy,
                   valid_obs=None,
                   valid_obs_future_true=None,
                   prediction_res=None,
                   r_star=None,
                   steps_to_annotate=(2, 5, 10),
                   annotate_ci=False,
                   ci_decimals=2,
                   ci_fontsize=7,
                   max_ci_annotations_per_step=None,
                   xlim=(-2.5, 10.0),  #(-7.5, 13.5)
                   ylim=(-10.0, 2.0)): #(-12.5, 5.5)
    """
    Draw history / GT future / prediction with static boxes and robot.
    Saves: <outdir>/frame_<frame_idx>.png
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Static boxes - gray
    if static_boxes:
        for b in static_boxes:
            if getattr(b, "vertices", None) is None:
                continue
            poly = Polygon(b.vertices, closed=True,
                           facecolor='gray', edgecolor='gray',
                           linewidth=1, zorder=0.1)
            ax.add_patch(poly)

    # Robot trajectory / current / goal
    px, py = robot_traj_xy
    if len(px) and len(py):
        ax.plot(px, py, linewidth=2)
    ax.scatter([robot_xy[0]], [robot_xy[1]], marker='o', s=30)
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker='*', s=80)

    # History(8)
    if valid_obs:
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)

    # GT future(12)
    if valid_obs_future_true:
        for _, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], linestyle='--', color='black', linewidth=1)

    # Prediction(12)
    if prediction_res:
        for _, arr in prediction_res.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], color='red', linewidth=1.5)

    # Optional CI annnotations (at steps 2/5/10 by default)
    if annotate_ci and (r_star is not None) and (r_star > 0) and valid_obs_future_true and prediction_res:
        common = set(prediction_res.keys()) & set(valid_obs_future_true.keys())
        offsets = {2: (0.05, 0.05), 5: (-0.05, 0.05), 10: (0.05, -0.05)}
        ann_counts = {s: 0 for s in steps_to_annotate}
        for s in steps_to_annotate:
            j = s - 1
            for pid in common:
                pred = np.asarray(prediction_res[pid], dtype=np.float64)
                gt   = np.asarray(valid_obs_future_true[pid], dtype=np.float64)
                if (pred.ndim == 2 and gt.ndim == 2 and pred.shape[1] >= 2 and gt.shape[1] >= 2
                    and len(pred) > j and len(gt) > j):
                    p = pred[j, :2]; g = gt[j, :2]
                    if not (np.isfinite(p).all() and np.isfinite(g).all()):
                        continue
                    err = float(np.linalg.norm(p - g))
                    ax.add_patch(Circle((p[0], p[1]), r_star,
                                        fill=True, edgecolor='none', facecolor='lightgray', zorder=0.5))
                    ax.add_patch(Circle((p[0], p[1]), err,
                                        fill=True, edgecolor='black', facecolor='orange', linewidth=1, zorder=1.0))
                    if (max_ci_annotations_per_step is None) or (ann_counts[s] < max_ci_annotations_per_step):
                        ci = (r_star - err) / r_star
                        if np.isfinite(ci):
                            dx, dy = offsets.get(s, (0.04, 0.04))
                            ax.text(p[0] + dx, p[1] + dy,
                                    f"CI@t+{s}={ci:.{ci_decimals}f}",
                                    fontsize=ci_fontsize, zorder=3.0)
                            ann_counts[s] += 1

    legend_elements = [
        Line2D([0], [0], color='navy',  lw=1,   linestyle='-',  label='History (8)'),
        Line2D([0], [0], color='black', lw=1,   linestyle='--', label='GT future (12)'),
        Line2D([0], [0], color='red',   lw=1.5, linestyle='-',  label='Prediction (12)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"frame_{frame_idx:05d}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)
