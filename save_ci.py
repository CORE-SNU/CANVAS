# save_ci.py
import os
import csv
import numpy as np
import pathlib
from typing import Dict, Tuple, Optional, Iterable
from src.canvas import Box

# --- Matplotlib (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import cv2

def _fallback_to_image_frame(pos: np.ndarray, H: np.ndarray):
    N = pos.shape[0]
    pos_h = np.hstack([pos, np.ones((N, 1))])  # [x, y, 1]
    img_h = np.linalg.solve(H, pos_h.T)  # H * img_h = world_h  => img_h = H^{-1} * world_h
    x = img_h[0] / img_h[2]
    y = img_h[1] / img_h[2]
    return x, y

try:
    import sys
    sys.path.append(os.path.dirname(__file__))
    import visualization_utils as vis
    to_image_frame = vis.to_image_frame
except Exception:
    to_image_frame = _fallback_to_image_frame
# ============================================================
# Save CI to .csv files per iteration 
# ============================================================
def save_ci_traj_positions_csv(iter_out_dir, iteration_index, rows):
    """
    Save per-location CI for pedestrian trajectories (global coordinates).
    Each row must be a dict with keys: frame, pid, step, x, y, ci
    Output: <iter_out_dir>/ci_traj_positions_<iteration_index:03d>.csv
    """
    outdir = pathlib.Path(iter_out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"ci_traj_positions_{iteration_index:03d}.csv"

    fieldnames = ["frame", "pid", "step", "x", "y", "ci"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return str(out_path)

def project_ctrl_step_to_local_xy(ctrl_step, dt, mode="unicycle"):
    """
    Map a single controller output (one step) to local (dx, dy) during dt.
    mode='cartesian' expects [vx, vy] (body frame), returns (vx*dt, vy*dt)
    mode='unicycle' expects [v, w], returns arc displacement for one step:
        if |w| ~ 0: (v*dt, 0)
        else: ( (v/w) * sin(w*dt), (v/w) * (1 - cos(w*dt)) )
    """
    cx, cy = float(ctrl_step[0]), float(ctrl_step[1])
    if mode == "cartesian":
        return cx * dt, cy * dt
    # unicycle (v, w)
    v, w = cx, cy
    if abs(w) < 1e-8:
        return v * dt, 0.0
    return (v / w) * np.sin(w * dt), (v / w) * (1.0 - np.cos(w * dt))


def save_ci_ctrl_local_csv(iter_out_dir, iteration_index, rows):
    """
    Save controller CI on robot-centered local plane (per-frame & per-step).
    Each row must be a dict with keys: frame, step, x, y, ci
    Output: <iter_out_dir>/ci_ctrl_local_<iteration_index:03d>.csv
    """
    outdir = pathlib.Path(iter_out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"ci_ctrl_local_{iteration_index:03d}.csv"

    fieldnames = ["frame", "step", "x", "y", "ci"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return str(out_path)

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
                   ylim=(-10.0, 2.0),  #(-12.5, 5.5)
                   background_image: Optional[np.ndarray] = None,  
                   background_extent: Optional[Tuple[float, float, float, float]] = None,  # (xmin, xmax, ymin, ymax)
                   background_alpha: Optional[float] = None):
                   
    """
    Draw history / GT future / prediction with static boxes and robot.
    Saves: <outdir>/frame_<frame_idx>.png
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Background image
    if background_image is not None and background_extent is not None:
        alpha = 0.6 if background_alpha is None else background_alpha
        xmin, xmax, ymin, ymax = background_extent
        ax.imshow(background_image, extent=(xmin, xmax, ymin, ymax),
                  alpha=alpha, aspect='auto')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        #ax.set_aspect((xmax - xmin) / (ymax - ymin))
        ax.autoscale(False)
        for artist in ax.get_children():
            try:
                artist.set_clip_on(True)
            except Exception:
                pass

    # Static boxes - gray
    if static_boxes:
        for b in static_boxes:
            if getattr(b, "vertices", None) is None:
                continue
            poly = Polygon(b.vertices, closed=True,
                           facecolor='gray', edgecolor='gray',
                           linewidth=1, zorder=0.1)
            #ax.add_patch(poly)

    # Robot trajectory / current / goal
    px, py = robot_traj_xy
    '''
    if len(px) and len(py):
        ax.plot(px, py, linewidth=2)
    ax.scatter([robot_xy[0]], [robot_xy[1]], marker='o', s=30)
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker='*', s=80)
    '''
    
    # History(8)
    if valid_obs:
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)   # path
                ax.plot(a[-1, 0], a[-1, 1], marker='o', markersize=6, linestyle='None')       

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

    #ax.set_aspect('equal', adjustable='box')
    #ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"frame_{frame_idx:05d}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)
def save_frame_png_spectrum_video(outdir,
                   frame_idx,
                   static_boxes,
                   robot_xy,
                   robot_traj_xy,
                   goal_xy,
                   valid_obs=None,
                   valid_obs_future_true=None,
                   prediction_res=None,
                   r_star=None,
                   steps_to_annotate=(2, 5, 10),   # kept for compat (unused here)
                   annotate_ci=False,              # if True, color prediction lines by CI
                   # --- NEW: spectrum controls for the prediction line ---
                   ci_cmap: str = 'plasma',
                   ci_vmin: Optional[float] = -1.0,
                   ci_vmax: Optional[float] = 1.0,
                   ci_linewidth: float = 1.8,
                   ci_alpha: float = 0.95,
                   ci_colorbar: bool = True,
                   # ------------------------------------------------------
                   ci_decimals=2,                 # kept for compat (unused)
                   ci_fontsize=7,                 # kept for compat (unused)
                   max_ci_annotations_per_step=None,  # kept for compat (unused)
                   xlim=(-2.5, 10.0),
                   ylim=(-10.0, 2.0),
                   background_image: Optional[np.ndarray] = None,  
                   background_extent: Optional[Tuple[float, float, float, float]] = None,
                   background_alpha: Optional[float] = None):

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    norm = mpl.colors.Normalize(vmin=ci_vmin, vmax=ci_vmax, clip=True)
    # Background
    if background_image is not None and background_extent is not None:
        alpha = 0.6 if background_alpha is None else background_alpha
        xmin, xmax, ymin, ymax = background_extent
        ax.imshow(background_image, extent=(xmin, xmax, ymin, ymax),
                  alpha=alpha, aspect='auto')
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal'); ax.autoscale(False)
        for artist in ax.get_children():
            try: artist.set_clip_on(True)
            except Exception: pass

    # Static boxes (disabled as in your code)
    if static_boxes:
        for b in static_boxes:
            if getattr(b, "vertices", None) is None: continue
            poly = Polygon(b.vertices, closed=True,
                           facecolor='gray', edgecolor='gray',
                           linewidth=1, zorder=0.1)
            # ax.add_patch(poly)

    # Robot (kept commented in your code)
    px, py = robot_traj_xy

    # History (8)
    if valid_obs:
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)
                ax.plot(a[-1, 0], a[-1, 1], marker='o', markersize=6, linestyle='None')
    
    # GT future (12)
    if valid_obs_future_true:
        for _, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], linestyle='--', color='black', linewidth=1)
        # Decide normalization
        


        for pid, arr in prediction_res.items():
            pred = np.asarray(arr, dtype=np.float64)
            if not (pred.ndim == 2 and pred.shape[1] >= 2): 
                continue
            p = pred[:12, :2]
            if len(p) < 2 or not np.isfinite(p).all():
                continue

            # Build line segments
            segs = np.stack([p[:-1], p[1:]], axis=1)  # (n-1, 2, 2)

            if annotate_ci and (r_star is not None) and (r_star > 0) and valid_obs_future_true and (pid in valid_obs_future_true):
                gt = np.asarray(valid_obs_future_true[pid], dtype=np.float64)[:len(p), :2]
                L = min(len(p), len(gt))
                if L >= 2:
                    errs = np.linalg.norm(p[:L] - gt[:L], axis=1)
                    cis = (r_star - errs) / r_star
                    cvals = cis[1:L]  # one value per segment
                    lc = LineCollection(segs[:L-1], cmap=ci_cmap, norm=norm, linewidths=ci_linewidth, alpha=ci_alpha)
                    lc.set_array(cvals)
                    ax.add_collection(lc)
                    continue  # done for this pid

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=ci_cmap)
    mappable.set_array([])  # needed for some Matplotlib versions
    cb = plt.colorbar(mappable, ax=ax, pad=0.01)
    cb.set_label('CI Prediction (12)', rotation=90, labelpad=12, va='center')

    legend_elements = [
        Line2D([0], [0], color='navy',  lw=1,   linestyle='-',  label='History (8)'),
        Line2D([0], [0], color='black', lw=1,   linestyle='--', label='GT future (12)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    #ax.set_aspect('equal', adjustable='box')
    #ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"frame_{frame_idx:05d}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)

def save_frame_png_spectrum_robot(outdir,
                   frame_idx,
                   static_boxes,
                   robot_xy,
                   robot_traj_xy,
                   goal_xy,
                   valid_obs=None,
                   valid_obs_future_true=None,
                   prediction_res=None,
                   r_star=None,
                   steps_to_annotate=(2, 5, 10),   # kept for compat (unused here)
                   annotate_ci=False,              # if True, color prediction lines by CI
                   # --- NEW: spectrum controls for the prediction line ---
                   ci_cmap: str = 'plasma',
                   ci_vmin: Optional[float] = -1.0,
                   ci_vmax: Optional[float] = 1.0,
                   ci_linewidth: float = 1.8,
                   ci_alpha: float = 0.95,
                   ci_colorbar: bool = True,
                   # ------------------------------------------------------
                   ci_decimals=2,                 # kept for compat (unused)
                   ci_fontsize=7,                 # kept for compat (unused)
                   max_ci_annotations_per_step=None,  # kept for compat (unused)
                   xlim=(-2.5, 10.0),
                   ylim=(-10.0, 2.0),
                   background_image: Optional[np.ndarray] = None,  
                   background_extent: Optional[Tuple[float, float, float, float]] = None,
                   background_alpha: Optional[float] = None):
    px, py = robot_traj_xy
 

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    if len(px) and len(py):
        ax.plot(px, py, linewidth=2)
    ax.scatter([robot_xy[0]], [robot_xy[1]], marker='o', s=30)
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker='*', s=80)
    norm = mpl.colors.Normalize(vmin=ci_vmin, vmax=ci_vmax, clip=True)
    # Background
    if background_image is not None and background_extent is not None:
        alpha = 0.6 if background_alpha is None else background_alpha
        xmin, xmax, ymin, ymax = background_extent
        ax.imshow(background_image, extent=(xmin, xmax, ymin-2, ymax),
                  alpha=alpha, aspect='auto')
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin-2, ymax)
        ax.set_aspect('equal'); ax.autoscale(False)
        for artist in ax.get_children():
            try: artist.set_clip_on(True)
            except Exception: pass

    # Static boxes (disabled as in your code)
    if static_boxes:
        for b in static_boxes:
            if getattr(b, "vertices", None) is None: continue
            poly = Polygon(b.vertices, closed=True,
                           facecolor='gray', edgecolor='gray',
                           linewidth=1, zorder=0.1)
            # ax.add_patch(poly)

    # Robot (kept commented in your code)
    px, py = robot_traj_xy

    # History (8)
    if valid_obs:
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:, :2]
                ax.plot(a[:, 0], a[:, 1], color='navy', linewidth=1)
                ax.plot(a[-1, 0], a[-1, 1], marker='o', markersize=6, linestyle='None')
    
    # GT future (12)
    if valid_obs_future_true:
        for _, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                a = a[:12, :2]
                ax.plot(a[:, 0], a[:, 1], linestyle='--', color='black', linewidth=1)
        # Decide normalization
        


        for pid, arr in prediction_res.items():
            pred = np.asarray(arr, dtype=np.float64)
            if not (pred.ndim == 2 and pred.shape[1] >= 2): 
                continue
            p = pred[:12, :2]
            if len(p) < 2 or not np.isfinite(p).all():
                continue

            # Build line segments
            segs = np.stack([p[:-1], p[1:]], axis=1)  # (n-1, 2, 2)

            if annotate_ci and (r_star is not None) and (r_star > 0) and valid_obs_future_true and (pid in valid_obs_future_true):
                gt = np.asarray(valid_obs_future_true[pid], dtype=np.float64)[:len(p), :2]
                L = min(len(p), len(gt))
                if L >= 2:
                    errs = np.linalg.norm(p[:L] - gt[:L], axis=1)
                    cis = (r_star - errs) / r_star
                    cvals = cis[1:L]  # one value per segment
                    lc = LineCollection(segs[:L-1], cmap=ci_cmap, norm=norm, linewidths=ci_linewidth, alpha=ci_alpha)
                    lc.set_array(cvals)
                    ax.add_collection(lc)
                    continue  # done for this pid

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=ci_cmap)
    mappable.set_array([])  # needed for some Matplotlib versions
    cb = plt.colorbar(mappable, ax=ax, pad=0.01)
    cb.set_label('CI Prediction (12)', rotation=90, labelpad=12, va='center')

    legend_elements = [
        Line2D([0], [0], color='navy',  lw=1,   linestyle='-',  label='History (8)'),
        Line2D([0], [0], color='black', lw=1,   linestyle='--', label='GT future (12)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    #ax.set_aspect('equal', adjustable='box')
    #ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"frame_{frame_idx:05d}_robot.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)
def save_frame_painted_then_mpl(
    outdir: str,
    frame_idx: int,
    static_boxes: Optional[list]=None,
    robot_xy: Optional[Tuple[float, float]]=None,
    robot_traj_xy: Optional[Tuple[list[float], list[float]]]=None,
    goal_xy: Optional[Tuple[float, float]]=None,
    
    # data
    valid_obs: Optional[Dict]=None,
    valid_obs_future_true: Optional[Dict]=None,
    prediction_res: Optional[Dict]=None,
    # homography
    homography_H: Optional[np.ndarray]=None,                # 3x3
    # CI controls
    annotate_ci: bool=True,
    r_star: Optional[float]=None,
    ci_cmap: str='plasma',
    ci_vmin: float=-1.0,
    ci_vmax: float= 1.0,
    ci_thick_min: int=2,
    ci_thick_max: int=5,
    # draw options
    pred_steps: Optional[int]=None,
    paint_alpha: float=1.0,  # <1.0 = blended overlay
    # background
    background_image: np.ndarray=None,  # BGR or RGB; uint8 preferred
    assume_bgr: bool=True,              # True if your bg/painting path uses cv2 BGR
    # legend/colorbar
    add_legend: bool=True,
    add_colorbar: bool=True,
    cbar_label: str='CI Prediction (12)',
) -> str:
    """Paint trajectories onto the image (cv2), then add legend + colorbar in Matplotlib."""
    assert background_image is not None, "background_image is required."

    img = background_image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    H_img, W_img = img.shape[:2]
    overlay = img.copy()

    # --- homography helper ---
    def apply_h(world_pts: np.ndarray) -> np.ndarray:
        if world_pts is None or len(world_pts) == 0:
            return None
        mask = np.isfinite(world_pts).all(axis=1)
        if not np.any(mask):
            return None
        wp = world_pts[mask]
        xs, ys = to_image_frame(wp, homography_H)
        return np.stack([xs, ys], axis=1)

    # --- color mapping (fixed range) ---
    norm = mpl.colors.Normalize(vmin=float(ci_vmin), vmax=float(ci_vmax), clip=True)
    cmap = cm.get_cmap(ci_cmap)

    def color_from_ci(ci_val: float):
        r, g, b, _ = cmap(norm(ci_val))
        # cv2 uses BGR
        return (int(255*b), int(255*g), int(255*r))

    def thickness_from_ci(ci_val: float) -> int:
        frac = (ci_val - norm.vmin) / (norm.vmax - norm.vmin + 1e-12)
        frac = float(np.clip(frac, 0.0, 1.0))
        return int(round(ci_thick_min + frac * (ci_thick_max - ci_thick_min)))

    # --- draw utilities ---
    def draw_polyline(uv: np.ndarray, color, thick=2):
        pts = np.round(uv).astype(int)
        for i in range(len(pts)-1):
            cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), color, thick, lineType=cv2.LINE_AA)

    # --- HISTORY (navy-ish) ---
    if valid_obs:
        navy = (128, 0, 0)  # BGR-ish dark blue
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                uv = apply_h(a[:, :2])
                if uv is None or len(uv) < 2 or not np.isfinite(uv).all():
                    continue
                draw_polyline(uv, navy, thick=2)
                c = tuple(np.round(uv[-1]).astype(int))
                cv2.circle(overlay, c, radius=3, color=navy, thickness=-1, lineType=cv2.LINE_AA)

    # --- GT FUTURE (black dashed-ish) ---
    if valid_obs_future_true:
        black = (0, 0, 0)
        for _, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                uv = apply_h(a[:12, :2])
                if uv is None or len(uv) < 2 or not np.isfinite(uv).all():
                    continue
                pts = np.round(uv).astype(int)
                for i in range(len(pts)-1):
                    if i % 2 == 0:
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), black, 1, cv2.LINE_AA)

    # --- PREDICTION (CI-colored) ---
    if prediction_res:
        for pid, arr in prediction_res.items():
            p_world = np.asarray(arr, dtype=np.float64)
            if not (p_world.ndim == 2 and p_world.shape[1] >= 2):
                continue
            n_keep = pred_steps if (pred_steps is not None) else 12
            p_world = p_world[:n_keep, :2]
            if len(p_world) < 2 or not np.isfinite(p_world).all():
                continue

            p_uv = apply_h(p_world)
            if p_uv is None or not np.isfinite(p_uv).all():
                continue
            pts = np.round(p_uv).astype(int)

            if annotate_ci and (r_star is not None) and (r_star > 0) and valid_obs_future_true and (pid in valid_obs_future_true):
                gt_world = np.asarray(valid_obs_future_true[pid], dtype=np.float64)[:len(p_world), :2]
                if len(gt_world) >= 2 and np.isfinite(gt_world).all():
                    errs = np.linalg.norm(p_world[:len(gt_world)] - gt_world, axis=1)
                    cis  = (r_star - errs) / r_star
                    cvals = cis[1:]
                    for i in range(len(pts)-1):
                        ci_val = cvals[i] if i < len(cvals) else cvals[-1]
                        color  = color_from_ci(float(ci_val))
                        thick  = thickness_from_ci(float(ci_val))
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), color, thick, cv2.LINE_AA)
                    continue

            # fallback: plain red
            red = (0, 0, 255)
            for i in range(len(pts)-1):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), red, ci_thick_min, cv2.LINE_AA)

    # --- blend overlay to base image ---
    painted = overlay if paint_alpha >= 1.0 else cv2.addWeighted(overlay, paint_alpha, img, 1.0 - paint_alpha, 0.0)

    # ---------- MATPLOTLIB: add legend + colorbar on top of painted image ----------
    # Convert to RGB for Matplotlib if image is BGR
    shown = painted if not assume_bgr else cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)

    fig_dpi = W_img/6
    #fig_w, fig_h = W_img / fig_dpi, H_img / fig_dpi
    fig, ax = plt.subplots(figsize=(6, 6), dpi=fig_dpi)
    #fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=fig_dpi)
    ax.imshow(shown, interpolation='lanczos')   # fill axes
    # Colorbar (inset so image size stays the same)

    # Legend (proxy handles; prediction color shown as mid-CI swatch)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=ci_cmap)
    mappable.set_array([])  # needed for some Matplotlib versions
    cb = plt.colorbar(mappable, ax=ax, pad=0.01)
    cb.set_label('CI Prediction (12)', rotation=90, labelpad=12, va='center')

    legend_elements = [
        Line2D([0], [0], color='navy',  lw=1,   linestyle='-',  label='History (8)'),
        Line2D([0], [0], color='black', lw=1,   linestyle='--', label='GT future (12)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)


    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"frame_{frame_idx:06d}.png")
    # IMPORTANT: don’t use bbox_inches="tight" or extra padding; we want 1:1 with the image
    fig.savefig(out_path, dpi=fig.dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path
def save_frame_mpl_traj(
    outdir: str,
    frame_idx: int,
    static_boxes: Optional[list]=None,
    robot_xy: Optional[Tuple[float, float]]=None,
    robot_traj_xy: Optional[Tuple[list[float], list[float]]]=None,
    goal_xy: Optional[Tuple[float, float]]=None,
    r_star: Optional[float]=None,
    # data
    valid_obs: Optional[Dict]=None,
    valid_obs_future_true: Optional[Dict]=None,
    prediction_res: Optional[Dict]=None,
    # homography
    homography_H: Optional[np.ndarray]=None,                # 3x3
    # CI controls
    annotate_ci: bool=False,
    ci_cmap: str='plasma',
    ci_vmin: float=0.0,
    ci_vmax: float= 1.0,
    ci_thick_min: int=2,
    ci_thick_max: int=5,
    # draw options
    pred_steps: Optional[int]=None,
    paint_alpha: float=1.0,  # <1.0 = blended overlay
    # background
    background_image: np.ndarray=None,  # BGR or RGB; uint8 preferred
    assume_bgr: bool=True,              # True if your bg/painting path uses cv2 BGR
    # legend/colorbar
    add_legend: bool=True,
    add_colorbar: bool=True,
    cbar_label: str='CI Control',
    ci_data: Optional[np.ndarray]=None,  # (N,) values to color robot trajectory
) -> str:
    """Paint trajectories onto the image (cv2), then add legend + colorbar in Matplotlib."""
    import os, cv2, numpy as np, matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    assert background_image is not None, "background_image is required."

    img = background_image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


    H_img, W_img = img.shape[:2]
    overlay = img.copy()

    # --- homography helper ---
    def apply_h(world_pts: np.ndarray) -> np.ndarray:
        if world_pts is None or len(world_pts) == 0:
            return None
        mask = np.isfinite(world_pts).all(axis=1)
        if not np.any(mask):
            return None
        wp = world_pts[mask]
        if homography_H is None:
            # If no H is given, assume world already in pixel frame
            return wp[:, :2]
        # Expected: to_image_frame(wp, H) -> xs, ys
        xs, ys = to_image_frame(wp, homography_H)  # your existing util
        return np.stack([xs, ys], axis=1)

    # --- color mapping (fixed range) ---
    norm = mpl.colors.Normalize(vmin=float(ci_vmin), vmax=float(ci_vmax), clip=True)
    cmap = cm.get_cmap(ci_cmap)

    def color_from_ci(ci_val: float):
        r, g, b, _ = cmap(norm(ci_val))  # Matplotlib returns RGBA in [0,1]
        # cv2 uses BGR with [0..255] ints
        return (int(255*b), int(255*g), int(255*r))  # BGR

    def thickness_from_ci(ci_val: float) -> int:
        frac = (ci_val - norm.vmin) / (norm.vmax - norm.vmin + 1e-12)
        frac = float(np.clip(frac, 0.0, 1.0))
        return int(round(ci_thick_min + frac * (ci_thick_max - ci_thick_min)))

    # --- draw utilities ---
    def draw_polyline(uv: np.ndarray, color, thick=2):
        pts = np.round(uv).astype(int)
        for i in range(len(pts)-1):
            cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), color, thick, lineType=cv2.LINE_AA)

    # --- HISTORY (navy-ish) ---
    if valid_obs:
        navy = (128, 0, 0)  # BGR-ish dark blue
        for _, arr in valid_obs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                uv = apply_h(a[:, :2])
                if uv is None or len(uv) < 2 or not np.isfinite(uv).all():
                    continue
                draw_polyline(uv, navy, thick=2)
                c = tuple(np.round(uv[-1]).astype(int))
                cv2.circle(overlay, c, radius=3, color=navy, thickness=-1, lineType=cv2.LINE_AA)

    # --- GT FUTURE (black dashed-ish) ---
    if valid_obs_future_true:
        black = (0, 0, 0)
        for _, arr in valid_obs_future_true.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                uv = apply_h(a[:12, :2])
                if uv is None or len(uv) < 2 or not np.isfinite(uv).all():
                    continue
                pts = np.round(uv).astype(int)
                for i in range(len(pts)-1):
                    if i % 2 == 0:
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), black, 1, cv2.LINE_AA)
    # --- PREDICTION (CI-colored) ---
    """
    if prediction_res:
        for pid, arr in prediction_res.items():
            p_world = np.asarray(arr, dtype=np.float64)
            if not (p_world.ndim == 2 and p_world.shape[1] >= 2):
                continue
            n_keep = pred_steps if (pred_steps is not None) else 12
            p_world = p_world[:n_keep, :2]
            if len(p_world) < 2 or not np.isfinite(p_world).all():
                continue

            p_uv = apply_h(p_world)
            if p_uv is None or not np.isfinite(p_uv).all():
                continue
            pts = np.round(p_uv).astype(int)

            if annotate_ci and (r_star is not None) and (r_star > 0) and valid_obs_future_true and (pid in valid_obs_future_true):
                gt_world = np.asarray(valid_obs_future_true[pid], dtype=np.float64)[:len(p_world), :2]
                if len(gt_world) >= 2 and np.isfinite(gt_world).all():
                    errs = np.linalg.norm(p_world[:len(gt_world)] - gt_world, axis=1)
                    cis  = (r_star - errs) / r_star
                    cvals = cis[1:]
                    for i in range(len(pts)-1):
                        ci_val = cvals[i] if i < len(cvals) else cvals[-1]
                        color  = color_from_ci(float(ci_val))
                        thick  = thickness_from_ci(float(ci_val))
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), color, thick, cv2.LINE_AA)
                    continue

            # fallback: plain red
            red = (0, 0, 255)
            for i in range(len(pts)-1):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), red, ci_thick_min, cv2.LINE_AA)"""
    # --- PREDICTION (plain red for now) ---
    if prediction_res:
        for pid, arr in prediction_res.items():
            p_world = np.asarray(arr, dtype=np.float64)
            if not (p_world.ndim == 2 and p_world.shape[1] >= 2):
                continue
            n_keep = pred_steps if (pred_steps is not None) else 12
            p_world = p_world[:n_keep, :2]
            if len(p_world) < 2 or not np.isfinite(p_world).all():
                continue

            p_uv = apply_h(p_world)
            if p_uv is None or not np.isfinite(p_uv).all():
                continue
            pts = np.round(p_uv).astype(int)
            red = (0, 0, 255)
            for i in range(len(pts)-1):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), red, ci_thick_min, cv2.LINE_AA)

    # === NEW: ROBOT TRAJECTORY (CI-colored) ===
    if robot_traj_xy is not None:
        px, py = robot_traj_xy
        if px is not None and py is not None and len(px) >= 2 and len(py) >= 2:
            rt_world = np.column_stack([np.asarray(px, float), np.asarray(py, float)])
            rt_uv = apply_h(rt_world)
            if rt_uv is not None and len(rt_uv) >= 2 and np.isfinite(rt_uv).all():
                pts = np.round(rt_uv).astype(int)

                # Determine per-segment CI values
                seg_vals = None
                if ci_data is not None:
                    cvals = np.asarray(ci_data, dtype=float).ravel()
                    if len(cvals) == len(pts):
                        # node-wise -> average to segment-wise
                        seg_vals = 0.5 * (cvals[:-1] + cvals[1:])
                    elif len(cvals) == len(pts) - 1:
                        seg_vals = cvals
                    else:
                        # interpolate to segment count
                        nseg = len(pts) - 1
                        seg_idx = np.linspace(0, max(1, len(cvals) - 1), nseg)
                        base_idx = np.arange(len(cvals))
                        seg_vals = np.interp(seg_idx, base_idx, cvals)

                # Draw with CI-based color/thickness (or fallback color)
                if seg_vals is not None:
                    for i in range(len(pts)-1):
                        ci_val = float(seg_vals[i])
                        color  = color_from_ci(ci_val)
                        thick  = thickness_from_ci(ci_val) if annotate_ci else ci_thick_min
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), color, thick, cv2.LINE_AA)
                else:
                    # Fallback single-color if ci_data missing
                    for i in range(len(pts)-1):
                        cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]),
                                 (0, 255, 0), ci_thick_min, cv2.LINE_AA)  # green

            # Optionally mark current robot & goal positions
            if robot_xy is not None and np.all(np.isfinite(robot_xy)):
                r_uv = apply_h(np.array([robot_xy], float))
                if r_uv is not None:
                    cv2.circle(overlay, tuple(np.round(r_uv[0]).astype(int)),
                               radius=4, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            if goal_xy is not None and np.all(np.isfinite(goal_xy)):
                g_uv = apply_h(np.array([goal_xy], float))
                if g_uv is not None:
                    cv2.drawMarker(overlay, tuple(np.round(g_uv[0]).astype(int)),
                                   color=(0, 255, 255), markerType=cv2.MARKER_STAR,
                                   markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    # --- blend overlay to base image ---
    painted = overlay if paint_alpha >= 1.0 else cv2.addWeighted(overlay, paint_alpha, img, 1.0 - paint_alpha, 0.0)

    # ---------- MATPLOTLIB: add legend + colorbar on top of painted image ----------
    shown = painted if not assume_bgr else cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)

    # keep 1:1 pixel size with a square default; adjust if your image isn't square
    fig_dpi = max(1, int(W_img / 6))  # keeps width ~W_img px for figsize=(6,6)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=fig_dpi)
    ax.imshow(shown, interpolation='lanczos')
    ax.set_axis_off()

    # Colorbar sharing the same norm & cmap
    if add_colorbar:
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cb = plt.colorbar(mappable, ax=ax, pad=0.01)
        cb.set_label(cbar_label, rotation=90, labelpad=12, va='center')

    # Legend with proxy handles
    if add_legend:
        proxy_ci_color = cmap(norm(0.5*(ci_vmin+ci_vmax)))
        legend_elements = [
            Line2D([0], [0], color='navy',  lw=2,   linestyle='-',  label='History (8)'),
            Line2D([0], [0], color='black', lw=1,   linestyle='--', label='GT future (12)'),
            Line2D([0], [0], color=proxy_ci_color, lw=3, label='Prediction (12)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.9)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"frame_{frame_idx:06d}.png")
    fig.savefig(out_path, dpi=fig.dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return out_path
