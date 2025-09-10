import os
import yaml
import pathlib
import numpy as np
import cv2
from matplotlib.patches import Polygon, Rectangle, Circle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog



def visualize_tracking_result(tracking_result, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = itertools.cycle(prop_cycle.by_key()['color'])
    colors = prop_cycle.by_key()['color']
    n_colors = len(colors)

    for obj_id, traj in tracking_result.items():
        traj_np = np.array(traj)
        # color = colors[obj_id % n_colors]
        visualize_trajectory(traj_np, ax, color='#88729a')
        center = traj_np[-1]
        radius = 0.3
        circ = Circle(center, radius, facecolor='#88729a', edgecolor='tab:gray', zorder=90)
        ax.add_patch(circ)

        ax.text(center[0], center[1], '{}'.format(obj_id), fontsize=8, zorder=100)

    return


def visualize_trajectory(trajectory, ax, color='black'):
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, zorder=60)


def visualize_prediction_result(prediction_result, ax, color='k', linestyle='solid', label=None):
    labeled = False
    for obj_id, t in prediction_result.items():
        if t is not None:
            if not labeled:
                ax.plot(t[:, 0], t[:, 1], zorder=80, linewidth=2, color=color, label=label, linestyle=linestyle)
                labeled = True
            else:
                ax.plot(t[:, 0], t[:, 1], zorder=80, linewidth=2, color=color, linestyle=linestyle)


def visualize_cp_result(confidence_intervals, prediction_result, selected_steps, ax):
    n_predictions = confidence_intervals.size
    n_selected = len(selected_steps)
    max_transparency = 0.6
    min_transparency = 0.3
    transparency_diff = max_transparency - min_transparency
    for obj_id, t in prediction_result.items():
        count = 0
        for i in selected_steps:
            center = t[i]
            radius = confidence_intervals[i]
            transparency = max_transparency - transparency_diff * count / (n_selected - 1)
            circ = Circle(center, radius=radius, color='tab:gray', alpha=transparency, zorder=30)
            count += 1
            ax.add_patch(circ)
    return


def visualize_controller_info(info, ax):
    if info['feasible']:
        paths = info['candidate_paths']
        safe_paths = info['safe_paths']
        final_path = info['final_path']
        '''
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], color='tab:gray', zorder=60, alpha=0.1)

        for sp in safe_paths:
            ax.plot(sp[:, 0], sp[:, 1], color='yellow', zorder=70, alpha=0.2)
        '''

        ax.plot(final_path[:-1, 0], final_path[:-1, 1], color='tab:cyan', zorder=80)
    return
