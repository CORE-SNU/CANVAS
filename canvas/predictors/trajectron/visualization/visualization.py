from utils import prediction_output_to_trajectories
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import pickle
import os
import sys
import matplotlib.image as mpimg


sys.path.append(os.path.dirname(os.path.realpath(__file__)))


import matplotlib.image
from cp.adaptive_cp import AdaptiveConformalPredictionModule
from controllers.grid_solver import GridMPC
from PIL import Image
from scipy import ndimage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from visualization_utils_custom import visualize_cp_result, visualize_tracking_result, visualize_prediction_result, visualize_controller_info



def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=1.0,
                      line_width=1.,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    # cmap = ['k', 'b', 'y', 'g', 'r']
    cmap = ['k', '#88729a', 'y', 'g', 'r']

    h_labeled = False
    p_labeled = False
    f_labeled = False

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]
        if np.isnan(history[-1]).any():
            continue

        history_label = 'history' if not h_labeled else None
        ax.plot(history[:, 0], history[:, 1], 'k--', label=history_label)
        h_labeled = True
        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            pred_label = 'Trajectron++' if not p_labeled else None
            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha, label=pred_label)
            p_labeled = True

            future_label = 'future' if not f_labeled else None
            ax.plot(future[:, 0],
                    future[:, 1],
                    'k',
                    linewidth=3,
                    # path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()],
                    zorder=500, label=future_label)
            f_labeled = True


            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='#88729a',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=600)
            ax.add_artist(circle)

    # ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, *kwargs)




def visualize_predictions(
                          predictions,
                          dt,
                          max_hl,
                          ph,
                          path,
                          robot_node=None,
                          baselines=None,
                          baseline_colors=None,
                          map=None,
                          **kwargs):

    os.makedirs(path, exist_ok=True)

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    # with open(os.path.join(path, 'history.pkl'), 'wb') as f:
    #     pic


    # assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return

    metric = {}
    for baseline_name, baseline_model in baselines.items():
        metric[baseline_name] = {'ADE': [], 'FDE': []}

    for ts_key in prediction_dict.keys():
        # plt.clf(), plt.cla()
        # fig, ax = plt.subplots()

        xmin, xmax = -10., 10.
        ymin, ymax = -10., 10.

        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)

        p_dict = prediction_dict[ts_key]
        h_dict = histories_dict[ts_key]
        f_dict = futures_dict[ts_key]


        # if map is not None:
            # ax.imshow(map.as_image(), origin='lower', alpha=0.5)
        # plot_trajectories(ax, p_dict, h_dict, f_dict, *kwargs)

        if baselines is not None:
            for baseline_name, baseline_model in baselines.items():
                p_baseline_dict = baseline_model(h_dict)
                color = baseline_colors[baseline_name]
                labeled = False


                for track_id, pos_xy in p_baseline_dict.items():

                    pos_xy_true = f_dict[track_id]

                    displacement_err = np.sum((pos_xy - pos_xy_true) ** 2, axis=-1) ** .5
                    ade = np.mean(displacement_err)
                    fde = displacement_err[-1]

                    metric[baseline_name]['ADE'].append(ade)
                    metric[baseline_name]['FDE'].append(fde)

                    label = baseline_name if not labeled else None
                    # ax.plot(pos_xy[:, 0], pos_xy[:, 1], color=color, label=label)
                    labeled = True

        # ax.legend()
        # fig.savefig(os.path.join(path, '{:03d}.png'.format(ts_key)))
        # plt.close()

    for baseline_name, baseline_model in baselines.items():
        metric[baseline_name] = {'ADE': np.mean(metric[baseline_name]['ADE']), 'FDE': np.mean(metric[baseline_name]['FDE'])}
    return metric




def run_mpc(
                          predictions,
        pos_x_mean, pos_y_mean,
                          dt,
                          max_hl,
                          ph,
                          path,
                          robot_node=None,
                          baselines=None,
                          baseline_colors=None,
                          map=None,
                          **kwargs):

    os.makedirs(path, exist_ok=True)

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    # with open(os.path.join(path, 'history.pkl'), 'wb') as f:
    #     pic

    prediction_len = 12

    max_interval_lengths = 2. * dt * np.arange(1, prediction_len + 1)
    offline_calibration_set = {i: [] for i in range(prediction_len)}

    cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.2,
                                                  step_size=0.05,
                                                  n_scores=prediction_len,
                                                  max_interval_lengths=max_interval_lengths,
                                                  sample_size=20,
                                                  offline_calibration_set=offline_calibration_set
                                                  )

    controller = GridMPC(n_steps=prediction_len, dt=dt)

    init_robot_pose = np.array([1., 6., 0.])

    goal_x, goal_y = 14., 5.
    goal = np.array([goal_x, goal_y])

    velocity = np.array([0., 0., ])

    # assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return

    metric = {}
    for baseline_name, baseline_model in baselines.items():
        metric[baseline_name] = {'ADE': [], 'FDE': []}

    position_x, position_y, orientation_z = init_robot_pose
    linear_x, angular_z = velocity

    robot_img = Image.open(os.path.join(os.path.dirname(__file__), "real_robot.png"))
    video_dir = './videos'
    os.makedirs(video_dir, exist_ok=True)

    shifted_mean = np.array([pos_x_mean, pos_y_mean])


    for ts_key in prediction_dict.keys():
        # plt.clf(), plt.cla()
        # fig, ax = plt.subplots()

        xmin, xmax = -10., 10.
        ymin, ymax = -10., 10.

        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)

        p_dict = prediction_dict[ts_key]
        h_dict = histories_dict[ts_key]
        f_dict = futures_dict[ts_key]

        p_dict = {track_id: p.squeeze() + shifted_mean for track_id, p in p_dict.items()}
        h_dict = {track_id: h.squeeze() + shifted_mean for track_id, h in h_dict.items()}
        f_dict = {track_id: f.squeeze() + shifted_mean for track_id, f in f_dict.items()}


        confidence_intervals = cp_module.update(h_dict, p_dict)

        velocity, info = controller(pos_x=position_x,
                                    pos_y=position_y,
                                    orientation_z=orientation_z,
                                    linear_x=linear_x,
                                    angular_z=angular_z,
                                    boxes=[],  # TODO
                                    predictions=p_dict,
                                    confidence_intervals=confidence_intervals,
                                    goal=goal
                                    )

        if not info['feasible']:
            # print('infeasible')
            velocity = np.array([0., 0.])
            print('linear_x={} / angular_z={} (infeasible)'.format(*velocity))
        else:
            # velocity = np.array([1., 0.])
            print('linear_x={} / angular_z={} (feasible)'.format(*velocity))

        linear_x, angular_z = velocity

        position_x += dt * linear_x * np.cos(orientation_z)
        position_y += dt * linear_x * np.sin(orientation_z)
        orientation_z += dt * angular_z

        plt.clf(), plt.cla()
        fig, ax = plt.subplots()

        xmin, xmax = -0.02104651, 15.13244069
        ymin, ymax = -0.2386598199999988, 12.3864436

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        img_rotated = OffsetImage(Image.fromarray(ndimage.rotate(
            robot_img, orientation_z / np.pi * 180. - 90.)), zoom=0.05)

        visualize_tracking_result(h_dict, ax)
        visualize_controller_info(info, ax)
        visualize_prediction_result(p_dict, ax, color='blue', linestyle='dashed', label='Trajectron++')
        visualize_prediction_result(f_dict, ax, color='black', linestyle='solid', label='true')
        visualize_cp_result(confidence_intervals, p_dict, [1, 4, 7, 10], ax)

        background = mpimg.imread('zara1.jpg')
        plt.imshow(background, extent=(xmin, xmax, ymin, ymax), alpha=0.4)

        ax.add_artist(AnnotationBbox(img_rotated, (position_x, position_y), frameon=False))
        # ax.scatter([position_x], [position_y], color='tab:blue', marker='o', s=80, label='robot')
        dx = 0.8 * np.cos(orientation_z)
        dy = 0.8 * np.sin(orientation_z)
        ax.arrow(position_x, position_y, dx, dy, head_width=0.05, head_length=0.1, fc='black', ec='black',
                 zorder=120)

        ax.scatter([goal_x], [goal_y], color='tab:red', marker='s', s=80, label='goal', zorder=500)

        ax.legend(loc='upper right')

        fig.savefig(os.path.join(video_dir, '{:03d}.png'.format(ts_key)))
        plt.close()

        # if map is not None:
            # ax.imshow(map.as_image(), origin='lower', alpha=0.5)
        # plot_trajectories(ax, p_dict, h_dict, f_dict, *kwargs)

        if baselines is not None:
            for baseline_name, baseline_model in baselines.items():
                p_baseline_dict = baseline_model(h_dict)
                color = baseline_colors[baseline_name]
                labeled = False

                for track_id, pos_xy in p_baseline_dict.items():

                    pos_xy_true = f_dict[track_id]

                    displacement_err = np.sum((pos_xy - pos_xy_true) ** 2, axis=-1) ** .5
                    ade = np.mean(displacement_err)
                    fde = displacement_err[-1]

                    metric[baseline_name]['ADE'].append(ade)
                    metric[baseline_name]['FDE'].append(fde)

                    label = baseline_name if not labeled else None
                    # ax.plot(pos_xy[:, 0], pos_xy[:, 1], color=color, label=label)
                    labeled = True

        # ax.legend()
        # fig.savefig(os.path.join(path, '{:03d}.png'.format(ts_key)))
        # plt.close()

    for baseline_name, baseline_model in baselines.items():
        metric[baseline_name] = {'ADE': np.mean(metric[baseline_name]['ADE']), 'FDE': np.mean(metric[baseline_name]['FDE'])}
    return metric



def visualize_distribution(ax,
                           prediction_distribution_dict,
                           map=None,
                           pi_threshold=0.05,
                           **kwargs):
    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()

        for timestep in range(means.shape[0]):
            for z_val in range(means.shape[1]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]
                pi = pis[timestep, z_val]

                if pi < pi_threshold:
                    continue

                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue' if node.type.name == 'VEHICLE' else 'orange')
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(pi/10)
                ax.add_artist(ell)
