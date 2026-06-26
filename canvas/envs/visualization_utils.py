import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon
from scipy import ndimage
from PIL import Image
import torch
import warnings
from matplotlib.collections import LineCollection

from canvas.envs.env_utils import Rectangle

def draw_robot(
        ax,
        H,
        position_x,
        position_y,
        orientation_z,
        robot_img
):

    dx = 0.001 * np.cos(orientation_z)
    dy = 0.001 * np.sin(orientation_z)
    # position of the infinitesimal displacement vector's endpoint (w.r.t. the world frame)
    x_next, y_next = position_x + dx, position_y + dy

    # projective transformation of the vector
    pos_img_x, pos_img_y = to_image_frame(np.array([[position_x, position_y]]), H)
    pos_img_x, pos_img_y = pos_img_x.item(), pos_img_y.item()
    pos_img_x_next, pos_img_y_next = to_image_frame(np.array([[x_next, y_next]]), H)
    pos_img_x_next, pos_img_y_next = pos_img_x_next.item(), pos_img_y_next.item()

    # vector repr. w.r.t. the image frame
    dx_img = pos_img_x_next - pos_img_x
    dy_img = pos_img_y_next - pos_img_y

    orientation_z_img = np.arctan2(dy_img, dx_img)
    # orientation_z_img = orientation_z_img - np.pi / 2.

    # robot figure
    img_rotated = OffsetImage(Image.fromarray(ndimage.rotate(
        robot_img, np.rad2deg(-orientation_z_img))), zoom=0.05, zorder=80)

    ax.add_artist(AnnotationBbox(img_rotated, (pos_img_x, pos_img_y), frameon=False))


def to_homogeneous_coordinates(pos):
    """
    [x, y] to [x: y: 1]
    :param pos: numpy array of shape (N, 2) containing the world coordinates of N points
    :return: numpy array of shape (N, 3) where the last column is filled with 1
             contains the homogeneous coordinates of N points in RP^2
    """
    #z = np.ones((pos.shape[0], 1))      
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy() # added due to torch tensor issue with hstack

    z = np.ones(
        (pos.shape[0], 1),
        dtype=pos.dtype,
    )# N-dimensional column vector: (N, 1)

    return np.hstack((pos, z))


def to_image_frame(pos, H):
    """
    :param pos: numpy array of shape (N, 2) containing the world coordinates of N points
    :param H: (3x3)-homography matrix that transforms image coordinates to their world coordinates
    :return: two array of shape (N,), each containing x positions & y positions
    """
    pos_h = to_homogeneous_coordinates(pos)
    pos_tf = np.linalg.solve(H, pos_h.T)     # array of shape (3, N) containing [x: y: z]'s
    pos_x_tf, pos_y_tf, pos_z_tf = pos_tf[0], pos_tf[1], pos_tf[2]      # [x: y: z] -> [x/z, y/z]
    return pos_x_tf / pos_z_tf, pos_y_tf / pos_z_tf


def visualize_trajectory(trajectory, H, ax, c=None, offset=0., **kwargs):
    """
    trajectory: numpy array of shape (N, 2)
    H: homography matrix
    c: numpy array of shape (N,) or None
    """
    # traj_np = np.array(trajectory)
    traj_image_x, traj_image_y = to_image_frame(trajectory + offset, H)
    if c is None:
        ax.plot(traj_image_x, traj_image_y, **kwargs)
    else:
        assert len(c) >= len(trajectory), (
    f"len(c)={len(c)} is smaller than len(trajectory)={len(trajectory)} "
    f"(c.shape={np.shape(c)}, trajectory.shape={np.shape(trajectory)})"
)
        colored_line(traj_image_x, traj_image_y, c[-len(trajectory):], ax, **kwargs)
    # TODO: color overlay

def add_arrow(x, y, x_next, y_next, H, ax, arrowprops):
    data = np.array([[x, y], [x_next, y_next]])
    image_x, image_y = to_image_frame(data, H)
    x, x_next = image_x
    y, y_next = image_y

    ax.annotate(text='', xy=(x_next, y_next), xytext=(x, y), arrowprops=arrowprops)


def visualize_point(point, H, ax, color, marker, s, label, zorder, linewidth):
    assert point.ndim == 1
    point = np.expand_dims(point, axis=0)
    x, y = to_image_frame(point, H)
    x, y = x.item(), y.item()
    ax.scatter(x, y, color=color, marker=marker, s=s, label=label, zorder=zorder, linewidth=linewidth)



def colored_line(x, y, c, ax, **lc_kwargs):
    """
    See: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt", "cmap": "plasma"}
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    default_kwargs["norm"] = norm
    # added to make sure the color map is the same for all plots
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)




def visualize_points(points, H, ax, color, label, alpha=1.0, s=10):
    assert points.ndim == 2
    xs, ys = to_image_frame(points, H)
    ax.scatter(xs, ys, color=color, alpha=alpha, s=s, label=label)


def project_rectangle_to_image(rectangle: Rectangle, H, color, alpha=1., zorder=0):
    # polygonal approximation of the projected conic
    vertices = rectangle.to_vertices()

    vertices_x, vertices_y = to_image_frame(vertices, H)

    quadrilateral = Polygon(xy=np.stack((vertices_x, vertices_y), axis=1),
                   closed=True, ec=color, fc=color, alpha=alpha, zorder=zorder)
    return quadrilateral



def project_circle_to_image(center, radius, H, color, alpha=1., zorder=0):
    # polygonal approximation of the projected conic
    num_pts = 100
    ths = np.linspace(-np.pi, np.pi, num=num_pts)
    c, s = np.cos(ths), np.sin(ths)
    vertices = center + radius * np.stack((c, s), axis=1)

    vertices_x, vertices_y = to_image_frame(vertices, H)

    ellipse = Polygon(xy=np.stack((vertices_x, vertices_y), axis=1),
                   closed=True, ec=color, fc=color, alpha=alpha, zorder=zorder)
    return ellipse



def visualize_tracking_result(tracking_result, H, ax, color):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = itertools.cycle(prop_cycle.by_key()['color'])
    colors = prop_cycle.by_key()['color']
    n_colors = len(colors)

    for node, traj in tracking_result.items():
        traj_np = np.array(traj)

        traj_image_x, traj_image_y = to_image_frame(traj_np, H)

        # color = colors[obj_id % n_colors]
        # ax.plot(traj_image_x, traj_image_y, color=color, zorder=60)
        radius = 0.2        # in meter
        center = traj_np[-1]
        ax.text(traj_image_x[-1]-8, traj_image_y[-1]+4, '{}'.format(node), fontsize=7, zorder=100)
        ellipse = project_circle_to_image(center, radius, H, color=color, zorder=100)
        # circ = Circle(center, radius, facecolor='#88729a', edgecolor='tab:gray', zorder=90)
        # ax.add_patch(circ)
        ax.add_patch(ellipse)
    return


def add_arrows(xs, ys, n_skips, ax, color):
    n_steps = xs.size
    for i in range(0, n_steps, n_skips):
        if i +1 < n_steps:
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            ax.arrow(x=xs[i], y=ys[i], dx=dx, dy=dy, color=color, head_width=8, head_length=8)


def visualize_prediction_result(prediction_result, H, ax, color='k', linestyle='solid', label=None):
    """
    assuming the prediction result is deterministic
    """
    labeled = False
    for node, pred in prediction_result.items():
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        pred_x, pred_y = to_image_frame(pred, H)
        if not labeled:
            ax.plot(pred_x, pred_y, zorder=80, linewidth=2, color=color, label=label, linestyle=linestyle)
            labeled = True
        else:
            ax.plot(pred_x, pred_y, zorder=80, linewidth=2, color=color, linestyle=linestyle)

        add_arrows(pred_x, pred_y, n_skips=2, ax=ax, color=color)


def visualize_cp_result(confidence_intervals, prediction_result, selected_steps, H, ax):
    n_selected = len(selected_steps)
    max_transparency = 0.95
    min_transparency = 0.8
    transparency_diff = max_transparency - min_transparency
    for obj_id, t in prediction_result.items():
        # print(obj_id)
        # if obj_id in ['PEDESTRIAN/37', 'PEDESTRIAN/41']:
        count = 0
        for i in selected_steps:
            center = t[i]
            # print(center.shape)
            radius = confidence_intervals[i]
            transparency = max_transparency - transparency_diff * count / (n_selected - 1)
            ellipse = project_circle_to_image(center, radius, H, color='black', alpha=transparency, zorder=-1)
            ax.add_patch(ellipse)
    return


def visualize_controller_info(info, H, ax, color='none', feasible_set=False, optimal_sol=True):
    if info['feasible']:
        # paths = info['candidate_paths']
        safe_paths = info['safe_paths']
        final_path = info['final_path']
        '''
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], color='tab:gray', zorder=60, alpha=0.1)
        '''
        if feasible_set:
            for sp in safe_paths:
                safe_path_x, safe_path_y = to_image_frame(sp, H)
                ax.plot(safe_path_x, safe_path_y, color=color, zorder=10)
        if optimal_sol:
            final_path_x, final_path_y = to_image_frame(final_path, H)
            ax.plot(final_path_x, final_path_y, color='tab:cyan', zorder=20)
    return

'''
def render(
        dataset,
        ts_key,
        ts_begin,
        position_x,
        position_y,
        orientation_z,
        robot_img,
        goal_pos,
        info,
        h_dict,
        f_dict,
        p_dict,
        video_dir,
        asset_dir,
        intervals
):

    plt.clf(), plt.cla()
    fig, ax = plt.subplots()
    image = cv2.imread(os.path.join(asset_dir, 'frames', dataset, '{}.png'.format(ts_begin+ts_key)))
    ax.imshow(image, cmap='gray', alpha=0.6)
    h, w, _ = image.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    # load the homography matrix of the dataset
    homography_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets/homographies', dataset + '.txt')
    H = np.loadtxt(homography_path, dtype=float)

    # mark the goal position
    visualize_point(goal_pos, H, ax, color='tab:red', marker='s', s=80, label='goal', zorder=10)

    visualize_controller_info(info, H, ax, color='#9dcda8', feasible_set=False, optimal_sol=True)

    # visualize the robot state
    draw_robot(ax, H, position_x, position_y, orientation_z, robot_img)
    visualize_tracking_result(h_dict, H, ax, color='#ffd300')

    # prediction result
    visualize_prediction_result(p_dict, H, ax, color='#e6a8d7', linestyle='dashed', label='prediction')

    # ground truth
    visualize_prediction_result(f_dict, H, ax, color='#ffd300', linestyle='solid', label='future')

    # ACP confidence set
    if intervals is not None:
        # if ACP-MPC is used
        selected_steps = [1, 3, 5, 7, 9, 11]
        visualize_cp_result(intervals, p_dict, selected_steps, H, ax)

    ax.legend()
    fig.savefig(os.path.join(video_dir, '{:03d}.png'.format(ts_key)), bbox_inches ='tight', pad_inches=0)
    plt.close()
    # print('results visualized at {}'.format(video_dir))
    return
'''