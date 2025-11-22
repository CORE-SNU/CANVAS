import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from canvas.envs.env import Environment
from canvas.envs.visualization_utils import project_rectangle_to_image
from canvas.datasets import RegisteredDatasets

def main():

    def state_dict_from_vec(v):
        return {'position_x': v[0], 'position_y': v[1], 'orientation_z': v[2]}
    scenario_config = {
        'init_robot_state': state_dict_from_vec(np.array([0., 0., 0.])), 'goal_pos': np.array([6., -5.]),
                     't_begin': 100, 't_end': 250
    }

    env = Environment(
        dataset=RegisteredDatasets['snu-asri'],
        **scenario_config,
        history_len=8,
        prediction_horizon=16,
        path_to_frames='',
        # directory from which the parsed frames are loaded
        path_to_save='..'  # directory to save the visualization result
    )

    tau = 1e-1
    xlim = env.geometry.xlim
    ylim = env.geometry.ylim
    xs = np.linspace(xlim[0], xlim[1], num=200)
    ys = np.linspace(ylim[0], ylim[1], num=200)
    xs, ys = np.meshgrid(xs, ys)
    xs_f = np.reshape(xs, (-1,))
    ys_f = np.reshape(ys, (-1,))
    ps = np.stack([xs_f, ys_f], axis=1)
    zs_list = []
    r_robot = .4
    for o in env.geometry:
        A, b = o.to_halfspaces()
        m = A.shape[0]


        h_neg = A @ ps.T - b[..., None]
        zs = tau * scipy.special.logsumexp(h_neg / tau, axis=0) - tau * np.log(m)
        zs = zs.reshape(200, 200)
        zs_list.append(zs)


    zs = np.min(zs_list, axis=0)
    z_min, z_max = zs.min(), zs.max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xs, ys, zs, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.contour(xs, ys, zs, levels=[r_robot], colors='black', linestyles='--', linewidths=2)

    ax.set_title('signed distancce field (smoothed)')
    ax.scatter([0.], [0.], s=80, marker='x', color='yellow')
    ax.axis([xs.min(), xs.max(), ys.min(), ys.max()])

    patch = matplotlib.patches.Patch(color='black', label=f'sd={r_robot}')
    ax.legend(handles=[patch])
    for obstacle in env.geometry:
        projected = project_rectangle_to_image(rectangle=obstacle, H=env.homography, color='tab:red', alpha=0.2, zorder=1)
        ax.add_patch(projected)

    fig.colorbar(c, ax=ax)
    fig.savefig('signed_distance.png')

    return

if __name__ == '__main__':
    main()