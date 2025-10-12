import numpy as np

from canvas.datasets import RegisteredDatasets
from canvas.envs.env_new import Environment


def test_env_visualization():

    dataset = RegisteredDatasets['zara1']
    env = Environment(
        dataset=dataset,
        init_robot_state={'position_x': 12., 'position_y': 5., 'orientation_z': np.pi},
        goal_pos=np.array([3., 6.]),
        t_begin=40,
        t_end=200,
        history_len=8,
        prediction_horizon=12,
        #path_to_frames='/media/sju5379/F6340D35340CF9FF/euped_assets/frames',
        path_to_frames='/home/core/Documents/CANVAS/canvas/assets/final/frames',
        path_to_save='./viz_example'
    )

    env.reset()
    truncated = False

    while not truncated:
        action = .5 * np.random.rand(2)
        env.step(action)

        colors = np.linspace(0., 1., 200)

        env.render(c=colors)

    return


if __name__ == '__main__':
    test_env_visualization()