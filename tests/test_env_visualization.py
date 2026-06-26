import numpy as np
import argparse
from canvas.datasets import RegisteredDatasets
from canvas.envs.env import Environment
import os


def test_env_visualization(frames_dir=None):

    dataset = RegisteredDatasets['zara1']
    env = Environment(
        dataset=dataset,
        init_robot_state={'position_x': 12., 'position_y': 5., 'orientation_z': np.pi},
        goal_pos=np.array([3., 6.]),
        t_begin=40,
        t_end=200,
        history_len=8,
        prediction_horizon=12,
        path_to_frames=frames_dir,
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
    parser = argparse.ArgumentParser()
    path = os.path.abspath(__file__)
    parts = path.split(os.sep)
    canvas_idx = parts.index("CANVAS")
    canvas_root = os.sep.join(parts[:canvas_idx + 1])
    target_path = os.path.join(canvas_root, "assets", "frames")
    parser.add_argument("--frames",nargs="?", help="frames dirpath", type=str, default=target_path)
    args = parser.parse_args()
    test_env_visualization(frames_dir=args.frames)