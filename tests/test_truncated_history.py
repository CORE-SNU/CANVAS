import numpy as np
from canvas.conformal_predictors.cp_utils import CircularQueue, HistoryBuffer


def test_circular_queue():
    q = CircularQueue(maxlen=5, dim=2)
    q.initialize(idx=2)
    data_stream = np.arange(16).reshape(8, 2)

    for d in data_stream:
        q.append(data=d)
        q.tick()
        print(q.snapshot())

    for _ in range(5):
        q.tick()
        print(q.snapshot())

    return


def test_truncated_history():
    active_agents = [
        [-1, 1],
        [-1, 1, 2],
        [1, 2],
        [0, 1, 2],
        [0, 2],
        [0, 2, 3],
        [2, 3],
        [2, 3]
    ]

    history = HistoryBuffer(history_length=6)

    for agents in active_agents:
        o = {a: np.random.rand(2) for a in agents}
        history.update(o)

    print(history)
    query_res = history.query(keys=[-1, 1, 2])
    print(query_res)


if __name__ == '__main__':
    # test_circular_queue()
    test_truncated_history()