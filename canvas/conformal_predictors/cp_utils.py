import numpy as np
from typing import Dict, Any


class CircularQueue:
    def __init__(self, maxlen: int, dim: int):

        assert maxlen > 0
        assert dim > 0

        self._maxlen = maxlen
        self._buffer = np.zeros((maxlen, dim))
        self._dim = dim

        self.head = None
        self.tail = None
        self.ticks = None

    def initialize(self, idx: int):
        self.ticks = 0   # for checking if the queue becomes full once
        self.tail = idx
        self.head = self.tail

    def append(self, data):
        self._buffer[self.tail % self._maxlen] = data
        self.tail += 1
        return

    def _pop(self):
        assert self.size > 0
        self.head += 1
        return

    def tick(self):
        # must be called every time step regardless of data insertion
        self.ticks += 1
        if self.ticks > self._maxlen:
            # If the queue reaches its size limit, then start to remove the oldest items.
            self._pop()

    @property
    def size(self):
        return self.tail - self.head

    def snapshot(self) -> np.ndarray:
        if self.size > 0:
            h = self.head % self._maxlen
            t = self.tail % self._maxlen
            if h < t:
                return np.copy(self._buffer[h: t])
            else:
                return np.concatenate((self._buffer[h:], self._buffer[:t]), axis=0)
        else:
            return np.zeros((0, self._dim))


class HistoryBuffer:
    """
    A data structure that stores the snapshot of the scene during last n steps
    """
    def __init__(self, history_length: int):
        self._history_len = history_length
        self._t_global = 0                      # global time step; lies in Z/nZ (for bounded storage)
        self._queue_dict: Dict[Any, CircularQueue] = {}

    def update(self, o: Dict[Any, np.ndarray]):
        """
        obs. as a dict. whose values are numpy arrays of shape (2,)
        """
        # TODO: An unbounded t_global may cause an issue (for long simulation time)
        # should make it updated in Z/nZ, but it will complicate the logic...
        for key, q in list(self._queue_dict.items()):
            if key in o:
                # append the observation to the existing queue
                data: np.ndarray = o[key]
                q.append(data=data)
                q.tick()
            else:
                q.tick()
                if q.size == 0:             # becomes empty
                    del self._queue_dict[key]

        for key, h in o.items():
            if key not in self._queue_dict:         # new key
                q = CircularQueue(maxlen=self._history_len, dim=2)
                q.initialize(idx=self._t_global)
                data: np.ndarray = h
                q.append(data=data)
                q.tick()
                self._queue_dict[key] = q

        self._t_global += 1             # update the global counter

    def query(self, keys):
        res = {}
        for k in keys:
            if k in self._queue_dict:
                q = self._queue_dict[k]
                assert q.head - self._t_global + self._history_len == 0
                res[k] = q.snapshot()
        return res

    def __repr__(self):
        """
        just for debugging
        """
        res = 't={} \n'.format(self._t_global)
        lines = []
        for k, q in self._queue_dict.items():
            h = q.head - self._t_global + self._history_len
            t = q.tail - self._t_global + self._history_len
            line = 'key {}: ' .format(k) + h * 'x' + (t - h) * 'o' + (self._history_len - t) * 'x'
            lines.append(line)
        return res + '\n'.join(lines)
