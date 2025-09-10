import numpy as np
from scipy.signal import savgol_filter

def identity(x):
    return x


class KoopmanModel:
    def __init__(self,
                 n_objects=1,
                 state_dim=3,
                 observation=identity,
                 reconstruction=identity,
                 default_n_steps=1,
                 history_len=3,
                 pad_len=10
                 ):
        # TODO: modelling multiple objects interacting with each other
        self._n_objects = n_objects
        self._state_dim = state_dim

        self._to_observables = observation
        self._to_states = reconstruction
        # Koopman matrix
        # Initialize K to the identity matrix before data is observed
        self._K = np.eye(state_dim)

        self._history_len = history_len
        self._pad_len = pad_len

        # TODO: ring buffer implementation for saving memory
        self._xs = []
        self._xs_filtered = None      # for the filtered trajectory

        # default prediction length
        self._n_steps = default_n_steps

        # caching the powers of the Koopman matrix
        self._Ks = None     # [K^T, (K^2)^T, ..., (K^N)^T]^T where N: prediction length

        self._cache_Ks(n_steps=self._n_steps)

    def update(self, x):
        if len(self._xs) < self._pad_len:
            # pad the initial segment of the trajectory with the initial state
            for _ in range(self._pad_len):
                self._append_history(x)
        else:
            self._append_history(x)

        xs = self._xs_filtered[-self._history_len-1:-1]
        xs_next = self._xs_filtered[-self._history_len:]
        self._fit(xs, xs_next)

    def _append_history(self, x):
        self._xs.append(x)
        xs = np.array(self._xs)
        # filter the trajectory as a cubic polynomial
        if xs.shape[0] >= self._pad_len:
            self._xs_filtered = savgol_filter(xs, window_length=5, polyorder=3, axis=0)

            if xs.shape[0] >= self._pad_len + 2:
                # recalculate the heading angle after filtering
                p1 = self._xs_filtered[:, 0]
                p2 = self._xs_filtered[:, 1]
                # first 5 data are duplicates
                d1 = p1[self._pad_len+1:] - p1[self._pad_len:-1]
                d2 = p2[self._pad_len+1:] - p2[self._pad_len:-1]
                th = np.pad(np.arctan2(d2, d1), pad_width=(self._pad_len+1, 0), mode='edge')
                self._xs_filtered[:, -2] = np.cos(th)
                self._xs_filtered[:, -1] = np.sin(th)

        else:
            self._xs_filtered = np.copy(xs)

    def _fit(self, xs, xs_next):
        """
        xs: np.ndarray of shape (# data, state dim.) representing the states
        xs_next: np.ndarray of shape (# data, state dim.) representing the next states

        This compute the Koopman matrix of the system via DMD given a batch of data.
        """
        # self._K = xs_next.T @ np.linalg.pinv(xs.T)
        self._K = xs_next.T @ np.linalg.pinv(xs.T)
        self._cache_Ks(self._n_steps)
        return

    def predict(self, x, n_steps=None):
        if n_steps is not None and n_steps != self._n_steps:
            self._cache_Ks(n_steps)
        if n_steps is None:
            n_steps = self._n_steps

        observables = self._to_observables(x)
        observables_next = self._Ks @ observables
        # prediction over N steps
        # [x(k+1), ..., x(k+N)]^T
        observables_next = np.reshape(observables_next, newshape=(n_steps, self._state_dim))
        xs_next = self._to_states(observables_next)     # shape = (# steps, state dim.)
        return xs_next

    def _cache_Ks(self, n_steps):
        """
        Cache the powers of K to avoid redundant computation when generating predictions.
        """
        self._n_steps = n_steps
        self._Ks = np.vstack([np.linalg.matrix_power(self._K, i) for i in range(1, n_steps+1)])
        return

    def get_filtered_trajectory(self):
        return np.copy(self._xs_filtered)