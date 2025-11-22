import os
from typing import Dict, List, Tuple
import json
import numpy as np


class Obstacle:
    def __init__(self, dim: int, name: str = ''):
        assert dim > 0
        self._dim = dim
        self._name = name

    def distance_from(self, points):
        raise NotImplementedError

    def to_vertices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self):
        return self._name

class Rectangle(Obstacle):
    def __init__(self, xmin, xmax, ymin, ymax, name: str = 'rectangle'):
        if xmin >= xmax:
            raise ValueError('xmin must be < xmax')
        if ymin >= ymax:
            raise ValueError('ymin must be < ymax')

        super().__init__(dim=2, name=name)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @classmethod
    def from_bounds(cls, data_dict: Dict[str, float], name: str = ''):
        xmin = data_dict['xmin']
        xmax = data_dict['xmax']
        ymin = data_dict['ymin']
        ymax = data_dict['ymax']
        return Rectangle(xmin, xmax, ymin, ymax, name=name)

    def distance_from(self, points: np.ndarray) -> np.ndarray:
        """
        points: numpy array of shape (*, 2) where the last axis contains 2D positions

        return: array of shape *
        """
        assert points.shape[-1] == self._dim
        x, y = points[..., 0], points[..., 1]
        dx = np.clip(x - self.xmax, a_min=0., a_max=None) + np.clip(self.xmin - x, a_min=0., a_max=None)
        dy = np.clip(y - self.ymax, a_min=0., a_max=None) + np.clip(self.ymin - y, a_min=0., a_max=None)
        return (dx ** 2 + dy ** 2) ** .5

    def to_vertices(self) -> np.ndarray:
        # counterclockwise order
        return np.array([[self.xmin, self.ymin], [self.xmax, self.ymin], [self.xmax, self.ymax], [self.xmin, self.ymax]])

    def to_halfspaces(self) -> Tuple[np.ndarray, np.ndarray]:
        # return (A, b) of the rectangle represented as Ax <= b
        A = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        b = np.array([self.xmax, -self.xmin, self.ymax, -self.ymin])
        return A, b

class Geometry:
    def __init__(self, xlim=None, ylim=None):
        self._obstacles: List[Obstacle] = []
        if xlim is not None:
            if xlim[0] >= xlim[1]:
                raise ValueError('xlim[0] must be < xlim[1]')
            self._xlim = xlim
        else:
            self._xlim = [-1e2, 1e2]
        if ylim is not None:
            if ylim[0] >= ylim[1]:
                raise ValueError('ylim[0] must be < ylim[1]')
            self._ylim = ylim
        else:
            self._ylim = [-1e2, 1e2]
        self._max_dist = ((self._xlim[1] - self._xlim[0]) ** 2 + (self._ylim[1] - self._ylim[0]) ** 2) ** .5

    def distance_from(self, points: np.ndarray) -> np.ndarray:
        if self._obstacles:
            distances = [o.distance_from(points) for o in self._obstacles]
            return np.min(distances, axis=0)
        else:
            # no obstacles
            shape = points.shape[:-1]
            return np.full(shape=shape, fill_value=self._max_dist)

    def add_obstacle(self, obstacle: Obstacle):
        self._obstacles.append(obstacle)

    @property
    def lower_bound(self):
        return np.array([self._xlim[0], self._ylim[0]])

    @property
    def upper_bound(self):
        return np.array([self._xlim[1], self._ylim[1]])

    @property
    def xlim(self):
        return np.copy(self._xlim)

    @property
    def ylim(self):
        return np.copy(self._ylim)

    def __iter__(self):
        return iter(self._obstacles)


def load_geometry(path) -> Geometry:
    assert os.path.exists(path), path
    with open(path, "r") as f:
        data_dict = json.load(f)
        xlim = data_dict['xlim'] if 'xlim' in data_dict else None
        ylim = data_dict['ylim'] if 'ylim' in data_dict else None
    geom = Geometry(xlim=xlim, ylim=ylim)
    for name, data in data_dict['obstacles'].items():
        o = Rectangle.from_bounds(data, name)
        geom.add_obstacle(o)
    return geom
