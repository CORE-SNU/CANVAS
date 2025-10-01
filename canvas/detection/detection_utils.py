import numpy as np
import cv2
from typing import List

from collections import namedtuple


Box = namedtuple('Box', ['x', 'y', 'w', 'h', 'deg', 'rad', 'area', 'vertices', 'resolution', 'pos'])


def points2box(points: np.ndarray) -> Box:
    """
    Compute the minimum enclosing rectangle of a point set

    points: (N, 2) numpy array of 2d points
    transformation: pose of the robot w.r.t. the world frame

    :return: parameters defining a rotated rectangle
    """

    RESOLUTION = .001                   # auxiliary value of resolution to transform points to an opencv input
    # ROBOT_RAD = .455 + .2             # physical size of the robot + safety margin
    ROBOT_RAD = 0.0
    points_as_int = (points / RESOLUTION).astype(int)
    rect = cv2.minAreaRect(points_as_int)
    # center: xy
    # size: (width, height)
    # angle: w.r.t. center, anticlockwise, degree
    center, size, angle = rect

    # vertices of the rectangle; needed for computing IoU
    # Note that the coordinates are given as integers
    vertices = np.int8(cv2.boxPoints(rect))

    center = tuple(RESOLUTION * c for c in center)
    size = tuple(ROBOT_RAD + RESOLUTION * s for s in size)

    x, y = center
    w, h = size
    rad = np.deg2rad(angle)

    return Box(
        x=x, y=y, w=w, h=h,
        deg=angle, rad=rad,
        area=w*h,
        vertices=vertices,              # unscaled value (to be used for computing convex hull intersections)
        resolution=RESOLUTION,
        pos=np.array([x, y])
        )


class DetectionResult:
    """
    An artificial data class that holds different representations of the detection result.
    """
    def __init__(self,
                 clusters,
                 boxes,
                 object_width_threshold=1.,
                 object_height_threshold=1.
                 ):

        self._clusters: List[np.ndarray] = clusters     # each element: array os shape (# points, 2)
        self._boxes: List[Box] = boxes

        self._positions: List[np.ndarray] = [b.pos for b in self._boxes]

        self._n_clusters = len(self._clusters)

        # divide the clusters into two groups: objects & surroundings
        # The objects will be tracked while the surroundings will be simply used as safety constraints.
        # criterion used here: width & height
        self._objects = [i for i in range(self._n_clusters)
                                    if self._boxes[i].w < object_width_threshold
                                        and self._boxes[i].h < object_height_threshold]

    def get_objects(self):
        return self._objects

    def get_object_positions(self):
        """
        for tracking & prediction
        """
        return [self._positions[i] for i in self._objects]

    def get_object_boxes(self):
        """
        for tracking & safety specification
        """
        return [self._boxes[i] for i in self._objects]

    def get_object_points(self):
        """
        for visualization
        """
        return [self._clusters[i] for i in self._objects]

    def get_object_position(self, obj):
        return self._positions[obj]

    def get_object_box(self, obj):
        return self._boxes[obj]

    def get_object_point(self, obj):
        return self._clusters[obj]

    def get_surrounding_boxes(self):
        return [self._boxes[i] for i in range(self._n_clusters) if i not in self._objects]

    def get_surrounding_points(self):
        return [self._clusters[i] for i in range(self._n_clusters) if i not in self._objects]
