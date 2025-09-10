import numpy as np
import itertools
from typing import Tuple, Dict
import cv2
from detection.detection_utils import Box


# from tracking.tracker import ObjectID

# TrackingResult = Dict[ObjectID, np.ndarray]

def compute_pairwise_IoU(boxes1, boxes2):
    """
    Computation of pairwise IOUs between boxes (B_1, ..., B_m) & (B_1', ..., B_n')
    The coordinate changes of (B_1, ..., B_m) are considered.

    Return a matrix M of shape (m, n) where M[i, j]: IOU between B_i & B_j'
    """

    n1, n2 = len(boxes1), len(boxes2)
    pairs = itertools.product(boxes1, boxes2)
    # position change according to the coordinate transform
    # Warning! Translate before rotation, not rotate before translation.
    scores = [compute_IoU(pair) for pair in pairs]
    return np.reshape(np.array(scores), newshape=(n1, n2))


def compute_IoU(pair: Tuple[Box, Box]):
    """
    Computes the IoU between two rectangles using OpenCV.

    Args:
        pair: A tuple of two rectangles (rect1, rect2) where each rectangle: a tuple (center, (width, height), angle)

    Returns:
        IoU: Intersection over Union between the two rectangles.
    """
    # Convert each rotated rectangle to a polygon (4 corner points)
    # TODO: Does using GIoU improve the performance?
    box1, box2 = pair
    resolution = box1.resolution

    vertices1 = np.array(box1.vertices, dtype=np.float32)
    vertices2 = np.array(box2.vertices, dtype=np.float32)

    if vertices1.size < 2 or np.isnan(vertices1).any():
        # Return IoU 0.0 if it is not invalid
        return 0.0
    if vertices2.size < 2 or np.isnan(vertices2).any():
        return 0.0

    # Reshape if the vertices is not (N, 2)
    if vertices1.ndim != 2 or vertices1.shape[1] != 2:
        try:
            vertices1 = vertices1.reshape(-1, 2)
        except Exception as e:
            print("Error reshaping vertices1:", vertices1, e)
            return 0.0
    if vertices2.ndim != 2 or vertices2.shape[1] != 2:
        try:
            vertices2 = vertices2.reshape(-1, 2)
        except Exception as e:
            print("Error reshaping vertices2:", vertices2, e)
            return 0.0

    if vertices1.shape[0] < 3 or vertices2.shape[0] < 3:
        return 0.0

    retval, intersection_polygon = cv2.intersectConvexConvex(vertices1, vertices2)

    if retval <= 0. or intersection_polygon is None or intersection_polygon.shape[0] < 3:
        return 0.

    intersection_area = retval * (resolution ** 2)
    union_area = box1.area + box2.area - intersection_area
    iou = intersection_area / union_area if union_area > 0. else 0.0
    return iou
