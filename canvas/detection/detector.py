import math
import numpy as np
from typing import List, Tuple
import open3d as o3d

from detection.detection_utils import Box, DetectionResult, points2box


class Detector:
    def __init__(self):
        return

    def __call__(self,
                 pcd,
                 transformation=None,
                 effective_range=math.inf,
                 min_cluster_size=4
                 ) -> DetectionResult:

        # TODO: deep learning-based?
        clusters = self.extract_clusters(pcd, transformation, effective_range, min_cluster_size)

        boxes: List[Box] = [points2box(c) for c in clusters]
        return DetectionResult(clusters=clusters, boxes=boxes, object_width_threshold=1., object_height_threshold=1.)

    @staticmethod
    def extract_clusters(
            pcd,
            transformation=None,
            effective_range=math.inf,
            min_cluster_size=4
    ) -> List[np.ndarray]:
        """
        Clustering of a single frame via DBSCAN.
        """
        # TODO: other segmentation methods?
        points = np.asarray(pcd.points)
        # remove the points corresponding to the ceiling & the floor
        z_min = .1
        z_max = 2.
        points_z = points[:, 2]
        mask = np.logical_and(points_z >= z_min, points_z <= z_max)

        if effective_range < math.inf:
            # radius to consider
            mask = np.logical_and(mask, np.sum(points[:, :2] ** 2, axis=-1) <= effective_range ** 2)

        pcd.points = o3d.utility.Vector3dVector(points[mask])

        # preprocessing: down-sampling + outlier removal
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        pcd, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.12)

        # TODO: parameter search: epsilon & minimum # of points
        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=4, print_progress=False))
        points = np.asarray(pcd.points)[:, :2]  # project onto xy-plane

        if transformation is not None:
            x = transformation['position_x']
            y = transformation['position_y']
            yaw = transformation['orientation_z']
            cos, sin = np.cos(yaw), np.sin(yaw)
            points = np.array([x, y]) + points @ np.array([[cos, sin], [-sin, cos]])

        max_label = labels.max()  # labels: 0, ..., max_label
        clusters = [points[labels == i] for i in range(max_label + 1)]
        # exclude clusters of size at most 4
        clusters = [c for c in clusters if c.shape[0] > min_cluster_size]
        return clusters
