from typing import Any, Dict, List, NewType
import numpy as np
from scipy.optimize import linear_sum_assignment
from detection.detection_utils import DetectionResult, Box

ObjectID = NewType('TrackingID', int)  # type alias for object id

def is_dynamic(track, vel_threshold=0.3, cov_threshold=0.5, num_frames=5):
    """
    track: KalmanTracker object
    vel_threshold: threshold for avg_velocity (e.g. static if the avg_vel less than 0.1 m/s)
    cov_threshold: threshold for covariance of the estimation of current velocity(if small, high confidence)
    num_frames: numbers of frames for calculating average

    calculate avg_velocity by using the trajectory of recent num_frames
    and decide static or dynamic by using current covariacne
    """
    traj = track.trajectory
    # static unless enough information
    if len(traj) < 2:
        return False  # static

    # extract trajectory 
    recent = traj[-num_frames:] if len(traj) >= num_frames else traj
    speeds = []
    for i in range(1, len(recent)):
        # Euclidean distance / dt
        d = np.linalg.norm(recent[i] - recent[i - 1])
        speeds.append(d / track.dt)
    avg_speed = np.mean(speeds) if speeds else 0.0

    cov_trace = np.trace(track.P[2:4, 2:4])

    # static if avg_vel less than the treshold and low uncertainty for velocity estimation
    if avg_speed < vel_threshold and cov_trace < cov_threshold:
        return False  # static
    elif traj[0][0] >= 0.3 and traj[0][0] <= 5.0 and traj[0][1] >= -12.0 and traj[0][1] <= -6.3:
        return False  # static, glass door below
    elif traj[0][0] >= -7.0 and traj[0][0] <= 0.3and traj[0][1] >= -12.0 and traj[0][1] <= -8.5:
        return False  # static, left glass
    elif traj[0][0] >= 5.0 and traj[0][0] <= 13.0 and traj[0][1] >= -12.0 and traj[0][1] <= -8.5:
        return False  # static, right glass
    elif traj[0][0] >= -7.0 and traj[0][0] <= -2.1 and traj[0][1] >= -12.0 and traj[0][1] <= -0.3:
        return False  # static, left wall
    elif traj[0][0] >= 7.8 and traj[0][0] <= 13.0 and traj[0][1] >= -12.0 and traj[0][1] <= -0.3:
        return False  # static, right wall
    elif traj[0][0] >= -7.0 and traj[0][0] <= -1.9 and traj[0][1] >= 1.1 and traj[0][1] <= 5.0:
        return False  # static, upper-left wall
    elif traj[0][0] >= -0.5 and traj[0][0] <= 13.0 and traj[0][1] >= 0.9 and traj[0][1] <= 5.0:
        return False  # static, upper wall
    elif traj[0][0] >= 2.0 and traj[0][0] <= 3.4 and traj[0][1] >= -4.6 and traj[0][1] <= -1.6:
        return False  # static, middle square
    elif traj[0][0] >= -0.7 and traj[0][0] <= 0.5 and traj[0][1] >= -1.5 and traj[0][1] <= -0.6:
        return False  # static, left cylinder
    elif traj[0][0] >= 5.3 and traj[0][0] <= 6.4 and traj[0][1] >= -1.8 and traj[0][1] <= -0.8:
        return False  # static, right cylinder
    else:
        return True  # dynamic

# KalmanTracker: SORT style
class KalmanTracker:
    def __init__(self, initial_position, dt=0.1):
        self.dt = dt
        # 상태: [x, y, vx, vy]
        self.x = np.array([initial_position[0], initial_position[1], 0, 0], dtype=float)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.P = np.eye(4) * 10.0
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 1.0
        self.time_since_update = 0
        self.id = None
        self.trajectory = [np.array(initial_position, dtype=float)]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        pred = self.x[:2].copy()
        self.trajectory.append(pred)
        return pred

    def update(self, measurement):
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        self.time_since_update = 0
        # Update last trajectory section after renewing
        self.trajectory[-1] = self.x[:2].copy()


class Tracker:
    def __init__(self, max_age=3, dist_threshold=1.0, dt=0.1):
        self.max_age = max_age
        self.dist_threshold = dist_threshold
        self.dt = dt
        self.tracks: List[KalmanTracker] = []
        self.next_id = 0

    def __call__(self, detection_result: DetectionResult):
        # extract object position list from detection_result ([x, y] of each objects)
        detections = detection_result.get_object_positions()  # list of positions
        if len(detections) == 0:
            # If no detection, only perform the estimation for all trackers
            for track in self.tracks:
                track.predict()
            # When return, returns static/dynamic of each track's trajectory
            trajectories = {track.id: np.array(track.trajectory) for track in self.tracks}
            object_types = {track.id: ('dynamic' if is_dynamic(track) else 'static') for track in self.tracks}
            return trajectories, object_types

        detections = np.array(detections)  # shape (N, 2)

        # Perform for all trackers
        predicted_positions = []
        for track in self.tracks:
            pred = track.predict()
            predicted_positions.append(pred)
        if len(predicted_positions) > 0:
            predicted_positions = np.array(predicted_positions)  # (num_tracks, 2)
        else:
            predicted_positions = np.empty((0, 2))

        # cost matrix: Euclidean distance between predicted positions and detections
        if predicted_positions.shape[0] > 0:
            cost_matrix = np.linalg.norm(predicted_positions[:, None, :] - detections[None, :, :], axis=2)
        else:
            cost_matrix = np.empty((0, detections.shape[0]))

        # Allocated: Hungarian algorithm
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        assigned_tracks = set()
        assigned_detections = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.dist_threshold:
                continue
            self.tracks[r].update(detections[c])
            assigned_tracks.add(r)
            assigned_detections.add(c)

        # Not allocated detection: create new tracker
        for i in range(detections.shape[0]):
            if i not in assigned_detections:
                new_track = KalmanTracker(detections[i], dt=self.dt)
                new_track.id = self.next_id
                self.next_id += 1
                self.tracks.append(new_track)

        # Not allocated tracker: increase time_since_update because already called predict()
        # Delete tracker that exceed max_age
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        # Return: trajectories(for each track), object_types(static/dynamic)
        trajectories = {track.id: np.array(track.trajectory) for track in self.tracks}
        object_types = {track.id: ('dynamic' if is_dynamic(track) else 'static') for track in self.tracks}
        return trajectories, object_types
