import math
import numpy as np
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import rospy
import roslaunch
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import PointCloud2
import ros_numpy
import open3d as o3d
from geometry_msgs.msg import PoseStamped

class Robot:
    def __init__(self):
        # self.unordered_state = None
        self.obstacles = None
        self.ordered_state = None
        self.velocity = None
        self.pcd = None
        self._state = None
        #self._state_trajectory = []
        #self._control_sequence = []
        #self._time_sequence = [0.]
        self.num_odom_callback = 0
        self.num_pcd_callback = 0
        self.drive_msg = None

        #self.dlio_process = start_dlio_launch()
        #rospy.loginfo("Waiting 10 seconds before starting dlio.launch...")
        #time.sleep(10)

        #odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback, queue_size=20)
        pose_sub = rospy.Subscriber("/jackal/dlio/odom_node/pose", PoseStamped, self.pose_callback, queue_size=20)
        odom_sub = rospy.Subscriber("/jackal/dlio/odom_node/odom", Odometry, self.odom_callback, queue_size=20)
        pcd_sub = rospy.Subscriber("/ouster/points", PointCloud2, self.pcd_callback, queue_size=20)

        self.drive_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        #rospy.loginfo("Waiting for initial messages...")
        rospy.wait_for_message("/jackal/dlio/odom_node/pose", PoseStamped)
        rospy.wait_for_message("/jackal/dlio/odom_node/odom", Odometry)
        rospy.wait_for_message("/ouster/points", PointCloud2)

    def sim(self, linear_x, angular_z):
        self.drive_msg = Twist()
        self.drive_msg.linear.x = linear_x
        self.drive_msg.angular.z = angular_z  # yaw
        #self._control_sequence.append(ctrl)
        self.drive_pub.publish(self.drive_msg)
        #time.sleep(0.1)

    def pose_callback(self, msg):
        self.num_odom_callback += 1
        mes = msg.pose
        q = (mes.orientation.x, mes.orientation.y, mes.orientation.z, mes.orientation.w)
        (_, _, yaw) = quat2euler(*q)
        self.ordered_state = np.array([mes.position.x, mes.position.y, yaw, msg.header.stamp])

    def odom_callback(self, msg):
        twist = msg.twist.twist
        self.velocity = np.array([twist.linear.x, twist.angular.z])

    def pcd_callback(self, msg):
        self.num_pcd_callback += 1
        self.pcd = pointcloud2_to_open3d(msg)

        #self.temp = pcd

    '''
    def get_state(self):
        t_elapsed = time.time() - self._t_init
        self._time_sequence.append(t_elapsed)
        self._state_trajectory.append(np.copy(self._state))
        return np.copy(self._state)

    def get_log(self):
        return {'x_trajectory': np.array(self._state_trajectory)[:, 0],
                'y_trajectory': np.array(self._state_trajectory)[:, 1],
                'th_trajectory': np.array(self._state_trajectory)[:, 2],
                'linear_vel_sequence': np.array(self._control_sequence)[:, 0],
                'angular_vel_sequence': np.array(self._control_sequence)[:, 1],
                'time_sequence': np.array(self._time_sequence),
                'state_trajectory': np.array(self._state_trajectory)
                }
    '''
def start_dlio_launch():
    rospy.loginfo("Starting dlio.launch using roslaunch API...")
    # roslaunch의 UUID를 생성합니다.
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    # launch 파일 경로를 지정합니다.
    # 절대 경로 또는 roslaunch 'find' 기능을 통해 찾은 경로를 사용하세요.
    launch_file = "/home/core/jackal_ws/src/direct_lidar_inertial_odometry/launch/dlio.launch"  # 실제 launch 파일의 경로로 변경해야 합니다.
    launch_parent = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
    launch_parent.start()
    rospy.loginfo("dlio.launch started successfully")
    return launch_parent

def terminate_dlio_launch(launch_parent):
    if launch_parent is not None:
        rospy.loginfo("Shutting down dlio.launch...")
        launch_parent.shutdown()

def pointcloud2_to_open3d(pcd_msg):
    # Convert ROS PointCloud2 message to a numpy array
    cloud_array = np.reshape(ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msg), newshape=(-1))
    # Extract XYZ coordinates from the array
    points = np.zeros((cloud_array.shape[0], 3))
    points[:, 0] = cloud_array['x']
    points[:, 1] = cloud_array['y']
    points[:, 2] = cloud_array['z']

    # Create an Open3D point cloud object and assign the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def quat2euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z
