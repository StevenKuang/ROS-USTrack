#!/usr/bin/env python3

# ROS-related imports
import rospy
import message_filters
from std_msgs.msg import Time, std_msgs
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image as ROSImage
from iiwa_msgs.msg import JointPosition, CartesianPose, JointVelocity

# System and OS-related imports
import sys
import os
import subprocess
import platform
import time

# Image processing and computer vision imports
import cv2
from skimage.measure import label
from PIL import Image 
from cv_bridge import CvBridge, CvBridgeError

# Segmentation and tracking imports
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from aot_tracker import _palette

# Scientific computing imports
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R
import gc

# Transformation imports
from tf.transformations import (
    quaternion_matrix, quaternion_about_axis, quaternion_multiply,
    quaternion_conjugate, quaternion_from_matrix
)

# Visualization imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Threading imports
import threading
data_lock = threading.Lock()

# Custom imports
from iiwa_python import *
from setup_control_mode import set_force_mode, set_position_control_mode
import cactuss_infer as ci
from cut.data.base_dataset import get_transform

# Ensure current working directory is in sys.path
sys.path.append(os.getcwd())


CACTUSS = False
if CACTUSS:
    cactuss=ci.CACTUSS()
else:
    cactuss=None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bridge = CvBridge()
image_received = False
cv_image = None
com_mask_stack = []

INIT_FORCE = 5
STIFFNESS = 500     # default: 500
NR_LAST_POS = 1  # average the last X positions
THRESHOLD = 5

# SIZE = 256
# Cephasonics img size in mm: 650px x 650px
img_width_mm = 97  # 161.19 -> in ImFusion  38
img_height_mm = 97  # 107.97 -> in ImFusion    90
# Cephasonics img size in px: 650px x 650px
img_width_px = 540  # 540 -> in ImFusion
img_height_px = 540  # 691 -> in ImFusion
LAMBDA_X = img_width_mm / img_width_px
LAMBDA_Y = img_height_mm / img_height_px
MM2M = 0.001

STEP_SIZE = 0.05  # 2cm
# STEP_SIZE = 0.05  # 5cm

class UltrasoundProbe:
    def __init__(self, length, probe_type='linear', kuka_pose=None):
        self.length = length
        self.probe_type = probe_type
        if kuka_pose is None:
            self.position, self.orientation = None, None
            self.x_dir, self.y_dir, self.z_dir = None, None, None
        else:
            self.position, self.orientation, [self.x_dir, self.y_dir, self.z_dir] = self.flange_to_probe(kuka_pose)

    # def flange_to_probe(self, kuka_pose):
    #     flange_position = np.array([kuka_pose.pose.position.x, kuka_pose.pose.position.y, kuka_pose.pose.position.z])
    #     flange_orientation = np.array([kuka_pose.pose.orientation.x, kuka_pose.pose.orientation.y, kuka_pose.pose.orientation.z, kuka_pose.pose.orientation.w])
    #     # z_dir from the flange orientation
    #     z_dir = quaternion_matrix(flange_orientation)[:3, 2]
    #     position = flange_position + self.length * z_dir
    #     dir_vecs = [quaternion_matrix(flange_orientation)[:3, i] for i in range(3)]
    #     return position, flange_orientation, dir_vecs

    def flange_to_probe(self, kuka_pose):
        # Optimized for speed   
        flange_position = np.array([kuka_pose.pose.position.x, kuka_pose.pose.position.y, kuka_pose.pose.position.z])
        flange_orientation = np.array([kuka_pose.pose.orientation.x, kuka_pose.pose.orientation.y, kuka_pose.pose.orientation.z, kuka_pose.pose.orientation.w])
        # Create a rotation object from the quaternion matrix
        rotation_matrix = R.from_quat(flange_orientation).as_matrix()
        # Compute the position of the probe with z_dir
        position = flange_position + self.length * rotation_matrix[:3, 2]
        # Extract the direction vectors
        dir_vecs = rotation_matrix[:, :3].T 
        return position, flange_orientation, dir_vecs

    def update_probe(self, kuka_pose):
        self.position, self.orientation, [self.x_dir, self.y_dir, self.z_dir] = self.flange_to_probe(kuka_pose)

    def get_probe_directions(self):
        return [self.x_dir, self.y_dir, self.z_dir]

class KukaControl:

    def __init__(self):
        self.joint_publisher = rospy.Publisher('/iiwa/command/JointPosition', JointPosition, queue_size=20)
        self.pose_publisher = rospy.Publisher('/iiwa/command/CartesianPoseLin', PoseStamped, queue_size=20)
        self.cartesian_pose_sub = rospy.Subscriber('/iiwa/state/CartesianPose', CartesianPose, self.pose_callback)
        self.image_pub_seg_stable = rospy.Publisher("/imfusion/sim_seg_s", ROSImage, queue_size=1)

        self.bridge = CvBridge()
        self.aorta_stats = {"x": 0, "y": 0, "aorta_size": 0, "width": 0, "height": 0}
        self.aorta_gt = {"gt_x": 0, "gt_y": 0}

        self.aorta_stats_arr = []
        self.init_steps = {
            "x": False,
            "y": False,
            "a7": False
        }
        self.step = 0

        self.connected = False
        self.probe = UltrasoundProbe(0.2)
        self.sweeped = False
        self.target_pose = None

        self.current_cartesian_pose = None
        self.destination_reached_counter = 0


    def attach_probe(self, probe):
        self.probe = probe
        self.probe.update_probe(self.current_cartesian_pose)

    def pose_callback(self, data):
        self.current_cartesian_pose = data.poseStamped
        self.probe.update_probe(data.poseStamped)

    def get_current_pose(self):
        try:
            current_pose = rospy.wait_for_message('/iiwa/state/CartesianPose', CartesianPose, timeout=2)
            if current_pose is None:
                raise rospy.ROSException('No pose from /iiwa/state/CartesianPose received')

            self.connected = True
            return current_pose.poseStamped
        # print('CURRENT POSE: ', current_pose.poseStamped)
        # rospy.sleep(0.5)        # why sleep here? probably to wait for the pose to be published
        except rospy.ROSException as e:
            print(e)
            return None
    
    def destination_reached(self):
        reached = rospy.wait_for_message('/iiwa/state/DestinationReached', Time)
        return reached

    def probe_length_calibration(self, axis = 'y'):
        errs = []
        length_base = 0.225
        length_1 = 0.001
        # length_2 = 0.001
        # old_probe = self.probe
        # do a grid search
        for i in range(10):
            probe = UltrasoundProbe(length=length_base + i * length_1)
            self.attach_probe(probe)
            errs.append(self.sweep(axis, 10))
        
        self.attach_probe(UltrasoundProbe(length=length_base + errs.index(min(errs)) * length_1))
        print('Errors: ', errs)
        print("Min error: ", min(errs))
        print("Min error length: ", length_base + errs.index(min(errs)) * length_1)

    def get_a7_velocity(self):
        velocities = rospy.wait_for_message('/iiwa/state/JointVelocity', JointVelocity, timeout=2)
        a7_velocity = velocities.velocity.a7    
        # eliminate tiny purturbing movements 
        if a7_velocity < 0.001:
            a7_velocity = 0
        return a7_velocity

    def get_a7_state(self):
        a7_state = rospy.wait_for_message('/iiwa/state/JointPosition', JointPosition, timeout=2)
        a7_position = a7_state.position.a7
        return a7_position

    def pub_a7_state(self, joint_state, new_a7_position):
        joint_state.position.a7 = new_a7_position
        self.joint_publisher.publish(joint_state)
        while not self.destination_reached():
            rospy.sleep(0.5)
        self.destination_reached_counter += 1

    def rotate_a7(self, angle=90):
        joint_state = rospy.wait_for_message('/iiwa/state/JointPosition', JointPosition, timeout=2)
        a7_position = joint_state.position.a7
        new_a7_position = a7_position + np.radians(angle)
        self.pub_a7_state(joint_state, new_a7_position)
    
    def get_flange_directions(self, current_pose):
        
        # Extract the quaternion
        quaternion = [
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix(quaternion)
        
        # Extract the direction vectors
        x_direction = rotation_matrix[:3, 0]
        y_direction = rotation_matrix[:3, 1]
        z_direction = rotation_matrix[:3, 2]

        return x_direction, y_direction, z_direction

    def msg2img(self, data):
        try:
            # print('DATA: \n', data)
            cv_image = self.bridge.imgmsg_to_cv2(data)  # , "mono8")
            print('IMAGE : ', cv_image.shape)
        except CvBridgeError as e:
            print(e)

        return cv_image

    def calc_center(self, img_nd):
        aorta_size = np.count_nonzero(img_nd == 1)

        min_x, min_y = img_nd.shape[0], img_nd.shape[1]
        max_x, max_y = 0, 0
        for iy, ix in np.ndindex(img_nd.shape):
            if img_nd[iy, ix] == 1:
                if ix < min_x: min_x = ix
                if iy < min_y: min_y = iy

                if ix > max_x: max_x = ix
                if iy > max_y: max_y = iy

        width = max_x - min_x
        height = max_y - min_y

        center_x_px = min_x + (max_x - min_x) / 2
        center_y_px = min_y + (max_y - min_y) / 2

        return center_x_px, center_y_px, width, height, aorta_size

    # convert px to meters
    def px2m(self, center_x_px):
        return LAMBDA_X * center_x_px * MM2M  # for meters


    def calc_travel_dist(self, start, end):
        # start and end are postStamped msgs
        start_vec = np.array([start.pose.position.x, start.pose.position.y, start.pose.position.z])
        end_vec = np.array([end.pose.position.x, end.pose.position.y, end.pose.position.z])
        return np.linalg.norm(end_vec - start_vec)

    def move_forward(self):
        current_pose = self.current_cartesian_pose
        # current_pose.pose.position.z = current_pose.pose.position.z - 0.2  # 20cm for tool
        current_pose.pose.position.x = current_pose.pose.position.x - STEP_SIZE     
        self.pose_publisher.publish(current_pose)
    
    def move_flange_with_dir_retention(self, direction, dist, current_pose=None):
        if current_pose is None:
            current_pose = self.current_cartesian_pose
        current_pose.pose.position.x = current_pose.pose.position.x + direction[0] * dist
        current_pose.pose.position.y = current_pose.pose.position.y + direction[1] * dist
        current_pose.pose.position.z = current_pose.pose.position.z + direction[2] * dist
        self.pose_publisher.publish(current_pose)
        while not self.destination_reached():
            rospy.sleep(0.5)
        self.destination_reached_counter += 1

    def rotate(self, axis='y', angle=30):
        '''
        rotate the flange around the x, y, or z axis
        if x, rotate around x axis
        the x, y, z axis are the flange directions not the world directions
        use positive angle for clockwise rotation
        '''
        # Get current pose
        current_pose = self.current_cartesian_pose
        print('CURRENT POSE: \n', current_pose.pose)
        # Extract the current orientation as a quaternion
        current_orientation = current_pose.pose.orientation
        [x_dir, y_dir, z_dir] = self.get_flange_directions(current_pose)
        current_quaternion = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]

        # Define the rotation quaternion based on the axis and angle
        angle_rad = np.radians(angle)

        if axis == 'x':
            rotation_quaternion = quaternion_about_axis(angle_rad, x_dir)
        elif axis == 'y':
            rotation_quaternion = quaternion_about_axis(angle_rad, y_dir)
        elif axis == 'z':
            rotation_quaternion = quaternion_about_axis(angle_rad, z_dir)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        # Compute the new orientation by combining the current orientation with the rotation quaternion
        new_quaternion = quaternion_multiply(current_quaternion, rotation_quaternion)

        # Set the new orientation in the current pose
        current_pose.pose.orientation.x = new_quaternion[0]
        current_pose.pose.orientation.y = new_quaternion[1]
        current_pose.pose.orientation.z = new_quaternion[2]
        current_pose.pose.orientation.w = new_quaternion[3]

        # Publish the new pose
        self.pose_publisher.publish(current_pose)
        print('PUBLISH POSE: \n', current_pose.pose)

        while not self.destination_reached():
            rospy.sleep(0.5)
        self.destination_reached_counter += 1

    def tilt(self, axis='y', angle=10, additional_depth=0.0):
        '''
        Tilt the probe in a circular motion
        Use the current pose as the start of the tilt,
        the move [angle] degrees to the left and right
        radius is the distance from the flange to the tip of the probe
        move x and z directions relative to the probe, and keep y direction
        '''

        # Get current pose
        current_pose = self.current_cartesian_pose
        # print('CURRENT POSE: \n', current_pose.pose)
        # print('--------------------------------------\n')
        # print('PROBE DIRECTIONS: \n', self.probe.x_dir, self.probe.y_dir, self.probe.z_dir)

        # Calculate the new positions of the flange maintaining the probe tip position
        flange_tip_position = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
    
        # Calculate the rotation quaternion for the given angle
        angle_rad = np.radians(angle)
        if axis == 'x':
            rotation_quaternion = quaternion_about_axis(angle_rad, self.probe.x_dir)
        elif axis == 'y':
            rotation_quaternion = quaternion_about_axis(angle_rad, self.probe.y_dir)
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        
        probe_length = self.probe.length + additional_depth
        rotated_probe_tip_position = quaternion_matrix(rotation_quaternion)[:3, :3].dot(probe_length * np.array(self.probe.z_dir)) + flange_tip_position

        new_z_dir = rotated_probe_tip_position - flange_tip_position
        new_z_dir = new_z_dir / np.linalg.norm(new_z_dir)
        if additional_depth != 0.0:
            old_probe_position = flange_tip_position + (self.probe.z_dir * probe_length)
            new_position = old_probe_position - (new_z_dir * probe_length)
        else:
            new_position = self.probe.position - (new_z_dir * probe_length)
        
        if axis == 'x':
            new_x_dir = self.probe.x_dir
            new_y_dir = np.cross(new_z_dir, new_x_dir)
            new_y_dir = new_y_dir / np.linalg.norm(new_y_dir)
        elif axis == 'y':
            new_y_dir = self.probe.y_dir
            new_x_dir = np.cross(new_y_dir, new_z_dir)
            new_x_dir = new_x_dir / np.linalg.norm(new_x_dir)
        else:
            raise ValueError("Axis must be 'x' or 'y'")

        new_transformation_matrix = np.eye(4)
        new_transformation_matrix[:3, 0] = new_x_dir
        new_transformation_matrix[:3, 1] = new_y_dir
        new_transformation_matrix[:3, 2] = new_z_dir
        new_transformation_matrix[:3, 3] = new_position

        new_quaternion = quaternion_from_matrix(new_transformation_matrix)
        
        # # Plotting the positions
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Extracting the coordinates
        # x_coords = [flange_tip_position[0], self.probe.position[0], rotated_probe_tip_position[0], new_position[0]]
        # y_coords = [flange_tip_position[1], self.probe.position[1], rotated_probe_tip_position[1], new_position[1]]
        # z_coords = [flange_tip_position[2], self.probe.position[2], rotated_probe_tip_position[2], new_position[2]]

        # # Plotting points
        # ax.scatter(x_coords, y_coords, z_coords, c=['r', 'g', 'b', 'm'], marker='o')

        # # Annotating points
        # ax.text(flange_tip_position[0], flange_tip_position[1], flange_tip_position[2], 'flange_tip_position', color='red')
        # ax.text(self.probe.position[0], self.probe.position[1], self.probe.position[2], 'self.probe.position', color='green')
        # ax.text(rotated_probe_tip_position[0], rotated_probe_tip_position[1], rotated_probe_tip_position[2], 'rotated_probe_tip_position', color='blue')
        # ax.text(new_position[0], new_position[1], new_position[2], 'new_position', color='magenta')


        # # Plotting the old and new z_dir vectors
        
        # ax.quiver(flange_tip_position[0], flange_tip_position[1], flange_tip_position[2], 
        #         self.probe.z_dir[0], self.probe.z_dir[1], self.probe.z_dir[2], color='black', length=probe_length, normalize=False)
        # ax.quiver(flange_tip_position[0], flange_tip_position[1], flange_tip_position[2], 
        #         new_z_dir[0], new_z_dir[1], new_z_dir[2], color='cyan', length=probe_length, normalize=False)
        # ax.quiver(new_position[0], new_position[1], new_position[2],
        #         new_z_dir[0], new_z_dir[1], new_z_dir[2], color='cyan', length=probe_length, normalize=False)
        
        # # plot flange directions
        # ax.quiver(self.probe.position[0], self.probe.position[1], self.probe.position[2], 
        #         self.probe.x_dir[0], self.probe.x_dir[1], self.probe.x_dir[2], color='red', length=0.05, normalize=False, label='x_dir')
        # ax.quiver(self.probe.position[0], self.probe.position[1], self.probe.position[2],
        #         self.probe.y_dir[0], self.probe.y_dir[1], self.probe.y_dir[2], color='green', length=0.05, normalize=False, label='y_dir')
        # ax.quiver(self.probe.position[0], self.probe.position[1], self.probe.position[2], 
        #         self.probe.z_dir[0], self.probe.z_dir[1], self.probe.z_dir[2], color='blue', length=0.05, normalize=False, label='z_dir')

        # # Setting the aspect ratio to be equal
        # max_range = np.array([x_coords, y_coords, z_coords]).ptp().max() / 2.0

        # mid_x = (max(x_coords) + min(x_coords)) * 0.5
        # mid_y = (max(y_coords) + min(y_coords)) * 0.5
        # mid_z = (max(z_coords) + min(z_coords)) * 0.5

        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # # Labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # # keep the scale the same
        # ax.set_box_aspect([1,1,1])
        

        # plt.title('3D Positions')
        # plt.show()


        new_pose = current_pose
        new_pose.pose.position.x = new_position[0]
        new_pose.pose.position.y = new_position[1]
        new_pose.pose.position.z = new_position[2]
        new_pose.pose.orientation.x = new_quaternion[0]
        new_pose.pose.orientation.y = new_quaternion[1]
        new_pose.pose.orientation.z = new_quaternion[2]
        new_pose.pose.orientation.w = new_quaternion[3]
        
        self.pose_publisher.publish(new_pose)
        # print('PUBLISH POSE: \n', new_pose.pose)
        # print('--------------------------------------\n')
        while not self.destination_reached():
            rospy.sleep(0.5)
        self.destination_reached_counter += 1

    def sweep(self, axis: str = 'y', angle: int = 15) -> float:
        '''
        Sweep the probe in a circular motion
        Use the current pose as the center of the sweep
        When angle is positive,
        sweep the probe [angle] degrees to counter clockwise first,
        then [angle*2] degrees to clockwise,
        then [angle] degrees to counter clockwise
        '''

        
        probe_pos = [self.probe.position]
        self.tilt(axis, angle)
        probe_pos.append(self.probe.position)
        self.tilt(axis, -angle*2)
        probe_pos.append(self.probe.position)
        self.tilt(axis, angle)
        probe_pos.append(self.probe.position)

        self.sweeped = True
        diffs = [np.linalg.norm(probe_pos[i] - probe_pos[0]) for i in range(1, 4)]
        print('DIFFS: ', diffs)
        print('Avg diff: ', np.mean(diffs))
        return np.mean(diffs)

    def sweep_list(self, axis: str = 'y', angles: [int] = [15, 30, 45], additional_depth: float = 0.0) -> float:
        '''
        Sweep the probe in a circular motion acoording to the angles list
        positive angles are counter clockwise
        '''
        probe_pos = [self.probe.position]
        for angle in angles:
            self.tilt(axis, angle, additional_depth)
            probe_pos.append(self.probe.position)

        # self.sweeped = True
        diffs = [np.linalg.norm(probe_pos[i] - probe_pos[0]) for i in range(1, len(angles) + 1)]
        print('DIFFS: ', diffs)
        print('Avg diff: ', np.mean(diffs))
        return np.mean(diffs)

    def flange_dir_check(self, dist=0.05):
        sleep_time = 0.5
        current_pose = self.current_cartesian_pose 
        center_pose = current_pose
        print('CURRENT POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('CURRENT DIRECTIONS: \n', current_dirs)

        print('--------------------------------------\n')
        print('moving in +X flange direction ('+ str(current_dirs[0]) +') for '+ str(100*dist) + ' cm: ')
        self.move_flange_with_dir_retention(current_dirs[0], dist)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)
        print('\n')

        print('moving in -X flange direction ('+ str(current_dirs[0]) +') for '+ str(100*dist) + ' cm: ')
        self.move_flange_with_dir_retention(-current_dirs[0], dist)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

        print('--------------------------------------\n')
        print('moving in +Y flange direction ('+ str(current_dirs[1]) +') for '+ str(100*dist) + ' cm: ')
        self.move_flange_with_dir_retention(current_dirs[1], dist)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)
        
        print('\n')
        print('moving in -Y flange direction ('+ str(current_dirs[1]) +') for '+ str(100*dist) + ' cm: ')
        self.move_flange_with_dir_retention(-current_dirs[1], dist)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

        print('--------------------------------------\n')
        print('moving in +Z flange direction ('+ str(current_dirs[2]) +') for '+ str(100*dist) + ' cm: ')
        self.move_flange_with_dir_retention(current_dirs[2], dist)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

        print('\n')
        print('moving in -Z flange direction ('+ str(current_dirs[2]) +') for '+ str(100*dist) + ' cm: ')
        self.move_flange_with_dir_retention(-current_dirs[2], dist)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

    def world_dir_check(self, dist=0.05):
        sleep_time = 0.5
        current_pose = self.current_cartesian_pose 
        center_pose = current_pose
        print('CURRENT POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('CURRENT DIRECTIONS: \n', current_dirs)

        print('--------------------------------------\n')
        print('moving in +X direction for '+ str(100*dist) + ' cm: ')
        current_pose.pose.position.x = current_pose.pose.position.x + dist 
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        # sleep till the kuka.destination_reached() is True
        while not self.destination_reached():
            rospy.sleep(sleep_time)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)
        print('\n')
        print('moving in -X direction for '+ str(100*dist) + ' cm: ')
        current_pose.pose.position.x = current_pose.pose.position.x - dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        while not self.destination_reached():
            rospy.sleep(sleep_time)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

        print('--------------------------------------\n')
        print('moving in +Y direction for '+ str(100*dist) + ' cm: ')
        current_pose.pose.position.y = current_pose.pose.position.y + dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        while not self.destination_reached():
            rospy.sleep(sleep_time)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)
        
        print('\n')
        print('moving in -Y direction for '+ str(100*dist) + ' cm: ')
        current_pose.pose.position.y = current_pose.pose.position.y - dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        while not self.destination_reached():
            rospy.sleep(sleep_time)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

        print('--------------------------------------\n')
        print('moving in +Z direction for '+ str(100*dist) + ' cm: ')
        current_pose.pose.position.z = current_pose.pose.position.z + dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        while not self.destination_reached():
            rospy.sleep(sleep_time)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)
        
        print('\n')
        print('moving in -Z direction for '+ str(100*dist) + ' cm: ')
        current_pose.pose.position.z = current_pose.pose.position.z - dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        while not self.destination_reached():
            rospy.sleep(sleep_time)
        current_pose = self.current_cartesian_pose 
        print('END POSE: \n', current_pose.pose.position)
        current_dirs = self.get_flange_directions(current_pose)
        print('END DIRECTIONS: \n', current_dirs)

    def draw_circle(self, radius):
        import math
        print('Select the plane to draw the circle: ')
        print('1. XY plane')
        print('2. XZ plane')
        print('3. YZ plane')
        plane = int(input())
        center_pose = self.current_cartesian_pose 
        num_points = 100  # Sparsely define the circle
        angle_step = 2 * math.pi / num_points  # radians
        sleep_time = 8  

        for i in range(num_points + 1):  
            if i == 0:
                sleep_time = 8
            else:
                sleep_time = 2
            angle = i * angle_step
            new_pose = self.current_cartesian_pose 
            
            if plane == 1:  # XY
                new_pose.pose.position.x = center_pose.pose.position.x + radius * math.cos(angle)
                new_pose.pose.position.y = center_pose.pose.position.y + radius * math.sin(angle)
                new_pose.pose.position.z = center_pose.pose.position.z

            elif plane == 2:  # XZ
                new_pose.pose.position.x = center_pose.pose.position.x + radius * math.cos(angle)
                new_pose.pose.position.y = center_pose.pose.position.y
                new_pose.pose.position.z = center_pose.pose.position.z + radius * math.sin(angle)

            elif plane == 3:  # YZ
                new_pose.pose.position.x = center_pose.pose.position.x
                new_pose.pose.position.y = center_pose.pose.position.y + radius * math.cos(angle)
                new_pose.pose.position.z = center_pose.pose.position.z + radius * math.sin(angle)

            else:
                print("Invalid plane selection. Please select 1, 2, or 3.")
                return

            rate = rospy.Rate(100)
            print('PUBLISH POSE: \n', new_pose.pose.position)
            self.pose_publisher.publish(new_pose)
            while not self.destination_reached():
                rate.sleep(0.5)
            
            current_pose = self.current_cartesian_pose 
            print('END POSE: \n', current_pose.pose.position)

        # return to the initial position
        print('Return to the initial position: ')
        print('PUBLISH POSE: \n', center_pose.pose.position)
        self.pose_publisher.publish(center_pose)
        while not self.destination_reached():
            rate.sleep(0.5)
        print('Circle drawing complete.')

    def from_a_to_b(self):
        print('###############################################')
        print('############ MOVING FROM A TO B ############')
        print('Please use hand guidance mode to move the robot to the END POSITION and press Enter.')
        input()
        end_pose = self.current_cartesian_pose 
        # print('END POSE: \n', end_pose.pose.position)
        print('END POINT registered.')
        print('###############################################')
        print('Please use hand guidance mode to move the robot to the START POSITION and press Enter.')
        input()
        start_pose = self.current_cartesian_pose 
        # print('START POSE: \n', start_pose.pose.position)
        print('START POINT registered.')
        print('###############################################')
        print('Press Enter to move the robot from the START to the END POSITION.')
        input()
        self.pose_publisher.publish(end_pose)
        while not self.destination_reached():
            rospy.sleep(0.5)
        self.destination_reached_counter += 1
        print('END POSE: \n', end_pose.pose.position)
        print('###############################################')

    def move_point(self, u1, v1, u2, v2):
        # when probe is mounted in a transversal position, moving the x_dir of the probe 
        # will shift the image plane, thus we first consider y_dir and z_dir
        y_dist = (u2 - u1) * LAMBDA_Y * MM2M
        z_dist = (v2 - v1) * LAMBDA_X * MM2M
        
        combined_dir = y_dist * self.probe.y_dir + z_dist * -self.probe.z_dir     # z_dir is inverted
        # normalize the combined direction
        combined_dist = np.linalg.norm(combined_dir)
        combined_dir = combined_dir / combined_dist

        self.move_flange_with_dir_retention(combined_dir, combined_dist)

    def follow_segmentation(self, frame, pred_mask, grace = False):
        global com_mask_stack
        com_u, com_v = com_mask_stack[-1]
        H, W = pred_mask.shape[:2]
        '''
        grace = False: keep com in the center of the image
        grace = True: keep com in the grace area around the center of the image
        grace area definition:
        (0,0)              Frame
        -------------------------------------------
        -                                         -
        -                   h/3                   -
        -                                         -
        -         -----------------------         -
        -         -     grace area      -         -
        -         -                     -         -
        -   w/4   -                     -   w/4   -
        -         -    · (x,y)          -         -
        -         -                     -         -
        -         -----------------------         -
        -                   h/5                   -
        ------------------------------------------- (H, W)
        '''
        if grace:
            # com
            grace_w_top = 0.45
            grace_w_bottom = 0.45
            grace_h_top = 0.45
            grace_h_bottom = 0.3
            # right tip
            grace_w = 0.47
        else:
            grace_w = 0.5
            grace_h_top = 0.5
            grace_h_bottom = 0.5

        # translate to a bbox of (x1, y1, x2, y2)
        grace_box = [W * grace_w_top, H * grace_h_top, W * (1 - grace_w_bottom), H * (1 - grace_h_bottom)]

        clamped_com_u = int(max(min(com_u, W * (1 - grace_w_bottom)), W * grace_w_top))
        clamped_com_v = int(max(min(com_v, H * (1 - grace_h_bottom)), H * grace_h_top))
        # check if the com is in the grace area
        if (clamped_com_u, clamped_com_v) != (com_u, com_v):
            # move the robot in the direction of the com
            self.move_point(com_u, com_v, clamped_com_u, clamped_com_v)
            print('Moving from (', com_u, com_v, ') to (', clamped_com_u, clamped_com_v, ')')
        else:
            print('Center of Mass in the grace area. No movement.')

        # debug, draw the grace area, center of mass and the movement
        # frame_with_mask = draw_mask(frame, pred_mask)
        # cv2.rectangle(img=frame_with_mask, pt1=(int(grace_box[0]), int(grace_box[1])), pt2=(int(grace_box[2]), int(grace_box[3])), color=(0, 255, 0), thickness=2)
        # cv2.circle(frame_with_mask, (int(com_u), int(com_v)), 5, (255, 255, 255), -1)
        # cv2.circle(frame_with_mask, (int(clamped_com_u), int(clamped_com_v)), 5, (0, 0, 255), -1)
        # cv2.imshow('Processed Frame', frame_with_mask)
        # cv2.waitKey(1)


def image_callback(msg):
    global bridge
    global image_received
    global cv_image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "mono8")
        image_received = True
    except CvBridgeError as e:
        print(e)

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = img.copy()  # Copy the original image

    if id_countour:
        # Using vectorized operations and reducing complexity
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        for id in obj_ids:
            if id <= 255:
                color = np.array(_palette[id * 3:id * 3 + 3])
            else:
                color = np.array([0, 0, 0])
            
            binary_mask = (mask == id)
            contours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            
            img_mask[binary_mask] = img[binary_mask] * (1 - alpha) + color * alpha
            img_mask[contours] = 0
    else:
        binary_mask = (mask != 0)
        contours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[contours] = 0

    return img_mask.astype(img.dtype)

def centers_of_mass_mask(pred_mask):
    # flatten all objects in the mask to 1 object
    mask = pred_mask.copy()
    mask[mask != 0] = 1
    if np.count_nonzero(mask) == 0:
        return np.array([0, 0])
    return np.array([int(np.mean(np.nonzero(mask)[1])), int(np.mean(np.nonzero(mask)[0]))])

def tip_mask(pred_mask, dir = 'right'):
    # find the rightmost non 0 pixel in the mask
    # if there's multiple non 0 pixels, index them and take the one in the middle
    mask = pred_mask.copy()
    mask[mask != 0] = 1
    if np.count_nonzero(mask) == 0:
        return np.array([0, 0])
    if dir == 'right':
        tip_x = np.max(np.nonzero(mask)[1])
    elif dir == 'left':
        tip_x = np.min(np.nonzero(mask)[1])
    else:
        raise ValueError('Direction must be either right or left')
    
    # find the list of coords where x is tip
    tip_coords = np.argwhere(mask[:, tip_x] == 1)
    # take the middle y coord
    tip_y = tip_coords[len(tip_coords) // 2][0]

    return np.array([tip_x, tip_y])

def cactuss_infer_online(frame):
    H, W = frame.shape[:2]
    with torch.no_grad():
        # ndarray to PIL image
        rgbimg = Image.fromarray(frame.copy())
        transform = get_transform(cactuss.cut_opt)
        A = transform(rgbimg).to(device)
        B = transform(cactuss.real_B).to(device)
        A = torch.unsqueeze(A, 0)
        data = {'A': A, 'B': B, 'A_paths': None, 'B_paths': None}
        output_cut = ci.tensor2numpy(cactuss.infer_cut_net(data))
        output_cut = cv2.resize(output_cut, (W, H), interpolation=cv2.INTER_NEAREST)
    return output_cut

def segment(kuka : KukaControl = None):
    def init_check_thread(kuka, com, data_lock):
        catheter_depth = kuka.px2m(com[1])
        kuka.move_point(com[0], com[1], W//2, 3 * H//4)
        kuka.init_steps['a7'] = True
        kuka.rotate_a7(angle=30)
        kuka.rotate_a7(angle=-60)
        
        # wait here until kuka.init_steps['a7'] is False
        while kuka.init_steps['a7']:
            time.sleep(0.5)

        if kuka.target_pose is not None:
            print('Go to final a7')
            kuka.pose_publisher.publish(kuka.target_pose)
            while not kuka.destination_reached():
                rospy.sleep(0.5)
            with data_lock:
                kuka.destination_reached_counter += 1
            kuka.target_pose = None
            print('Reached final a7')
        kuka.rotate_a7(angle=-90)
        # time.sleep(2)
        kuka.init_steps['x'] = True
        kuka.sweep_list('x', [15, -30], 0.06)
        
        while kuka.init_steps['x']:
            time.sleep(0.5)

        if kuka.target_pose is not None:
            print('Go to final x')
            kuka.pose_publisher.publish(kuka.target_pose)
            while not kuka.destination_reached():
                rospy.sleep(0.5)
            with data_lock:
                kuka.destination_reached_counter += 1
            kuka.target_pose = None
            print('Reached final x')

        kuka.init_steps['y'] = True
        kuka.sweep_list('y', [10, -20], 0.06)
        while kuka.init_steps['y']:
            time.sleep(0.5)

        if kuka.target_pose is not None:
            print('Go to final y')
            kuka.pose_publisher.publish(kuka.target_pose)
            while not kuka.destination_reached():
                rospy.sleep(0.5)
            with data_lock:
                kuka.destination_reached_counter += 1
            kuka.target_pose = None
            print('Reached final y')

        with data_lock:
            kuka.sweeped = True

    
    def follow_segmentation_thread(kuka, cv_image, pred_mask):
        kuka.follow_segmentation(cv_image, pred_mask, grace=True)

    def move_to_pose_thread(kuka, pose, data_lock):
        kuka.pose_publisher.publish(pose)
        while not kuka.destination_reached():
            rospy.sleep(0.5)
        with data_lock:
            kuka.destination_reached_counter += 1

    if kuka is not None:
        # print('##################################set_force_mode############################################')
        # print("============ START FORCE ADJUSTMENT ...")
        # set_force_mode(cartesian_dof=3, desired_force=INIT_FORCE, desired_stiffness=STIFFNESS, max_deviation_pos=1000,
        #             max_deviation_rotation_in_deg=1000)
        # print("============ Press `Enter` to continue ...")
        # input()

        # set position mode 
        print('##################################set_position_control_mode############################################')
        set_position_control_mode()
        print("============ Press `Enter` to continue ...")
        input()


        #desired_force = INIT_FORCE
        """ while True:
            set_force_mode(cartesian_dof=3, desired_force=desired_force, desired_stiffness=STIFFNESS, max_deviation_pos=1000,
                    max_deviation_rotation_in_deg=1000)

            print("============ Press s for stop and `Enter` to increase force by 1 ...")
            prompt = input()
            if prompt is 's':
                print("============ BREAK ============")
                print("============ END FORCE IS: ", desired_force)
                break
            else:
                desired_force += 1
                print("============ CURRENT FORCE: ", desired_force)
        """

    image_sub = rospy.Subscriber("/imfusion/imgs", ROSImage, image_callback)

    image_callback(rospy.wait_for_message("/imfusion/imgs", ROSImage, timeout=2))
    if cv_image is None:
        raise Exception("No image received in topic /imfusion/imgs. Please check the image topic.")

    W, H = cv_image.shape[:2]

    segtracker_args = {
    'sam_gap': 99999,
    'min_area': 200,
    'max_obj_num': 4,
    'min_new_obj_iou': 0.8,
    }

    frame_idx = 0
    segtracker = SegTracker(segtracker_args,sam_args,aot_args)
    segtracker.restart_tracker()

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = segtracker_args['sam_gap']

    start_time = None
    init_mask_size = 0
    max_total_brightness = 0
    max_brightness_per_pixel = 0
    init_pose = None
    if kuka is not None:
        start_reached_counter = kuka.destination_reached_counter
    best_frame = None
    best_pred_mask = None

    first_sweep = False
    initialized = False
    with torch.cuda.amp.autocast():
        while not rospy.is_shutdown():
            if image_received:
                frame = cv_image
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if kuka is not None:
                    kuka_pose = kuka.current_cartesian_pose 
                if CACTUSS:
                    frame = cactuss_infer_online(frame)
                    
                if frame_idx == 0:
                    # Select the initial bbox
                    roi = cv2.selectROI("ROI selector", frame, fromCenter=False, showCrosshair=True)
                    bbox = [[roi[0], roi[1]], [roi[0] + roi[2], roi[1] + roi[3]]]
                    cv2.destroyWindow("ROI selector")

                    # bbox mode
                    pred_mask, _ = segtracker.seg_acc_bbox(frame, bbox)
                    # pred_mask = segtracker.seg(frame)

                    torch.cuda.empty_cache()
                    gc.collect()
                    segtracker.add_reference(frame, pred_mask)

                    init_mask_size = np.count_nonzero(pred_mask)
                    start_time = time.time()    # ignore the first frame when calculating the FPS

                elif (frame_idx % sam_gap) == 0:
                    seg_mask = segtracker.seg(frame)
                    torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = segtracker.track(frame)
                    new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)
                    pred_mask = track_mask + new_obj_mask
                    segtracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = segtracker.track(frame,update_memory=True)
                
                torch.cuda.empty_cache()
                gc.collect()

                global com_mask_stack
                if pred_mask is not None:
                    # com_mask_stack.append(centers_of_mass_mask(pred_mask))
                    com_mask_stack.append(tip_mask(pred_mask, 'right'))
                else:
                    com_mask_stack.append(com_mask_stack[-1])
                
                if kuka is None:
                    frame_with_mask = draw_mask(frame, pred_mask)
                    # Display the processed frame with segmentation mask
                    cv2.circle(frame_with_mask, (com_mask_stack[-1][0], com_mask_stack[-1][1]), 5, (0, 255, 0), -1)
                    # Draw the moving direction of the catheter
                    if len(com_mask_stack) > 1:
                        com_mask_moving_direction = (com_mask_stack[-1] - com_mask_stack[-2])/np.linalg.norm(com_mask_stack[-1] - com_mask_stack[-2])
                        # cv2.arrowedLine(frame_with_mask, (com_mask[0], com_mask[1]), (2*int(com_mask[0] + com_mask_moving_direction[0]), 2*int(com_mask[1] + com_mask_moving_direction[1])), (0, 255, 0), 2)
                        # cv2.arrowedLine(frame_with_mask, (com_mask_stack[-2][0], com_mask_stack[-2][1]), (com_mask_stack[-1][0], com_mask_stack[-1][1]), (0, 255, 0), 2)
                        # draw the full trajectory
                        # for i in range(len(com_mask_stack) - 1):
                        #     if com_mask_stack[i][0] == 0 and com_mask_stack[i][1] == 0:
                        #         continue
                        #     cv2.line(frame_with_mask, (com_mask_stack[i][0], com_mask_stack[i][1]), (com_mask_stack[i+1][0], com_mask_stack[i+1][1]), (0, 255, 0), 2)
                    cv2.imshow('Processed Frame', frame_with_mask)
                    cv2.waitKey(1)
                else:
                    if pred_mask is None:
                        continue
                    # threading.Thread(target=follow_segmentation_thread, args=(kuka, frame, pred_mask)).start()
                    # print('Catheter Center of Mass: ', com_mask_stack[-1], end='\n')
                    if not initialized:
                        if not first_sweep:
                            com = centers_of_mass_mask(pred_mask)
                            init_thread = threading.Thread(target=init_check_thread, args=(kuka, com, data_lock))
                            init_thread.start()
                            first_sweep = True

                        # mask frame with the mask, only keep the pixels in the maskm rest is black
                        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        grayscale_masked_frame = cv2.bitwise_and(grayscale_frame, grayscale_frame, mask=pred_mask)

                        mask_size = np.count_nonzero(pred_mask)
                        total_brightness = np.sum(grayscale_masked_frame)
                        brightness_per_pixel = total_brightness / mask_size
                        
                        if kuka.init_steps['a7']:
                            if brightness_per_pixel > max_brightness_per_pixel:
                                max_brightness_per_pixel = brightness_per_pixel
                                init_pose = kuka_pose
                                init_mask_size = mask_size
                                best_frame = frame
                                best_pred_mask = pred_mask
                        else:
                            if total_brightness > max_total_brightness:
                                max_total_brightness = total_brightness
                                init_pose = kuka_pose
                                init_mask_size = mask_size
                                best_frame = frame
                                best_pred_mask = pred_mask

                        # if mask_size > init_mask_size:
                        #     init_mask_size = mask_size
                        #     init_pose = kuka_pose
                        
                        with data_lock:
                            if kuka.init_steps['a7'] and kuka.destination_reached_counter - start_reached_counter == 3:
                                print('Enter a7 if\n')
                                kuka.target_pose = init_pose
                                kuka.init_steps['a7'] = False
                                max_total_brightness = 0

                                segtracker.restart_tracker()
                                segtracker.add_reference(best_frame, best_pred_mask)
                                segtracker.first_frame_mask = best_pred_mask

                                start_reached_counter = kuka.destination_reached_counter
                                cv2.imshow('Best a7 Frame', draw_mask(best_frame, best_pred_mask))

                            if kuka.init_steps['x'] and kuka.destination_reached_counter - start_reached_counter == 4:
                                print('Enter x if\n')
                                kuka.target_pose = init_pose
                                kuka.init_steps['x'] = False
                                max_total_brightness = 0
                                
                                segtracker.restart_tracker()
                                segtracker.add_reference(best_frame, best_pred_mask)
                                segtracker.first_frame_mask = best_pred_mask

                                start_reached_counter = kuka.destination_reached_counter
                                cv2.imshow('Best x Frame', draw_mask(best_frame, best_pred_mask))

                            if kuka.init_steps['y'] and kuka.destination_reached_counter - start_reached_counter == 3:
                                kuka.target_pose = init_pose
                                print('Enter sweep if\n')
                                initialized = True
                                kuka.init_steps['y'] = False

                                segtracker.restart_tracker()
                                segtracker.add_reference(best_frame, best_pred_mask)
                                segtracker.first_frame_mask = best_pred_mask

                                print('INITIALIZED')
                                cv2.imshow('Best y Frame', draw_mask(best_frame, best_pred_mask))
                                continue
                        
                    else:
                        # do nothing
                        # pass
                        
                        if pred_mask is not None:
                            threading.Thread(target=follow_segmentation_thread, args=(kuka, frame, pred_mask)).start()

                # frame_with_mask = draw_mask(frame, pred_mask)
                if CACTUSS:
                    cv2.imshow('CACTUSS', draw_mask(frame, pred_mask))
                    frame_with_mask = draw_mask(cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB), pred_mask)
                else:
                    frame_with_mask = draw_mask(frame, pred_mask)
                #show the com
                cv2.circle(frame_with_mask, (com_mask_stack[-1][0], com_mask_stack[-1][1]), 5, (0, 255, 0), -1)
                cv2.imshow('Processed Frame', frame_with_mask)
                cv2.waitKey(1)

                if frame_idx > 0 and start_time is not None:
                    elapsed_time = time.time() - start_time
                    fps = frame_idx / elapsed_time
                    print("processed frame {}, obj_num {}, FPS: {:.2f}".format(frame_idx, segtracker.get_obj_num(), fps), end='\r')
                else:
                    print("processed frame {}, obj_num {}".format(frame_idx, segtracker.get_obj_num()), end='\r')
                frame_idx += 1

        print('\nfinished')
        cv2.destroyAllWindows()

        del segtracker
        torch.cuda.empty_cache()
        gc.collect()

drawing = False
mock_mask = None
def mouse_callback(event, x, y, flags, param):
    global drawing, mock_mask
    draw_color = (255, 255, 255)  
    size = 5
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mock_mask, (x, y), size, draw_color, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mock_mask, (x, y), size, draw_color, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(mock_mask, (x, y), size, draw_color, -1)




def main():
    kuka = KukaControl()
    probe = UltrasoundProbe(length=0.227)       # 0.227
    rospy.init_node('kuka_control', anonymous=True)  # , disable_signals=True)
    
    # global mock_mask
    # mock_mask = np.zeros((480, 640), dtype=np.uint8)
    # win_name = 'Draw mask'
    # cv2.namedWindow(win_name)
    # cv2.setMouseCallback(win_name, mouse_callback)
    # while True:
    #     cv2.imshow(win_name, mock_mask)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    # tip = tip_mask(mock_mask, 'right')

    # # draw the tip of the probe as red
    # cv2.circle(mock_mask, (tip[0], tip[1]), 5, (255, 0, 0), -1)
    # cv2.imshow('Tip of the probe', mock_mask)
    # cv2.waitKey(0)

    # segment()
    if kuka.get_current_pose() is None:
        print('KUKA not connected, segmenting without moving the robot.')
        segment()
    else:
        kuka.attach_probe(probe)
        segment(kuka)

    

    # print('##################################set_force_mode############################################')
    # print("============ START FORCE ADJUSTMENT ...")
    # set_force_mode(cartesian_dof=3, desired_force=INIT_FORCE, desired_stiffness=STIFFNESS, max_deviation_pos=1000,
    #             max_deviation_rotation_in_deg=1000)
    # print("============ Press `Enter` to continue ...")
    # input()

    # set position mode 
    print('##################################set_position_control_mode############################################')
    set_position_control_mode()
    print("============ Press `Enter` to continue ...")
    input()


    #desired_force = INIT_FORCE
    """ while True:
        set_force_mode(cartesian_dof=3, desired_force=desired_force, desired_stiffness=STIFFNESS, max_deviation_pos=1000,
                max_deviation_rotation_in_deg=1000)

        print("============ Press s for stop and `Enter` to increase force by 1 ...")
        prompt = input()
        if prompt is 's':
            print("============ BREAK ============")
            print("============ END FORCE IS: ", desired_force)
            break
        else:
            desired_force += 1
            print("============ CURRENT FORCE: ", desired_force)
    """
        
    print(rospy.get_namespace())
    # current_pose = kuka.get_current_pose()
    while not rospy.is_shutdown():
        kuka.attach_probe(probe)
        # kuka.world_dir_check(dist=0.03)
        # kuka.flange_dir_check(dist=0.03)

        # kuka.rotate_a7(angle=60)
        # kuka.sweep('y', 10)
        # kuka.sweep_list('y', [-20, 30, -10])
        kuka.rotate_a7(angle=90)
        # kuka.rotate_a7(angle=-60)
        # kuka.rotate_a7(angle=30)

        # catheter_depth = 0.045

        # kuka.tilt('y', 5, catheter_depth)
        # kuka.tilt('y', -10, catheter_depth)
        # kuka.tilt('y', 5, catheter_depth)


        # rep = 2
        # kuka.move_flange_with_dir_retention(kuka.probe.x_dir, -0.02)
        # for i in range(rep):
        #     kuka.move_flange_with_dir_retention(kuka.probe.x_dir, -0.015, current_pose)
        #     kuka.move_flange_with_dir_retention(kuka.probe.x_dir, +0.015, current_pose)
        # kuka.move_flange_with_dir_retention(kuka.probe.x_dir, 0.002, current_pose)

        # kuka.rotate(angle=10, axis='y')
        # kuka.rotate(angle=-10, axis='y')
        # kuka.run()
        # kuka.draw_circle(radius=0.05)
        # kuka.from_a_to_b()

        # rospy.sleep(0.3)

        # quit the loop
        break
        
        set_position_control_mode()

    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
