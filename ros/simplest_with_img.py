#!/usr/bin/env python3
import rospy
import std_msgs.msg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image
from iiwa_msgs.msg import JointPosition, CartesianPose, JointVelocity
from iiwa_msgs.msg import *
from iiwa_python import *
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import sys
import os

sys.path.append(os.getcwd())
from setup_control_mode import set_force_mode, set_position_control_mode
from skimage.measure import label
from std_msgs.msg import Time

from PIL import Image as pimg
import numpy as np
import cv2

# from iiwa_ros import cartesian_pose




INIT_FORCE = 5
STIFFNESS = 500     # default: 500
NR_LAST_POS = 1  # average the last X positions
THRESHOLD = 5

# SIZE = 256
# Cephasonics img size in mm: 650px x 650px
img_width_mm = 179.15  # 161.19 -> in ImFusion
img_height_mm = 100  # 107.97 -> in ImFusion
# Cephasonics img size in px: 650px x 650px
img_width_px = 650  # 540 -> in ImFusion
img_height_px = 650  # 691 -> in ImFusion
LAMBDA_X = img_width_mm / img_width_px
LAMBDA_Y = img_height_mm / img_height_px
MM2M = 0.001

STEP_SIZE = 0.005  # 2cm
# STEP_SIZE = 0.05  # 20cm


class AortaControl:

    def __init__(self):
        self.joint_publisher = rospy.Publisher('/iiwa/command/JointPosition', JointPosition, queue_size=20)
        self.pose_publisher = rospy.Publisher('/iiwa/command/CartesianPoseLin', PoseStamped, queue_size=1)
        # self.pose_action_publisher = rospy.Publisher('/iiwa/action/move_to_cartesian_pose/goal', MoveToCartesianPoseGoal, queue_size=1)
        self.image_pub_seg_stable = rospy.Publisher("/imfusion/sim_seg_s", Image, queue_size=1)

        self.bridge = CvBridge()
        self.aorta_stats = {"x": 0, "y": 0, "aorta_size": 0, "width": 0, "height": 0}
        self.aorta_gt = {"gt_x": 0, "gt_y": 0}

        self.aorta_stats_arr = []
        self.init = True
        self.step = 0

    def get_current_pose(self):
        current_pose = rospy.wait_for_message('/iiwa/state/CartesianPose', CartesianPose, timeout=2)
        # print('CURRENT POSE: ', current_pose.poseStamped)
        # rospy.sleep(0.5)        # why sleep here? probably to wait for the pose to be published
        return current_pose.poseStamped

    def destination_reached(self):
        reached = rospy.wait_for_message('/iiwa/state/DestinationReached', Time)
        return reached

    def get_a7_velocity(self):
        velocities = rospy.wait_for_message('/iiwa/state/JointVelocity', JointVelocity, timeout=2)
        a7_velocity = velocities.velocity.a7    
        # eliminate tiny purturbing movements 
        if a7_velocity < 0.001:
            a7_velocity = 0
        return a7_velocity

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

    def calc_move(self, center_x_px):
        move_dist_x_px = np.absolute(img_width_px / 2 - center_x_px)
        if center_x_px > img_width_px / 2:
            move_dist_x_px = - move_dist_x_px
        print('MOVE DIST X px: ', move_dist_x_px)

        center_x_meters = self.px2m(center_x_px)
        # center_y_meters = self.px2m(center_y_px) #for meters
        print('CENTER POINT meters: ', center_x_meters)  # , center_y_meters)

        move_dist_x_meters = self.px2m(move_dist_x_px)  # for meters
        print('MOVE DIST X meters: ', move_dist_x_meters)

        print('###############################################')
        print('############ AORTA STATISTICS ############')
        print('SIZE in px: ', self.aorta_stats["aorta_size"])
        print('WIDTH: ', self.aorta_stats["width"])
        print('HEIGHT: ', self.aorta_stats["height"])
        print('###############################################')

        return move_dist_x_meters

    def center_img(self, center_x_px):

        center_x_meters = self.calc_move(center_x_px)

        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.x - center_x_meters  # 20cm for tool
        self.pose_publisher.publish(current_pose)
        print('PUBLISH POSE CENTER IMG', current_pose)

    def calc_travel_dist(self, start, end):
        # start and end are postStamped msgs
        start_vec = np.array([start.pose.position.x, start.pose.position.y, start.pose.position.z])
        end_vec = np.array([end.pose.position.x, end.pose.position.y, end.pose.position.z])
        return np.linalg.norm(end_vec - start_vec)


    def move_forward(self):
        current_pose = self.get_current_pose()
        # current_pose.pose.position.z = current_pose.pose.position.z - 0.2  # 20cm for tool
        current_pose.pose.position.x = current_pose.pose.position.x - STEP_SIZE     
        self.pose_publisher.publish(current_pose)
        # print('PUBLISH POSE STEP', current_pose)

        # current_pose = rospy.wait_for_message('/iiwa/state/CartesianPose', CartesianPose, timeout=2)
        # current_pose.poseStamped.pose.position.x = current_pose.poseStamped.pose.position.x - STEP_SIZE
        # goal = MoveToCartesianPoseGoal()
        # goal.cartesian_pose = current_pose
        # self.pose_action_publisher.publish(goal)

    def movement_dir_check(self, dist):
        sleep_time = 6
        current_pose = self.get_current_pose()
        center_pose = current_pose
        print('CURRENT POSE: \n', current_pose.pose.position)

        print('--------------------------------------\n')
        print('moving in +X direction for '+ str(400*dist) + ' cm: ')
        current_pose.pose.position.x = current_pose.pose.position.x + dist 
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        rospy.sleep(sleep_time)
        current_pose = self.get_current_pose()
        print('END POSE: \n', current_pose.pose.position)

        print('\n')
        print('moving in -X direction for '+ str(400*dist) + ' cm: ')
        current_pose.pose.position.x = current_pose.pose.position.x - dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        rospy.sleep(sleep_time)
        current_pose = self.get_current_pose()
        print('END POSE: \n', current_pose.pose.position)

        print('--------------------------------------\n')
        print('moving in +Y direction for '+ str(400*dist) + ' cm: ')
        current_pose.pose.position.y = current_pose.pose.position.y + dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        rospy.sleep(sleep_time)
        current_pose = self.get_current_pose()
        print('END POSE: \n', current_pose.pose.position)
        
        print('\n')
        print('moving in -Y direction for '+ str(400*dist) + ' cm: ')
        current_pose.pose.position.y = current_pose.pose.position.y - dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        rospy.sleep(sleep_time)
        current_pose = self.get_current_pose()
        print('END POSE: \n', current_pose.pose.position)

        print('--------------------------------------\n')
        print('moving in +Z direction for '+ str(400*dist) + ' cm: ')
        current_pose.pose.position.z = current_pose.pose.position.z + dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        rospy.sleep(sleep_time)
        current_pose = self.get_current_pose()
        print('END POSE: \n', current_pose.pose.position)
        
        print('\n')
        print('moving in -Z direction for '+ str(400*dist) + ' cm: ')
        current_pose.pose.position.z = current_pose.pose.position.z - dist
        print('PUBLISH POSE: \n', current_pose.pose.position)
        self.pose_publisher.publish(current_pose)
        rospy.sleep(sleep_time)
        current_pose = self.get_current_pose()
        print('END POSE: \n', current_pose.pose.position)

    def draw_circle(self, radius):
        import math
        print('Select the plane to draw the circle: ')
        print('1. XY plane')
        print('2. XZ plane')
        print('3. YZ plane')
        plane = int(input())
        center_pose = self.get_current_pose()
        num_points = 36  # Sparsely define the circle
        angle_step = 2 * math.pi / num_points  # radians
        sleep_time = 8  

        for i in range(num_points + 1):  
            if i == 0:
                sleep_time = 8
            else:
                sleep_time = 2
            angle = i * angle_step
            new_pose = self.get_current_pose()
            
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

            print('PUBLISH POSE: \n', new_pose.pose.position)
            self.pose_publisher.publish(new_pose)
            dist = self.calc_travel_dist(center_pose, new_pose)
            a7_speed = self.get_a7_velocity()
            print('Estimated time: ', dist / a7_speed)
            rospy.sleep(sleep_time)
            current_pose = self.get_current_pose()
            print('END POSE: \n', current_pose.pose.position)

        print('Circle drawing complete.')


        


    def display_img_stream(self):
        data = rospy.wait_for_message("/imfusion/imgs", Image)
        cv_image = self.msg2img(data)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)




    def run(self):
        # data = rospy.wait_for_message("/imfusion/imgs", Image)
        # cv_image = self.msg2img(data)
    

        # offline debug
        cv_image = cv2.imread('/home/liming/Documents/datasets/results/phantom/phantom_transverse_02/filtered/Catheter_0/0.png', cv2.IMREAD_GRAYSCALE)
        cv_image = cv2.resize(cv_image, (img_width_px, img_height_px), interpolation=cv2.INTER_AREA)

        l = label(cv_image)
        biggest_coh_area = (l == np.bincount(l.ravel())[1:].argmax() + 1).astype(int)
        biggest_coh_area *= 255
        center_x_px, center_y_px, width, height, aorta_size = self.calc_center(biggest_coh_area)
        # plot center point on the Image
        # cv2.circle(biggest_coh_area, (int(center_x_px), int(center_y_px)), 5, (255, 255, 255), -1)

        return center_x_px, center_y_px, width, height, aorta_size, biggest_coh_area


def main():
    ao = AortaControl()
    rospy.init_node('aorta_control', anonymous=True)  # , disable_signals=True)


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
    # ao.get_current_pose()
    # ao.set_init_pose()
    while not rospy.is_shutdown():

        """         
        while True:
            finished = False
            ao.init = True
            save_last_n_pos = []
            while not finished:
                center_x_px, center_y_px, width, height, aorta_size, segm = ao.run()
                # print('save_last_n_pos: ', save_last_n_pos)
                if len(save_last_n_pos) < NR_LAST_POS:
                    save_last_n_pos.append((center_x_px, center_y_px))
                else:
                    print(' len(save_last_n_pos): ', len(save_last_n_pos))
                    save_last_n_x_pos = [tuple[0] for tuple in save_last_n_pos]
                    if np.std(save_last_n_x_pos) > THRESHOLD:
                        save_last_n_pos.pop(0)
                    else:
                        finished = True
                        segm = np.array(segm).astype('float32')
                        ao.image_pub_seg_stable.publish(ao.bridge.cv2_to_imgmsg(segm))
                        print('-------image_pub_seg_stable------')

            print('####################### FINISHED ###############################')
            print('save_last_n_pos: ', save_last_n_pos)

            ao.step += 1 
        """
            # input()   #print("============ Press `Enter` to move forward...")
        # ao.move_forward()
        ao.movement_dir_check(dist=0.05)
        # ao.run()
        # ao.display_img_stream()
        # ao.draw_circle(radius=0.1)

        # rospy.sleep(0.3)

        # quit the loop
        break

        # set_force_mode(cartesian_dof=3, desired_force=INIT_FORCE, desired_stiffness=STIFFNESS,
        #                    max_deviation_pos=1000, max_deviation_rotation_in_deg=1000)
        
        set_position_control_mode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
