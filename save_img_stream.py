#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

# Set the directory where images will be saved
save_directory = "/media/steven_kuang/My Passport/Work/dataset/transverse_cactuss_retrain/trainA/test"
image_topic = "/imfusion/imgs"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Initialize the CvBridge class
bridge = CvBridge()

# Counter for image filenames
def get_initial_counter():
    existing_files = [f for f in os.listdir(save_directory) if f.endswith('.png')]
    if not existing_files:
        return 0
    existing_numbers = [int(f.split('.')[0]) for f in existing_files]
    return max(existing_numbers) + 1

counter = get_initial_counter()

# Interval for saving frames
save_interval = 30  # Set this to your desired interval

# Variable to control recording state
recording = False
current_frame = None
exit_flag = False

def image_callback(msg):
    global counter, recording, current_frame
    try:
        # Convert the ROS Image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        current_frame = cv_image

        if recording:
            # Only save the frame if it meets the interval condition
            if counter % save_interval == 0:
                # Construct the file path
                file_path = os.path.join(save_directory, f"{counter // save_interval}.png")

                # Save the image
                cv2.imwrite(file_path, cv_image)

                rospy.loginfo(f"Saved image {file_path}")

            # Increment the counter
            counter += 1

    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")

def key_listener():
    global recording, current_frame, exit_flag
    cv2.namedWindow("Image Saver")  # Create a named window

    while not rospy.is_shutdown() and not exit_flag:
        if current_frame is not None:
            # Display the current frame
            display_frame = current_frame.copy()
            
            # Add the recording status to the frame
            status_text = "Recording: ON" if recording else "Recording: OFF"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2)
            
            # Add the instruction text to the frame
            instruction_text = "Press 's' to save"
            cv2.putText(display_frame, instruction_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            quit_text = "Press 'q' to quit"
            cv2.putText(display_frame, quit_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow("Image Saver", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = not recording
            if recording:
                rospy.loginfo("Recording started")
            else:
                rospy.loginfo("Recording stopped")
        elif key == ord('q'):
            exit_flag = True
            # cv2.destroyAllWindows()

def main():
    rospy.init_node('image_saver', anonymous=True)
    
    # Subscribe to the imagqe topic
    rospy.Subscriber(image_topic, Image, image_callback)
    
    # Start key listener in a separate thread
    import threading
    key_listener_thread = threading.Thread(target=key_listener)
    key_listener_thread.start()
    
    key_listener_thread.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
