import rospy
from std_msgs.msg import Float64MultiArray
from iiwa_msgs.msg import CartesianPose
import numpy as np
import open3d as o3d
import os
import threading
from scipy.spatial.transform import Rotation as R

FBG_TOPIC = '/matrix_data'
POSE_TOPIC = '/iiwa/state/CartesianPose'
MESH_PATH = '../../INR_3DUS/sampled_pointsstable1.ply'

class FBGWireReconstructor:
    def __init__(self):
        # Set environment variable to avoid potential GLFW issues
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        
        rospy.init_node('fbg_reconstructor', anonymous=True)
        rospy.Subscriber(FBG_TOPIC, Float64MultiArray, self.callback_fbg, queue_size=1)
        rospy.Subscriber(POSE_TOPIC, CartesianPose, self.callback_tip_pose, queue_size=1)
        
        self.pcd = o3d.geometry.PointCloud()
        self.data_lock = threading.Lock()

        self.raw_movement = False
        self.load_mesh = False
    

        self.fbg_data_received = False
        self.aorta_mesh = None
        if self.load_mesh:
            self.aorta_mesh = AortaMesh(MESH_PATH)


        self.tip_position = np.array([0.0, 0.0, 0.0])
        self.tip_orientation = R.from_quat([0.0, 0.0, 0.0, 1.0])  # identity 
        self.tip_last_orientation = R.from_quat([0.0, 0.0, 0.0, 1.0])

        # Create an initial catheter
        self.init_catheter_length = 0.2
        self.initialize_static_catheter()

        # Visualize in a separate thread for better performance
        self.vis_thread = threading.Thread(target=self.visualizer_thread, daemon=True)
        self.vis_thread.start()

    def visualizer_thread(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)
        
        # add aorta mesh
        if self.aorta_mesh:
            self.vis.add_geometry(self.aorta_mesh.mesh)
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(coord_frame)

        # Initial view parameters
        view_control = self.vis.get_view_control()
        view_control.set_front([0.0, -1.0, 0.0])
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(2)

        while not rospy.is_shutdown():
            with self.data_lock:
                if len(self.pcd.points) > 0:
                    self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
        self.vis.destroy_window()

    def create_init_catheter(self, length):
        # Create a straight line representing the static catheter of given length
        z_coords = np.linspace(0, length, num=1900)  # Adjust resolution as needed
        points = np.vstack((np.zeros_like(z_coords), np.zeros_like(z_coords), -z_coords)).T
        return points

    def create_init_colors(self, num_points):
        # Create initial colors for the catheter
        colors = np.tile([0.2, 0.2, 0.2], (num_points, 1))
        colors[:min(100, num_points)] = [1, 0, 0]  # Set the first 100 points to red
        return colors

    def initialize_static_catheter(self):
        # Initialize the catheter with static points and colors
        initial_points = self.create_init_catheter(self.init_catheter_length)
        initial_colors = self.create_init_colors(len(initial_points))
        self.pcd.points = o3d.utility.Vector3dVector(initial_points)
        self.pcd.colors = o3d.utility.Vector3dVector(initial_colors)

    def callback_fbg(self, msg):
        data = np.array(msg.data)
        N = len(data) // 3
        if len(data) % 3 != 0:
            rospy.logwarn("Received data size is not divisible by 3, check input data format.")
            return

        # Reshape the data to [N, 3] -> x, y, z
        x_coords = data[:N]
        y_coords = data[N:2*N]
        z_coords = data[2*N:]
        points = np.vstack((x_coords, y_coords, z_coords)).T

        # Scale the points
        scale_factor = 1.0  
        points *= scale_factor

        if not self.raw_movement:
            points = points - points[-1]  # Translate tip to the origin
            points = points[::-1]  # Reverse the order of points so that the tip is fixed and the tail moves
            colors = np.tile([1, 0.706, 0], (N, 1))  # yellow
            colors[:min(100, N)] = [1, 0, 0]  # Set the last 100 points to red
        else:
            colors = np.tile([1, 0.706, 0], (N, 1))
            colors[-min(100, N):] = [1, 0, 0]

        with self.data_lock:
            rotated_points = self.tip_orientation.apply(points)
            points = rotated_points + self.tip_position
            # Update catheter
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.fbg_data_received = True

    def callback_tip_pose(self, msg):
        position = msg.poseStamped.pose.position
        orientation = msg.poseStamped.pose.orientation
        
        with self.data_lock:
            self.tip_position = np.array([position.x, position.y, position.z])
            self.tip_orientation = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
            # revert the last orientation
            rotated_points = self.tip_orientation.apply(self.tip_last_orientation.inv().apply(np.asarray(self.pcd.points)))
            translated_points =  rotated_points + self.tip_position - self.pcd.points[0]
            
            self.tip_last_orientation = self.tip_orientation
            # Update catheter
            self.pcd.points = o3d.utility.Vector3dVector(translated_points)

    def run(self):
        rospy.spin()

class AortaMesh:
    def __init__(self, file_path):
        # Load the mesh from file
        try:
            self.mesh = o3d.io.read_triangle_mesh(file_path)
            if hasattr(self.mesh, 'compute_vertex_normals'): self.mesh.compute_vertex_normals()
            # Preset transformation parameters (scale, rotation, translation)
            self.scale = 1
            self.translation = np.array([0.0, 0.0, 0.0])
            self.rotation = R.from_euler('xyz', [0.0, 0.0, 0.0])
            self.apply_transformations()
        except Exception as e:
            rospy.logwarn(f"Failed to load aorta mesh: {e}")
            self.mesh = None

    def apply_transformations(self):
        if self.mesh:
            self.mesh.scale(self.scale, center=self.mesh.get_center())
            rotation_matrix = self.rotation.as_matrix()
            self.mesh.rotate(rotation_matrix, center=self.mesh.get_center())
            self.mesh.translate(self.translation)

if __name__ == '__main__':
    try:
        reconstructor = FBGWireReconstructor()
        reconstructor.run()
    except rospy.ROSInterruptException:
        pass