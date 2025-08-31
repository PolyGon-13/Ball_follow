import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    yolo_project_package_dir=get_package_share_directory('yolo_project_package')

    cam_x,cam_y,cam_z='0.04','0.0','0.15'
    cam_yaw,cam_pitch,cam_roll='0.0','0.0','0.0'

    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            parameters=[{
                'align_depth.enable': True,
                'color_width':640,
                'color_height': 480,
                'color_fps': 30.0,
                'depth_width': 640,
                'depth_height': 480,
                'depth_fps': 30.0,
            }]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_robot_tf_publisher',
            arguments=[cam_x,cam_y,cam_z,cam_yaw,cam_pitch,cam_roll,'base_link','camera_link']
        ),

        Node(
            package='yolo_project_package',
            executable='yolo_node',
            name='yolo_node'
        ),

        Node(
            package='yolo_project_package',
            executable='vel_pub_node',
            name='vel_pub_node'
        )
    ])