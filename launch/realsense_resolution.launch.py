from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='camera',
                namespace='camera',
                parameters=[{
                    'align_depth.enable': True, # Depth를 컬러 기준으로 맞춤
                    'color_width':640, # 컬러 해상도 낮춤 (720p -> 480p)
                    'color_height': 480,
                    'color_fps': 30.0, # fps 증가
                    'depth_width': 640, # Depth 해상도 낮춤
                    'depth_height': 480,
                    'depth_fps': 30.0,
                }]
            ),

            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='camera_to_robot_publisher',
                arguments=['0.05','0','0.1','0','0','0','base_link','camera_link']
            )
    ])