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
                    'enable_accel': True, # 선형 가속도 데이터 발행
                    'enable_gyro': True, # 각속도 데이터 발행
                    'unite_imu_method': 2, # 나뉘어 있는 가속도계와 자이로 데이터를 하나의 통합된 IMU 토픽으로 합쳐주는 방법 지정 (여기서는 선형 보간법 사용)
                    'align_depth.enable': True,
                    'color_width':640,
                    'color_height': 480,
                    'color_fps': 15.0,
                    'depth_width': 640,
                    'depth_height': 480,
                    'depth_fps': 15.0,
                }]
            )
    ])