import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import Twist,PointStamped
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
import cv2
import tf2_ros
from tf2_ros import Buffer,TransformListener,TransformException
from tf2_geometry_msgs import do_transform_point

class VelocityEstimatorNode(Node):
    def __init__(self):
        super().__init__('vel_pub_node')
        self.bridge=CvBridge()

        self.K_matrix=None

        # PID 제어 변수
        self.last_linear_error=0.0
        self.integral_linear_error=0.0
        self.last_angular_error=0.0
        self.integral_angular_error=0.0

        self.tf_buffer=Buffer()
        self.tf_listener=TransformListener(self.tf_buffer,self)

        self.cmd_vel_publisher=self.create_publisher(Twist,'/cmd_vel',10)

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.camera_info_sub=self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.camera_info_callback,10)

        # 3개의 서로 다른 토픽을 동시에 구독하기 위해 message_filters.Subscriber 생성
        detection_sub=message_filters.Subscriber(self,Detection2DArray,'/yolo/detections',qos_profile=qos_profile)
        depth_sub=message_filters.Subscriber(self,Image,'/camera/camera/aligned_depth_to_color/image_raw',qos_profile=qos_profile)
        color_sub=message_filters.Subscriber(self,Image,'/camera/camera/color/image_raw',qos_profile=qos_profile)
        # 3개의 Subscriber를 시간 동기화 장치에 등록 -> 타임스탬프가 거의 동일한 메시지 3개가 한 세트로 도착할 때까지 기다렸다가, 세 메시지를 묶어서 콜백함수로 전달
        self.ts=message_filters.TimeSynchronizer([detection_sub,depth_sub,color_sub],30)
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("Velocity Estimator Node with TurtleBot Control has been started.")

    def camera_info_callback(self,msg):
        if self.K_matrix is None:
            self.K_matrix=np.array(msg.k).reshape((3,3))
            self.destroy_subscription(self.camera_info_sub)

    def synced_callback(self,detection_msg,depth_msg,color_msg):
        if self.K_matrix is None:
            self.get_logger().warn('Camera intrinsics not available yet.')
            return
        
        if not detection_msg.detections:
            stop_twist=Twist()
            self.cmd_vel_publisher.publish(stop_twist)
            self.integral_linear_error=0.0
            self.integral_angular_error=0.0
            return
        
        try:
            color_image=self.bridge.imgmsg_to_cv2(color_msg,desired_encoding='bgr8')
            depth_image=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return
        
        fx,fy=self.K_matrix[0,0],self.K_matrix[1,1] # 초점거리 (3D 공간의 물체가 2D 이미지 센서에 얼마나 크거나 작게 맺히는지를 결정하는 배율)
        cx,cy=self.K_matrix[0,2],self.K_matrix[1,2] # 주점 (2D 이미지 픽셀 좌표계의 진짜 원점이 어딘지 알려줌)
        
        detection=detection_msg.detections[0]
        bbox=detection.bbox
        u=int(bbox.center.position.x*depth_image.shape[1]) # shape[1]에는 이미지의 너비 값
        # 전체길이가 128이고 x가 0.75이면 원하는 위치는 75% 지점인 960 
        v=int(bbox.center.position.y*depth_image.shape[0]) # shape[0]에는 이미지의 높이 값

        if not (0<=v<depth_image.shape[0] and 0<=u<depth_image.shape[1]):
            return

        depth_mm=depth_image[v,u]
        if depth_mm==0:
            return

        Z_cam=float(depth_mm)/1000.0 # 리얼센스는 거리값을 mm 단위로 알려줌
        # 역투영
        X_cam=(u-cx)*Z_cam/fx
        Y_cam=(v-cy)*Z_cam/fy

        try:
            point_in_camera_frame=PointStamped()
            point_in_camera_frame.header=color_msg.header
            point_in_camera_frame.point.x=X_cam
            point_in_camera_frame.point.y=Y_cam
            point_in_camera_frame.point.z=Z_cam

            target_frame='base_link'

            transform=self.tf_buffer.lookup_transform(target_frame,color_msg.header.frame_id,rclpy.time.Time())
            point_in_base_frame=do_transform_point(point_in_camera_frame,transform)

            X_base=point_in_base_frame.point.x
            Y_base=point_in_base_frame.point.y

            linear_p_gain,linear_i_gain,linear_d_gain=0.6,0.0,0.1 # Kp, Ki, Kd
            angular_p_gain,angular_i_gain,angular_d_gain=0.7,0.0,0.2
            target_distance=0.3 # 로봇과 공의 목표거리
            linear_dead_zone,angular_dead_zone=0.05,0.03

            twist_msg=Twist()

            linear_error=X_base-target_distance # 목표거리와의 오차
            linear_error_diff=linear_error-self.last_linear_error # 오차의 변화량

            if abs(linear_error)>linear_dead_zone:
                self.integral_linear_error+=linear_error # 과거의 오차를 더함

                p_term=linear_p_gain*linear_error
                i_term=linear_i_gain*self.integral_linear_error
                d_term=linear_d_gain*linear_error_diff
                
                twist_msg.linear.x=p_term+i_term+d_term
            else:
                self.integral_linear_error=0.0
                twist_msg.linear.x=0.0


            angular_error=Y_base
            angular_error_diff=angular_error-self.last_angular_error

            if abs(angular_error)>angular_dead_zone:
                self.integral_angular_error+=angular_error

                p_term=angular_p_gain*angular_error
                i_term=angular_i_gain*self.integral_angular_error
                d_term=angular_d_gain*angular_error_diff

                twist_msg.angular.z=p_term+i_term+d_term
            else:
                self.integral_angular_error=0.0
                twist_msg.angular.z=0.0
            
            self.cmd_vel_publisher.publish(twist_msg)

            self.last_linear_error=linear_error
            self.last_angular_error=angular_error

            self.get_logger().info(f'Ball at Robot frame: X={X_base:.2f}, Y={Y_base:.2f} | Sent cmd_vel: linear_x={twist_msg.linear.x:.2f}, angular_z={twist_msg.angular.z:.2f}')

        except TransformException as ex:
            self.get_logger().warn(f'Could not transform: {ex}')

        # 전체 길이에서 박스가 어느정도를 차지하고 있는지 계산
        w_px,h_px=int(bbox.size_x*color_image.shape[1]),int(bbox.size_y*color_image.shape[0])

        x1=u-w_px//2 # 왼쪽 끝 x좌표
        y1=v-h_px//2 # 위쪽 끝 x좌표
        x2=u+w_px//2 # 오른쪽 끝 x좌표
        y2=v+h_px//2 # 아래쪽 끝 x좌표

        cv2.rectangle(color_image,(x1,y1),(x2,y2),(255,0,0),2)
        
        #speed_text=f"Speed: {speed:.2f} m/s"
        #text_pos=(x1,y1-10) # 왼쪽 위 끝에서 위로 10픽셀 증가 (컴퓨터 이미지의 좌표계에서는 오른쪽으로 갈수록 x값 증가, 아래로 갈수록 y값이 증가)
        #cv2.putText(color_image,speed_text,text_pos,cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        cv2.imshow("Velocity Visualization",color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node=VelocityEstimatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_twist=Twist()
        node.cmd_vel_publisher.publish(stop_twist)
        node.get_logger().info('Shutting down and stopping the robot.')

        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()