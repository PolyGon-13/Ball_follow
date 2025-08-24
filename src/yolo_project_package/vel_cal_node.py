import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
import cv2

class VelocityEstimatorNode(Node):
    def __init__(self):
        super().__init__('velocity_estimator_node')
        self.bridge=CvBridge()
        self.K_matrix=None
        self.tracked_objects={}

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.camera_info_sub=self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.camera_info_callback,10)

        detection_sub=message_filters.Subscriber(self,Detection2DArray,'/yolo/detections',qos_profile=qos_profile)
        depth_sub=message_filters.Subscriber(self,Image,'/camera/camera/aligned_depth_to_color/image_raw',qos_profile=qos_profile)
        color_sub=message_filters.Subscriber(self,Image,'/camera/camera/color/image_raw',qos_profile=qos_profile)

        self.ts=message_filters.TimeSynchronizer([detection_sub,depth_sub,color_sub],30)
        self.ts.registerCallback(self.synced_callback)

    def camera_info_callback(self,msg):
        if self.K_matrix is None:
            self.K_matrix=np.array(msg.k).reshape((3,3))
            self.destroy_subscription(self.camera_info_sub)

    def synced_callback(self,detection_msg,depth_msg,color_msg):
        if self.K_matrix is None:
            self.get_logger().warn('Camera intrinsics not available yet. Skipping frame.')
            return
        
        try:
            depth_image=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='16UC1')
            color_image=self.bridge.imgmsg_to_cv2(color_msg,desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return
        
        fx,fy=self.K_matrix[0,0],self.K_matrix[1,1]
        cx,cy=self.K_matrix[0,2],self.K_matrix[1,2]

        for detection in detection_msg.detections:
            bbox=detection.bbox
            u=int(bbox.center.position.x*depth_image.shape[1])
            v=int(bbox.center.position.y*depth_image.shape[0])

            if not (0<=v<depth_image.shape[0] and 0<=u<depth_image.shape[1]):
                continue

            depth_mm=depth_image[v,u]
            if depth_mm==0:
                continue

            Z=float(depth_mm)/1000.0
            X=(u-cx)*Z/fx
            Y=(v-cy)*Z/fy

            current_pos=np.array([X,Y,Z])
            current_time=detection_msg.header.stamp.sec+detection_msg.header.stamp.nanosec*1e-9
            class_id=detection.results[0].hypothesis.class_id

            speed=0.0
            if class_id in self.tracked_objects:
                last_pos=self.tracked_objects[class_id]['last_pos']
                last_time=self.tracked_objects[class_id]['last_time']

                dt=current_time-last_time

                if dt>0:
                    velocity=(current_pos-last_pos)/dt
                    speed=np.linalg.norm(velocity)

            self.tracked_objects[class_id]={'last_pos':current_pos,'last_time':current_time}

            w_px = int(bbox.size_x * color_image.shape[1])
            h_px = int(bbox.size_y * color_image.shape[0])
            x1 = u - w_px // 2
            y1 = v - h_px // 2
            x2 = u + w_px // 2
            y2 = v + h_px // 2

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            speed_text = f"Speed: {speed:.2f} m/s"
            text_pos = (x1, y1 - 10)
            cv2.putText(color_image, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Velocity Visualization", color_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node=VelocityEstimatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()