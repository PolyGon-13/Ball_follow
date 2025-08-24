import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy

class CoordinateCalculatorNode(Node):
    def __init__(self):
        super().__init__('coordinate_calculator_node')
        self.bridge=CvBridge()
        self.K_matrix=None

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.camera_info_sub=self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.camera_info_callback,10)
        self.detection_sub=message_filters.Subscriber(self,Detection2DArray,'/yolo/detections',qos_profile=qos_profile)
        self.depth_sub=message_filters.Subscriber(self,Image,'/camera/camera/aligned_depth_to_color/image_raw',qos_profile=qos_profile)

        self.ts=message_filters.TimeSynchronizer([self.detection_sub,self.depth_sub],30)
        self.ts.registerCallback(self.synced_callback)

    def camera_info_callback(self,msg):
        if self.K_matrix is None:
            self.K_matrix=np.array(msg.k).reshape((3,3))
            self.destroy_subscription(self.camera_info_sub)

    def synced_callback(self,detection_msg,depth_msg):
        if self.K_matrix is None:
            self.get_logger().warn('Camera intrinsics not available yet. Skipping frame.')
            return
        
        try:
            depth_image=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return
        
        fx,fy=self.K_matrix[0,0],self.K_matrix[1,1]
        cx,cy=self.K_matrix[0,2],self.K_matrix[1,2]

        for detection in detection_msg.detections:
            bbox=detection.bbox
            x_norm=bbox.center.position.x
            y_norm=bbox.center.position.y

            u=int(x_norm*depth_image.shape[1])
            v=int(y_norm*depth_image.shape[0])

            depth_mm=depth_image[v,u]

            if depth_mm==0:
                self.get_logger().warn(f'Depth at ({u},{v}) is zero. Skipping frame.')
                continue

            Z=float(depth_mm)/1000.0

            X=(u-cx)*Z/fx
            Y=(v-cy)*Z/fy

            self.get_logger().info(f'Orange 3D Coordinates (m) : [X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}]')

def main(args=None):
    rclpy.init(args=args)
    node=CoordinateCalculatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()