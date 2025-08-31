import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
import os
from vision_msgs.msg import Detection2DArray,Detection2D,BoundingBox2D,ObjectHypothesisWithPose

pt_path='weights/yolov8n.pt'
engine_path='weights/yolov8n.engine'

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        if not os.path.exists(engine_path):
            model_builder=YOLO(pt_path)
            model_builder.export(format='engine',half=True,imgsz=640,device=0)
            
        self.model=YOLO(engine_path,task='detect')
        self.bridge=CvBridge()

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.subscription=self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,qos_profile)
        self.detection_publisher=self.create_publisher(Detection2DArray,'/yolo/detections',qos_profile)

    def image_callback(self,msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
            results=self.model(cv_image,verbose=False,conf=0.20)

            #annotated_frame=results[0].plot()
            #cv2.imshow("YOLO_Realsense_ROS",annotated_frame)
            #cv2.waitKey(1)

            detections_msg=Detection2DArray()
            detections_msg.header=msg.header
            for box in results[0].boxes:
                if int(box.cls)==49:
                    detection=Detection2D()
                    x_center,y_center,width,height=box.xywhn[0]

                    # 해상도 독립성을 위해 수치가 아닌 비율로 값을 저장
                    bbox=BoundingBox2D()
                    # 이미지의 가로세로 길이를 100%라고 할 때, 사각형의 중심이 몇 % 지점에 있는지 알려주는 비율 값
                    bbox.center.position.x=float(x_center)
                    bbox.center.position.y=float(y_center)
                    # 사각형의 너비와 높이를 이미지 전체 크기에 대한 비율로 저장
                    bbox.size_x=float(width)
                    bbox.size_y=float(height)
                    detection.bbox=bbox

                    hypothesis=ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id=str(int(box.cls))
                    hypothesis.hypothesis.score=float(box.conf)
                    detection.results.append(hypothesis)

                    detections_msg.detections.append(detection)

            self.detection_publisher.publish(detections_msg)
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    yolo_node=YoloDetectionNode()

    try:
        rclpy.spin(yolo_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_node.destroy_node()
        rclpy.shutdown()
        #cv2.destroyAllWindows()

if __name__=='__main__':
    main()