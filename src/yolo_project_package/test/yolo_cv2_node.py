import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import QoSProfile,QoSReliabilityPolicy,QoSHistoryPolicy
import os
from vision_msgs.msg import Detection2DArray,Detection2D,BoundingBox2D,ObjectHypothesisWithPose

# BoundingBox2D : 2차원 이미지에서 검출된 사각형 영역을 정의하는 메시지
# ObjectHypothesisWithPose : 검출된 객체의 클래스와 신뢰도, 필요시 위치를 담는 메시지

pt_path='weights/yolov8n.pt'
engine_path='weights/yolov8n.engine'

class YoloTestNode(Node):
    def __init__(self):
        super().__init__('yolo_cv2_node')

        if not os.path.exists(engine_path):
            model_builder=YOLO(pt_path)
            model_builder.export(format='engine',half=True,imgsz=640,device=0)
            
        self.model=YOLO(engine_path,task='detect')
        self.bridge=CvBridge()

        qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,history=QoSHistoryPolicy.KEEP_LAST,depth=1)

        self.subscription=self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,qos_profile)
        self.detection_publisher=self.create_publisher(Detection2DArray,'/yolo/detections',10)

    def image_callback(self,msg):
        try:
            cv_image=self.bridge.imgmsg_to_cv2(msg,'bgr8')
            results=self.model(cv_image,verbose=False)

            annotated_frame=results[0].plot()
            cv2.imshow("YOLO_Realsense_ROS",annotated_frame)
            cv2.waitKey(1)

            detections_msg=Detection2DArray()
            detections_msg.header=msg.header
            for box in results[0].boxes:
                # boxes가 가진 정보
                # xyxy : 왼쪽 위, 오른쪽 아래 좌표
                # xywh : 중심좌표(x,y), width, height
                # xywhn : 중심좌표, width, height를 정규환된 값(0~1)으로
                # cls : 클래스 ID
                # conf : 신뢰도
                detection=Detection2D()
                x_center,y_center,width,height=box.xywhn[0]

                # yolo 결과를 ROS 메시지 형식으로 변환
                bbox=BoundingBox2D()
                bbox.center.position.x=float(x_center)
                bbox.center.position.y=float(y_center)
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
    yolo_node=YoloTestNode()

    try:
        rclpy.spin(yolo_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()