import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import os
import threading
import queue # queue와 threading 임포트

# engine_path='weights/yolov8n.engine' # 경로 확인
engine_path='/home/user/ros2_ws/src/yolo_project_package/weights/yolov8n.engine' # <-- 실제 절대 경로로 수정해주세요!

class YoloEngineNode(Node):
    def __init__(self):
        super().__init__('yolo_engine_node')

        # .engine 파일이 있는지 확인 (없으면 생성은 yolo export로 미리 해두는 것을 권장)
        if not os.path.exists(engine_path):
            self.get_logger().error(f"Engine file not found at {engine_path}")
            # 여기서 프로그램을 종료하거나, 이전처럼 .pt에서 생성하는 로직을 넣을 수 있습니다.
            # 예: temp_model = YOLO('weights/yolov8n.pt').export(format='engine', half=True)
            return

        self.model = YOLO(engine_path, task='detect')
        self.bridge = CvBridge()
        
        # 스레드 간 데이터 전달을 위한 큐 생성 (최신 프레임 1개만 저장)
        self.image_queue = queue.Queue(maxsize=1)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile
        )

        self.publisher = self.create_publisher(Image, '/yolo/annotated_image', 10)

        # YOLO 처리를 담당할 별도의 스레드 생성 및 시작
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def image_callback(self, msg):
        # 콜백 함수는 이미지를 받아서 큐에 넣는 역할만 빠르게 수행
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 큐가 가득 차 있으면 오래된 프레임을 버리고 새 프레임을 넣음 (Non-blocking)
            if self.image_queue.full():
                self.image_queue.get_nowait()
            self.image_queue.put_nowait(cv_image)
        except queue.Full:
            pass
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def processing_loop(self):
        # 이 함수는 별도의 스레드에서 계속 실행됨
        while rclpy.ok():
            try:
                # 큐에서 처리할 이미지를 가져옴 (이미지가 올 때까지 대기)
                cv_image = self.image_queue.get(timeout=1.0)

                # YOLO 추론 및 시각화 (시간이 오래 걸리는 작업)
                results = self.model(cv_image, verbose=False) # half=True, imgsz=640 등은 engine 생성 시 이미 적용됨
                annotated_frame = results[0].plot()

                # 결과 이미지를 ROS 메시지로 변환하여 발행
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
                annotated_msg.header.stamp = self.get_clock().now().to_msg()
                self.publisher.publish(annotated_msg)

            except queue.Empty:
                # 큐가 비어있으면 아무 작업도 하지 않음
                continue
            except Exception as e:
                self.get_logger().error(f'Error in processing_loop: {e}')

def main(args=None):
    rclpy.init(args=args)
    yolo_engine_node = YoloEngineNode()
    rclpy.spin(yolo_engine_node)
    yolo_engine_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()