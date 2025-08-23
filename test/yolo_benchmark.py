from ultralytics import YOLO
import torch
import numpy as np
import time
import os

script_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(script_dir,'../weights/yolov8n.engine')

def run_benchmark():
    # CUDA 환경 확인
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is not available. Exiting. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return
    
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    try:
        model=YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 더미 이미지 생성
    dummy_image=np.zeros((480, 640, 3), dtype=np.uint8)

    model.predict(dummy_image, half=True,imgsz=640,device=0,verbose=False)
    
    # 성능 벤치마크
    num_iterations=300
    start_time=time.time()
    for _ in range(num_iterations):
        model.predict(dummy_image,half=True,imgsz=640,device=0,verbose=False)
    end_time=time.time()

    total_time=end_time-start_time
    avg_time_per_frame=total_time/num_iterations
    fps=1/avg_time_per_frame

    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_iterations} frames: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")

if __name__=='__main__':
    run_benchmark()