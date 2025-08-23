from ultralytics import YOLO
import torch
import numpy as np
import time
import os

script_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(script_dir,'../weights/yolov8n.engine')

def run_benchmark():
    # 1. CUDA 환경 확인
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is not available. Exiting. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return
    
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")

    # 2. 모델 로드 (.pt 파일에서 직접 로드하여 engine 빌드 유도)
    print(f"Loading model from '{model_path}'...")
    try:
        model=YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. 더미 이미지 생성
    dummy_image=np.zeros((480, 640, 3), dtype=np.uint8)

    # 4. 워밍업 (첫 추론은 초기화 때문에 느림)
    print("Running warm-up inference...")
    model.predict(dummy_image, half=True,imgsz=640,device=0,verbose=False)
    
    # 5. 성능 벤치마크
    print("Starting benchmark...")
    num_iterations=300
    start_time=time.time()
    for _ in range(num_iterations):
        model.predict(dummy_image,half=True,imgsz=640,device=0,verbose=False)
    end_time=time.time()

    total_time=end_time-start_time
    avg_time_per_frame=total_time/num_iterations
    fps=1/avg_time_per_frame

    print("\n--- Inference Benchmark Results ---")
    print(f"Total time for {num_iterations} frames: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f} 🚀")

if __name__=='__main__':
    run_benchmark()