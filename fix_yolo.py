# fix_yolo.py
import os
import shutil
from ultralytics import YOLO

# Download YOLOv5s model
model = YOLO('yolov5s.pt')

# Create directory if it doesn't exist
os.makedirs('/Users/tanishyadav/.cache/torch/hub/ultralytics_yolov5_master', exist_ok=True)

# Print success message
print("YOLOv5 model initialized and cached successfully!")