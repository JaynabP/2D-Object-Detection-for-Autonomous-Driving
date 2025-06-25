"""
Configuration file for KITTI 2D Object Detection Project.
Contains paths, model parameters, and training settings.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'kitti')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'checkpoints')

# KITTI dataset paths
KITTI_TRAIN_IMAGES = os.path.join(DATA_ROOT, 'training', 'image_2')
KITTI_TRAIN_LABELS = os.path.join(DATA_ROOT, 'training', 'label_2')
KITTI_TRAIN_CALIB = os.path.join(DATA_ROOT, 'training', 'calib')
KITTI_TEST_IMAGES = os.path.join(DATA_ROOT, 'testing', 'image_2')

# Dataset settings
KITTI_CLASSES = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1
}

# Image settings
IMG_SIZE = (608, 608)  # Width, Height
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Data split
VAL_RATIO = 0.2
RANDOM_SEED = 42

# Data augmentation settings
AUGMENT = True
AUG_PARAMS = {
    'horizontal_flip': 0.5,
    'brightness_contrast': 0.3,
    'blur_limit': 3,
    'rotate_limit': 10,
}

# Model settings (YOLOv5)
YOLO_CONFIG = {
    'model_type': 'yolov5s',  # Options: yolov5s, yolov5m, yolov5l, yolov5x
    'pretrained': True,
    'conf_thres': 0.25,   # Confidence threshold
    'iou_thres': 0.45,    # NMS IoU threshold
    'max_det': 300,       # Maximum detections per image
    'agnostic_nms': False # Class-agnostic NMS
}

# Training parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
LR_SCHEDULER = 'cosine'  # Options: step, cosine
WARMUP_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 10

# Evaluation settings
EVAL_FREQUENCY = 1  # Epochs between evaluations

# Logging and saving
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
SAVE_BEST_ONLY = True