"""
YOLO model implementation for KITTI object detection.
Uses YOLOv5 as the base model with transfer learning.
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import YOLO_CONFIG, KITTI_CLASSES

class KITTIYOLOModel:
    """
    YOLO model wrapper for KITTI object detection.
    Handles loading pretrained models and adapting to KITTI classes.
    """
    def __init__(self, num_classes: int = len(KITTI_CLASSES) - 1):
        """
        Initialize YOLO model.
        
        Args:
            num_classes: Number of object classes to detect
        """
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self):
        """
        Build YOLO model architecture.
        Uses YOLOv5 pretrained model and adapts to KITTI classes.
        """
        print(f"Building YOLO model on device: {self.device}")
        print(f"Using YOLOv5 {YOLO_CONFIG['model_type']} with {self.num_classes} classes")
        
        # Import here to prevent loading YOLOv5 at module import time
        import torch.hub
        
        # Determine model size from config
        model_type = YOLO_CONFIG['model_type']
        pretrained = YOLO_CONFIG['pretrained']
        
        try:
            # Load model from torch hub
            self.model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=pretrained)
            
            # Adapt model to our number of classes
            if self.num_classes != 80:  # Default COCO classes
                self.model.nc = self.num_classes
                # Recreate the detection layer with correct number of classes
                self.model.model[-1].nc = self.num_classes
                # Reset the detection heads to retrain them
                for layer in self.model.model[-1].m:
                    if hasattr(layer, 'nc'):
                        layer.nc = self.num_classes
                print(f"Adapted YOLOv5 model to {self.num_classes} classes")
            
            # Move model to device
            self.model.to(self.device)
            
            return self.model
            
        except Exception as e:
            print(f"Error building YOLO model: {e}")
            if 'torch.hub' in str(e):
                print("Consider downloading YOLOv5 manually:")
                print("git clone https://github.com/ultralytics/yolov5.git")
                print("cd yolov5")
                print("pip install -r requirements.txt")
                print("Then modify this code to load the local model")
            raise
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            # Import here to prevent loading YOLOv5 at module import time
            import torch.hub
            
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
                self.model.to(self.device)
                print(f"Model loaded from {path}")
                return self.model
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        else:
            print(f"Model path {path} does not exist")
            return None
    
    def configure_training(self, hyp: Optional[Dict] = None):
        """
        Configure training hyperparameters.
        
        Args:
            hyp: Dictionary of hyperparameters to override defaults
        """
        default_hyp = {
            'lr0': 0.01,              # Initial learning rate
            'lrf': 0.1,               # Final learning rate factor
            'momentum': 0.937,        # SGD momentum
            'weight_decay': 0.0005,   # Weight decay
            'warmup_epochs': 3.0,     # Warmup epochs
            'warmup_momentum': 0.8,   # Warmup momentum
            'warmup_bias_lr': 0.1,    # Warmup bias learning rate
            'box': 0.05,              # Box loss gain
            'cls': 0.5,               # Cls loss gain
            'cls_pw': 1.0,            # Cls BCELoss positive weight
            'obj': 1.0,               # Obj loss gain
            'obj_pw': 1.0,            # Obj BCELoss positive weight
            'iou_t': 0.20,            # IoU training threshold
            'anchor_t': 4.0,          # Anchor threshold
            'fl_gamma': 0.0,          # Focal loss gamma
            'hsv_h': 0.015,           # Image HSV-Hue augmentation
            'hsv_s': 0.7,             # Image HSV-Saturation augmentation
            'hsv_v': 0.4,             # Image HSV-Value augmentation
            'degrees': 0.0,           # Image rotation (+/- deg)
            'translate': 0.1,         # Image translation (+/- fraction)
            'scale': 0.5,             # Image scale (+/- gain)
            'shear': 0.0              # Image shear (+/- deg)
        }
        
        # Override defaults with provided hyperparameters
        if hyp:
            default_hyp.update(hyp)
        
        # Set hyperparameters in model
        if self.model is not None:
            self.model.hyp = default_hyp
            print("Training hyperparameters configured")
        else:
            print("Model not initialized yet")
            
        return default_hyp
        
    def prepare_data_config(self, train_path: str, val_path: str = None):
        """
        Prepare data configuration file for YOLOv5.
        
        Args:
            train_path: Path to training data YAML
            val_path: Path to validation data YAML (optional)
            
        Returns:
            Path to generated data config YAML
        """
        # Prepare class names without DontCare
        names = [k for k, v in KITTI_CLASSES.items() if v != -1]
        
        # Prepare data config
        data_config = {
            'path': os.path.dirname(train_path),
            'train': train_path,
            'val': val_path if val_path else train_path,
            'nc': self.num_classes,
            'names': names
        }
        
        # Save to YAML
        config_path = os.path.join(os.path.dirname(train_path), 'data_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)
        
        print(f"Data config saved to {config_path}")
        return config_path
    
    def train(self, data_yaml: str, epochs: int = 100, batch_size: int = 16, 
              img_size: List[int] = [640, 640], save_dir: str = './runs/train'):
        """
        Train the YOLOv5 model.
        
        Args:
            data_yaml: Path to data configuration YAML file
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size [height, width]
            save_dir: Directory to save results
            
        Returns:
            Path to best trained weights
        """
        if self.model is None:
            print("Model not initialized. Building model first...")
            self.build_model()
        
        # Set training parameters
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=img_size,
            save_dir=save_dir
        )
        
        # Return path to best weights
        return os.path.join(save_dir, 'weights', 'best.pt')
    
    def predict(self, image_path: str, conf_thres: float = None, iou_thres: float = None):
        """
        Run inference on an image.
        
        Args:
            image_path: Path to image file
            conf_thres: Confidence threshold (0-1)
            iou_thres: IoU threshold for NMS (0-1)
            
        Returns:
            Detection results
        """
        if self.model is None:
            print("Model not initialized")
            return None
        
        # Use config values if not provided
        if conf_thres is None:
            conf_thres = YOLO_CONFIG['conf_thres']
        if iou_thres is None:
            iou_thres = YOLO_CONFIG['iou_thres']
        
        # Run inference
        results = self.model(image_path, size=IMG_SIZE[0], conf=conf_thres, iou=iou_thres)
        return results