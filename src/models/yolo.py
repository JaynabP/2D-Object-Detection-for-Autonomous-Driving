#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from pathlib import Path

class KITTIYOLOModel:
    """YOLO model adapted for KITTI dataset."""
    
    def __init__(self, num_classes=8, output_dir=None, device=None):
        """Initialize YOLOv5 model.
        
        Args:
            num_classes (int): Number of classes to detect
            output_dir (str): Directory to save model outputs
            device (str): Device to use ('cpu' or 'cuda')
        """
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = 'yolov5s'  # Default model type
        
    def build_model(self):
        """Initialize the YOLOv5 model."""
        try:
            print(f"Building YOLO model on device: {self.device}")
            print(f"Using YOLOv5 {self.model_type} with {self.num_classes} classes")
            
            # Import required functions from ultralytics package
            from ultralytics import YOLO
            
            # Load the model
            self.model = YOLO(f"yolov5{self.model_type[6:] if self.model_type.startswith('yolov5') else 's'}.pt")
            
            # Configure model for custom training with our number of classes
            self.model.overrides['nc'] = self.num_classes
            
            print(f"Model loaded successfully with {self.num_classes} classes")
            return True
        except Exception as e:
            print(f"Error building YOLO model: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_model(self, weights_path):
        """Load a pre-trained YOLOv5 model from weights file.
        
        Args:
            weights_path (str): Path to the weights file
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            print(f"Loading model from weights: {weights_path}")
            
            # Import required functions from ultralytics package
            from ultralytics import YOLO
            
            # Check if weights file exists
            if not os.path.exists(weights_path):
                print(f"Error: Weights file not found: {weights_path}")
                return False
                
            # Load the model with weights
            self.model = YOLO(weights_path)
            
            print(f"Model loaded successfully from {weights_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def configure_training(self):
        """Configure hyperparameters for training."""
        print("Configuring hyperparameters...")
        
        # Default hyperparameters for YOLOv5 training
        hyp = {
            'lr0': 0.01,              # initial learning rate
            'lrf': 0.1,               # final learning rate factor
            'momentum': 0.937,        # SGD momentum/Adam beta1
            'weight_decay': 0.0005,   # optimizer weight decay
            'warmup_epochs': 3.0,     # warmup epochs
            'warmup_momentum': 0.8,   # warmup initial momentum
            'warmup_bias_lr': 0.1,    # warmup initial bias lr
            'box': 0.05,              # box loss gain
            'cls': 0.5,               # cls loss gain
            'cls_pw': 1.0,            # cls BCELoss positive_weight
            'obj': 1.0,               # obj loss gain
            'obj_pw': 1.0,            # obj BCELoss positive_weight
            'iou_t': 0.20,            # IoU training threshold
            'anchor_t': 4.0,          # anchor-multiple threshold
            'fl_gamma': 0.0,          # focal loss gamma
            'hsv_h': 0.015,           # image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,             # image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,             # image HSV-Value augmentation (fraction)
            'degrees': 0.0,           # image rotation (+/- deg)
            'translate': 0.1,         # image translation (+/- fraction)
            'scale': 0.5,             # image scale (+/- gain)
            'shear': 0.0,             # image shear (+/- deg)
            'perspective': 0.0,       # image perspective (+/- fraction)
            'flipud': 0.0,            # image flip up-down (probability)
            'fliplr': 0.5,            # image flip left-right (probability)
            'mosaic': 1.0,            # image mosaic (probability)
            'mixup': 0.0,             # image mixup (probability)
            'copy_paste': 0.0         # segment copy-paste (probability)
        }
        
        print("Hyperparameters configured successfully")
        return hyp
            
    def train(self, data_yaml, epochs, batch_size, img_size=640, save_dir=None):
        """Train the YOLOv5 model."""
        if self.model is None:
            print("Model not initialized yet")
            return None
            
        try:
            print(f"Starting training for {epochs} epochs...")
            print(f"Using batch size {batch_size} and image size {img_size}")
            print(f"Dataset config: {data_yaml}")
            
            # Set up the proper command arguments
            project_dir = os.path.dirname(save_dir) if save_dir else os.path.dirname(self.output_dir)
            run_name = os.path.basename(save_dir) if save_dir else os.path.basename(self.output_dir)
            
            # Train using the correct syntax
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=self.device,
                project=project_dir,
                name=run_name,
                exist_ok=True
            )
            
            # Return path to best weights
            if hasattr(results, 'best') and results.best:
                best_weights = results.best
                print(f"Training complete. Best weights saved at {best_weights}")
                return str(best_weights)
            else:
                print("Training completed but couldn't find best weights path")
                return os.path.join(project_dir, run_name, 'weights/best.pt')
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate(self, data_yaml, weights_file=None, img_size=640, batch_size=16):
        """Evaluate the model on validation data."""
        if self.model is None and weights_file is None:
            print("Model not initialized and no weights provided")
            return None
            
        try:
            # Load weights if provided
            if weights_file:
                from ultralytics import YOLO
                self.model = YOLO(weights_file)
                
            # Run validation
            results = self.model.val(
                data=data_yaml,
                imgsz=img_size,
                batch=batch_size,
                device=self.device
            )
            
            print(f"Validation results: {results}")
            return results
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
            
    def predict(self, image_path, conf_thresh=0.25, iou_thresh=0.45, conf_thres=None):
        """Run inference on a single image.
        
        Args:
            image_path (str): Path to input image
            conf_thresh (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thresh (float, optional): IOU threshold. Defaults to 0.45.
            conf_thres (float, optional): Alternative name for conf_thresh for API compatibility
            
        Returns:
            Results: Prediction results
        """
        if self.model is None:
            print("Model not initialized")
            return None
            
        try:
            # Handle both parameter names (conf_thresh and conf_thres)
            confidence = conf_thres if conf_thres is not None else conf_thresh
            
            results = self.model(image_path, conf=confidence, iou=iou_thresh)
            return results
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def export(self, format='onnx', img_size=640):
        """Export the model to different formats."""
        if self.model is None:
            print("Model not initialized")
            return None
            
        try:
            path = self.model.export(format=format, imgsz=img_size)
            print(f"Model exported to {path}")
            return path
        except Exception as e:
            print(f"Error during model export: {e}")
            return None