"""
KITTI Dataset class for loading and preprocessing images and annotations.
Converts KITTI format to YOLO format for training YOLOv5.
"""
import os
import numpy as np
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import sys
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import KITTI_CLASSES

class KITTIDataset(Dataset):
    """
    KITTI dataset class for object detection.
    Converts KITTI format annotations to YOLO format.
    """
    def __init__(self, img_dir, label_dir=None, transform=None, is_test=False):
        """
        Initialize KITTI dataset.
        
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing label files (can be None for test set)
            transform: Optional transforms to apply to images
            is_test: Whether this is a test dataset (no labels)
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get list of image files
        self.img_files = sorted([os.path.join(img_dir, f) 
                             for f in os.listdir(img_dir) 
                             if f.endswith('.png')])
        
        # Create corresponding label file paths
        if not is_test and label_dir is not None:
            self.label_files = [os.path.join(label_dir, 
                                os.path.basename(f).replace('.png', '.txt')) 
                                for f in self.img_files]
        else:
            self.label_files = [None] * len(self.img_files)
            
        # Class mapping
        self.class_dict = KITTI_CLASSES
        
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """Get image and labels by index"""
        # Load image
        img_path = self.img_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Get image file name for reference
        img_name = os.path.basename(img_path)
        
        # Initialize targets
        boxes = []
        labels = []
        
        # Parse label file if not test set
        if not self.is_test and self.label_files[idx] is not None:
            label_path = self.label_files[idx]
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        obj_class = parts[0]
                        
                        # Skip classes that are not in our class dictionary
                        if obj_class not in self.class_dict or self.class_dict[obj_class] == -1:
                            continue
                            
                        # KITTI format: [left, top, right, bottom] in absolute coordinates
                        bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                        
                        # Convert to YOLO format: [x_center, y_center, width, height] normalized
                        x_center = (bbox[0] + bbox[2]) / 2.0 / orig_w
                        y_center = (bbox[1] + bbox[3]) / 2.0 / orig_h
                        width = (bbox[2] - bbox[0]) / orig_w
                        height = (bbox[3] - bbox[1]) / orig_h
                        
                        # Skip invalid boxes
                        if width <= 0 or height <= 0:
                            continue
                            
                        boxes.append([x_center, y_center, width, height])
                        labels.append(self.class_dict[obj_class])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'img_name': img_name,
            'orig_size': (orig_h, orig_w)
        }
        
        # Apply transformations if specified
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            
            if not self.is_test:
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
                target['boxes'] = boxes
                target['labels'] = labels
        else:
            # Normalize image manually if no transform
            image = image / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, target

def create_data_splits(img_dir, label_dir, val_ratio=0.2, seed=42):
    """
    Create training and validation splits.
    
    Args:
        img_dir: Directory containing images
        label_dir: Directory containing labels
        val_ratio: Ratio of validation set size to total dataset size
        seed: Random seed for reproducibility
        
    Returns:
        train_files: List of training image files
        val_files: List of validation image files
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image files
    all_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    # Shuffle files
    random.shuffle(all_files)
    
    # Calculate split point
    split_idx = int(len(all_files) * (1 - val_ratio))
    
    # Split files
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    return train_files, val_files