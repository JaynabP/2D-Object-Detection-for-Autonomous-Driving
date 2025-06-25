"""
Data preprocessing utilities for the KITTI object detection dataset.
Handles data splitting, normalization, and format conversion.
"""
import os
import sys
import numpy as np
import random
import shutil
import cv2
from tqdm import tqdm
from pathlib import Path

# Add this code to preprocessing.py
KITTI_CLASSES = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1  # Usually ignored in training
}

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import KITTI_TRAIN_IMAGES, KITTI_TRAIN_LABELS, KITTI_TRAIN_CALIB
from src.utils.kitti_utils import parse_kitti_label, convert_kitti_to_yolo

def split_train_val(image_dir, label_dir, val_ratio=0.2, output_dir=None, seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing label files
        val_ratio: Ratio of validation set size
        output_dir: Directory to save split lists
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, val_files)
    """
    # Set random seed
    random.seed(seed)
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split point
    val_size = int(len(image_files) * val_ratio)
    
    # Split dataset
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    # Save split lists if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Write train files
        with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
            for file in train_files:
                f.write(f"{file}\n")
        
        # Write val files
        with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
            for file in val_files:
                f.write(f"{file}\n")
        
        print(f"Train list ({len(train_files)} files) saved to {os.path.join(output_dir, 'train.txt')}")
        print(f"Validation list ({len(val_files)} files) saved to {os.path.join(output_dir, 'val.txt')}")
    
    return train_files, val_files

def create_yolo_dataset(image_dir, label_dir, output_dir, train_files=None, val_files=None, copy_images=False):
    """
    Create YOLO format dataset from KITTI format.
    
    Args:
        image_dir: Directory containing KITTI images
        label_dir: Directory containing KITTI labels
        output_dir: Directory to save YOLO format files
        train_files: List of training files (optional)
        val_files: List of validation files (optional)
        copy_images: Whether to copy images or create symbolic links
        
    Returns:
        Dictionary with paths of created directories
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train, val directories if train/val files are provided
    if train_files and val_files:
        train_img_dir = os.path.join(output_dir, 'images', 'train')
        train_label_dir = os.path.join(output_dir, 'labels', 'train')
        val_img_dir = os.path.join(output_dir, 'images', 'val')
        val_label_dir = os.path.join(output_dir, 'labels', 'val')
        
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)
        
        # Process training files
        print(f"Processing {len(train_files)} training files...")
        for img_file in tqdm(train_files):
            # Get label file name
            label_file = img_file.replace('.png', '.txt')
            
            # Get file paths
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            
            # Skip if image file doesn't exist
            if not os.path.exists(img_path):
                continue
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Parse KITTI label if it exists
            if os.path.exists(label_path):
                objects = parse_kitti_label(label_path)
                yolo_annotations = convert_kitti_to_yolo(objects, img_width, img_height)
                
                # Write YOLO format label file
                yolo_label_path = os.path.join(train_label_dir, label_file)
                with open(yolo_label_path, 'w') as f:
                    for ann in yolo_annotations:
                        class_id, x_center, y_center, width, height = ann
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            
            # Copy or symlink image
            if copy_images:
                shutil.copy(img_path, os.path.join(train_img_dir, img_file))
            else:
                # Use absolute path for symlink source
                src_path = os.path.abspath(img_path)
                dst_path = os.path.join(train_img_dir, img_file)
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
        
        # Process validation files
        print(f"Processing {len(val_files)} validation files...")
        for img_file in tqdm(val_files):
            # Get label file name
            label_file = img_file.replace('.png', '.txt')
            
            # Get file paths
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            
            # Skip if image file doesn't exist
            if not os.path.exists(img_path):
                continue
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Parse KITTI label if it exists
            if os.path.exists(label_path):
                objects = parse_kitti_label(label_path)
                yolo_annotations = convert_kitti_to_yolo(objects, img_width, img_height)
                
                # Write YOLO format label file
                yolo_label_path = os.path.join(val_label_dir, label_file)
                with open(yolo_label_path, 'w') as f:
                    for ann in yolo_annotations:
                        class_id, x_center, y_center, width, height = ann
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            
            # Copy or symlink image
            if copy_images:
                shutil.copy(img_path, os.path.join(val_img_dir, img_file))
            else:
                # Use absolute path for symlink source
                src_path = os.path.abspath(img_path)
                dst_path = os.path.join(val_img_dir, img_file)
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
        
        return {
            'train_img_dir': train_img_dir,
            'train_label_dir': train_label_dir,
            'val_img_dir': val_img_dir,
            'val_label_dir': val_label_dir
        }
    
    else:
        # If no train/val split is provided, process all files into a single directory
        all_img_dir = os.path.join(output_dir, 'images', 'all')
        all_label_dir = os.path.join(output_dir, 'labels', 'all')
        
        os.makedirs(all_img_dir, exist_ok=True)
        os.makedirs(all_label_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        
        # Process all files
        print(f"Processing {len(image_files)} files...")
        for img_file in tqdm(image_files):
            # Get label file name
            label_file = img_file.replace('.png', '.txt')
            
            # Get file paths
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            
            # Skip if image file doesn't exist
            if not os.path.exists(img_path):
                continue
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Parse KITTI label if it exists
            if os.path.exists(label_path):
                objects = parse_kitti_label(label_path)
                yolo_annotations = convert_kitti_to_yolo(objects, img_width, img_height)
                
                # Write YOLO format label file
                yolo_label_path = os.path.join(all_label_dir, label_file)
                with open(yolo_label_path, 'w') as f:
                    for ann in yolo_annotations:
                        class_id, x_center, y_center, width, height = ann
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            
            # Copy or symlink image
            if copy_images:
                shutil.copy(img_path, os.path.join(all_img_dir, img_file))
            else:
                # Use absolute path for symlink source
                src_path = os.path.abspath(img_path)
                dst_path = os.path.join(all_img_dir, img_file)
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
        
        return {
            'all_img_dir': all_img_dir,
            'all_label_dir': all_label_dir
        }

def create_yolo_yaml(img_dirs, output_dir, class_names):
    """
    Create YAML configuration file for YOLOv5 training.
    
    Args:
        img_dirs: Dictionary with image directory paths
        output_dir: Directory to save YAML file
        class_names: List of class names
        
    Returns:
        Path to YAML file
    """
    yaml_data = {
        'path': os.path.abspath(os.path.dirname(output_dir)),
        'train': img_dirs.get('train_img_dir', ''),
        'val': img_dirs.get('val_img_dir', ''),
        'test': '',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Write YAML file
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    
    with open(yaml_path, 'w') as f:
        f.write("# KITTI Dataset for YOLOv5\n\n")
        
        # Write path
        f.write(f"path: {yaml_data['path']}\n\n")
        
        # Write train/val/test paths
        f.write(f"train: {yaml_data['train']}\n")
        f.write(f"val: {yaml_data['val']}\n")
        f.write(f"test: {yaml_data['test']}\n\n")
        
        # Write number of classes
        f.write(f"nc: {yaml_data['nc']}\n\n")
        
        # Write class names
        f.write("names:\n")
        for name in yaml_data['names']:
            f.write(f"  - '{name}'\n")
    
    print(f"YAML configuration saved to {yaml_path}")
    return yaml_path

def normalize_image(image, mean, std):
    """
    Normalize image by subtracting mean and dividing by standard deviation.
    
    Args:
        image: Input image
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized image
    """
    # Convert image to float
    image = image.astype(np.float32) / 255.0
    
    # Normalize each channel
    for i in range(3):  # Assuming 3 color channels
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    return image

def preprocess_dataset(image_dir, label_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Preprocess KITTI dataset: split into train/val, convert to YOLO format.
    
    Args:
        image_dir: Directory containing KITTI images
        label_dir: Directory containing KITTI labels
        output_dir: Directory to save preprocessed dataset
        val_ratio: Ratio of validation set size
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with dataset paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Splitting dataset into train and validation sets...")
    train_files, val_files = split_train_val(
        image_dir=image_dir,
        label_dir=label_dir,
        val_ratio=val_ratio,
        output_dir=output_dir,
        seed=seed
    )
    
    print("\nStep 2: Creating YOLO format dataset...")
    img_dirs = create_yolo_dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        train_files=train_files,
        val_files=val_files,
        copy_images=False  # Use symlinks to save disk space
    )
    
    print("\nStep 3: Creating YAML configuration...")
    # Get class names (excluding DontCare)
    class_names = [k for k, v in KITTI_CLASSES.items() if v != -1]
    yaml_path = create_yolo_yaml(img_dirs, output_dir, class_names)
    
    # Return paths
    return {
        'train_files': train_files,
        'val_files': val_files,
        'img_dirs': img_dirs,
        'yaml_path': yaml_path
    }

# Main function to demonstrate usage
def main():
    """Main function to demonstrate preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess KITTI dataset for YOLOv5')
    parser.add_argument('--image-dir', type=str, default=KITTI_TRAIN_IMAGES, help='Directory containing images')
    parser.add_argument('--label-dir', type=str, default=KITTI_TRAIN_LABELS, help='Directory containing labels')
    parser.add_argument('--output-dir', type=str, default='../data/processed', help='Output directory')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main()