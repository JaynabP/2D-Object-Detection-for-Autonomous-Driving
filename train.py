"""
Main training script for KITTI 2D object detection using YOLO.
Handles the complete training pipeline including data loading,
model configuration, training, and evaluation.
"""
import os
import sys
import argparse
import yaml
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import project modules
from src.data.dataset import KITTIDataset, create_data_splits
from src.data.augmentation import get_training_augmentations, get_validation_augmentations
from src.models.yolo import KITTIYOLOModel
from src.utils.visualization import visualize_dataset_samples, visualize_predictions
from src.config import (
    KITTI_TRAIN_IMAGES, KITTI_TRAIN_LABELS, KITTI_CLASSES,
    VAL_RATIO, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS, 
    NUM_EPOCHS, LEARNING_RATE, YOLO_CONFIG, MODEL_SAVE_DIR,
    IMG_SIZE
)

def prepare_data_files(train_images, val_images, label_dir, output_dir):
    """
    Prepare YOLO format data files listing the training and validation images.
    
    Args:
        train_images: List of training image filenames
        val_images: List of validation image filenames
        label_dir: Directory containing label files
        output_dir: Directory to save the data files
        
    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train file path
    train_file = os.path.join(output_dir, 'train.txt')
    
    # Write train image paths to file
    with open(train_file, 'w') as f:
        for img in train_images:
            img_path = os.path.join(KITTI_TRAIN_IMAGES, img)
            f.write(f"{img_path}\n")
    
    # Create val file path
    val_file = os.path.join(output_dir, 'val.txt')
    
    # Write val image paths to file
    with open(val_file, 'w') as f:
        for img in val_images:
            img_path = os.path.join(KITTI_TRAIN_IMAGES, img)
            f.write(f"{img_path}\n")
    
    print(f"Created train file with {len(train_images)} images at {train_file}")
    print(f"Created val file with {len(val_images)} images at {val_file}")
    
    return train_file, val_file

def prepare_yaml(train_file, val_file, output_dir):
    """
    Prepare YAML configuration for YOLOv5.
    
    Args:
        train_file: Path to train.txt file
        val_file: Path to val.txt file
        output_dir: Directory to save the YAML file
        
    Returns:
        Path to YAML file
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names (excluding DontCare)
    names = [k for k, v in KITTI_CLASSES.items() if v != -1]
    
    # Create YAML data
    data = {
        'train': train_file,
        'val': val_file,
        'nc': len(names),
        'names': names
    }
    
    # Write YAML file
    yaml_path = os.path.join(output_dir, 'kitti.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created YAML config at {yaml_path}")
    return yaml_path

def visualize_samples(dataset, output_dir, num_samples=3):
    """
    Visualize random samples from the dataset and save the visualizations.
    
    Args:
        dataset: Dataset to visualize
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create output directory if needed
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize dataset samples
    fig = visualize_dataset_samples(dataset, num_samples=num_samples)
    
    # Save visualization
    fig_path = os.path.join(vis_dir, 'dataset_samples.png')
    fig.savefig(fig_path)
    plt.close(fig)
    
    print(f"Saved dataset visualizations to {fig_path}")

def main(args):
    """Main training function"""
    print("=" * 80)
    print(f"KITTI 2D Object Detection Training - {datetime.datetime.now()}")
    print("=" * 80)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"yolo_kitti_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create data splits
    print("\n📊 Creating data splits...")
    train_files, val_files = create_data_splits(
        img_dir=KITTI_TRAIN_IMAGES,
        label_dir=KITTI_TRAIN_LABELS,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Prepare YOLOv5 format files
    print("\n📁 Preparing YOLOv5 format files...")
    train_file, val_file = prepare_data_files(
        train_files, 
        val_files, 
        KITTI_TRAIN_LABELS, 
        output_dir
    )
    
    # Create YAML config
    yaml_path = prepare_yaml(train_file, val_file, output_dir)
    
    # Create datasets for visualization
    print("\n👁️ Creating dataset for visualization...")
    vis_dataset = KITTIDataset(
        img_dir=KITTI_TRAIN_IMAGES,
        label_dir=KITTI_TRAIN_LABELS,
        transform=get_validation_augmentations()
    )
    
    # Visualize samples
    print("Visualizing dataset samples...")
    visualize_samples(vis_dataset, output_dir)
    
    # Initialize YOLO model
    print("\n🚀 Initializing YOLO model...")
    # Count number of classes (excluding DontCare)
    num_classes = sum(1 for v in KITTI_CLASSES.values() if v != -1)
    model = KITTIYOLOModel(num_classes=num_classes)
    model.build_model()
    
    # Configure model for training
    print("Configuring hyperparameters...")
    hyp = model.configure_training()
    
    # Create model save directory
    model_save_dir = os.path.join(output_dir, 'weights')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Start training
    print("\n🏋️ Starting training...")
    start_time = time.time()
    
    best_weights_path = model.train(
        data_yaml=yaml_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=IMG_SIZE,
        save_dir=output_dir
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    print(f"Best weights saved to: {best_weights_path}")
    
    print("\n📋 Training Summary:")
    print(f"- Model: {YOLO_CONFIG['model_type']}")
    print(f"- Classes: {num_classes}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Image size: {IMG_SIZE}")
    print(f"- Training samples: {len(train_files)}")
    print(f"- Validation samples: {len(val_files)}")
    
    print("\nTo evaluate the model on test data, run:")
    print(f"python evaluate.py --weights {best_weights_path}")
    
    print("\nTo run inference on new images, run:")
    print(f"python inference.py --weights {best_weights_path} --img /path/to/image.png")
    
    return best_weights_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model on KITTI dataset")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, default=MODEL_SAVE_DIR, help="Output directory")
    
    args = parser.parse_args()
    main(args)