#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on KITTI dataset")
    parser.add_argument('--model', type=str, default='yolov5su.pt', help='Initial model weights')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=608, help='Image size')
    parser.add_argument('--data_dir', type=str, default='./data/kitti', help='Path to KITTI dataset directory')
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("KITTI Object Detection Training")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Dataset path: {args.data_dir}")
    print("=" * 80)
    
    # Check if dataset directories exist
    train_dir = os.path.join(args.data_dir, 'training/image_2')
    label_dir = os.path.join(args.data_dir, 'training/label_2')
    test_dir = os.path.join(args.data_dir, 'testing/image_2')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training images directory not found at {train_dir}")
        return 1
    
    if not os.path.exists(label_dir):
        print(f"Error: Training labels directory not found at {label_dir}")
        return 1
    
    print(f"Found training images: {len(os.listdir(train_dir))}")
    print(f"Found training labels: {len(os.listdir(label_dir))}")
    
    # Load model
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Define class names for KITTI
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    print(f"Classes: {class_names}")
    
    # Create temporary YAML file
    temp_yaml_path = 'temp_kitti.yaml'
    data_dict = {
        'path': args.data_dir,
        'train': 'training/image_2',
        'val': 'training/image_2',
        'nc': 8,
        'names': class_names
    }
    
    with open(temp_yaml_path, 'w') as yaml_file:
        yaml.dump(data_dict, yaml_file, default_flow_style=False)
    
    print(f"Created temporary YAML config at: {temp_yaml_path}")
    
    # Configure training parameters
    try:
        results = model.train(
            data=temp_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project='runs/train',
            name='kitti_detection',
            split=0.9
        )
        
        print("\nTraining complete!")
        print(f"Best model saved at: runs/train/kitti_detection/weights/best.pt")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up temporary file
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())