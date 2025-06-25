#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convert_kitti_to_yolo(label_path, image_width, image_height, class_map):
    """
    Convert KITTI format label to YOLO format
    KITTI format: [class_name truncated occluded alpha x1 y1 x2 y2 h w l x y z rot_y score]
    YOLO format: [class_id center_x center_y width height] (normalized)
    """
    yolo_labels = []
    
    # Read KITTI label file
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            class_name = parts[0]
            
            if class_name in class_map:
                class_id = class_map[class_name]
                
                # KITTI bbox coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                # Convert to YOLO format (center_x, center_y, width, height) - normalized
                center_x = ((x1 + x2) / 2) / image_width
                center_y = ((y1 + y2) / 2) / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height
                
                yolo_labels.append([class_id, center_x, center_y, width, height])
    
    return yolo_labels

def main():
    # Directory paths
    data_dir = './data/kitti'
    images_dir = os.path.join(data_dir, 'training', 'image_2')
    labels_dir = os.path.join(data_dir, 'training', 'label_2')
    
    # Create output directory for converted labels
    yolo_labels_dir = os.path.join(data_dir, 'training', 'labels')
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    # Define class mapping
    class_map = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7
    }
    
    # Get all label files
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
    print(f"Found {len(label_files)} label files")
    
    # Sample a few files to display
    samples = random.sample(label_files, min(5, len(label_files)))
    
    for label_file in samples:
        base_name = os.path.basename(label_file)
        name_without_ext = os.path.splitext(base_name)[0]
        image_file = os.path.join(images_dir, name_without_ext + '.png')
        
        if not os.path.exists(image_file):
            print(f"Warning: Image not found for {label_file}")
            continue
        
        # Load image to get dimensions
        img = Image.open(image_file)
        img_width, img_height = img.size
        
        # Read example KITTI label
        print(f"\nExample KITTI label ({base_name}):")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):  # Show first 3 lines
                print(f"  Line {i+1}: {line.strip()}")
            if len(lines) > 3:
                print(f"  ... ({len(lines)-3} more lines)")
        
        # Convert to YOLO format
        yolo_labels = convert_kitti_to_yolo(label_file, img_width, img_height, class_map)
        
        # Show example of converted YOLO labels
        print("Converted YOLO format:")
        for i, label in enumerate(yolo_labels[:3]):  # Show first 3 labels
            print(f"  Label {i+1}: {label}")
        if len(yolo_labels) > 3:
            print(f"  ... ({len(yolo_labels)-3} more labels)")
        
        # Save converted labels
        yolo_label_path = os.path.join(yolo_labels_dir, name_without_ext + '.txt')
        with open(yolo_label_path, 'w') as f:
            for label in yolo_labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    # Check if we need to convert all labels
    print("\nDo you want to convert all KITTI labels to YOLO format? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nConverting all labels to YOLO format...")
        converted = 0
        
        for label_file in label_files:
            base_name = os.path.basename(label_file)
            name_without_ext = os.path.splitext(base_name)[0]
            image_file = os.path.join(images_dir, name_without_ext + '.png')
            
            if not os.path.exists(image_file):
                print(f"Warning: Image not found for {label_file}")
                continue
            
            # Load image to get dimensions
            img = Image.open(image_file)
            img_width, img_height = img.size
            
            # Convert to YOLO format
            yolo_labels = convert_kitti_to_yolo(label_file, img_width, img_height, class_map)
            
            # Save converted labels
            yolo_label_path = os.path.join(yolo_labels_dir, name_without_ext + '.txt')
            with open(yolo_label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(' '.join(map(str, label)) + '\n')
            
            converted += 1
            if converted % 100 == 0:
                print(f"  Converted {converted}/{len(label_files)} labels...")
        
        print(f"\nConversion complete! {converted} labels converted to YOLO format.")
        print(f"Converted labels saved to: {yolo_labels_dir}")
    
    return 0

if __name__ == "__main__":
    main()