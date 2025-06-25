"""
Utilities for working with the KITTI dataset.
Includes functions for parsing and converting KITTI format annotations.
"""
import os
import numpy as np
import cv2
from collections import defaultdict
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import KITTI_CLASSES

def parse_kitti_label(label_path):
    """
    Parse KITTI label file and extract object information.
    
    KITTI label format:
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
    
    Args:
        label_path: Path to KITTI label file
        
    Returns:
        List of objects with bounding box and class information
    """
    objects = []
    
    if not os.path.exists(label_path):
        return objects
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            
            # Skip empty lines
            if not parts:
                continue
                
            obj_class = parts[0]
            
            # Skip classes that are not in our class dictionary
            if obj_class not in KITTI_CLASSES or KITTI_CLASSES[obj_class] == -1:
                continue
            
            # Extract bounding box coordinates
            x1 = float(parts[4])
            y1 = float(parts[5])
            x2 = float(parts[6])
            y2 = float(parts[7])
            
            # Additional information
            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            
            # 3D data (if available)
            height = float(parts[8]) if len(parts) > 8 else 0
            width = float(parts[9]) if len(parts) > 9 else 0
            length = float(parts[10]) if len(parts) > 10 else 0
            x = float(parts[11]) if len(parts) > 11 else 0
            y = float(parts[12]) if len(parts) > 12 else 0
            z = float(parts[13]) if len(parts) > 13 else 0
            rotation_y = float(parts[14]) if len(parts) > 14 else 0
            
            # Create object dictionary
            obj = {
                'class': obj_class,
                'class_id': KITTI_CLASSES[obj_class],
                'bbox': [x1, y1, x2, y2],  # [left, top, right, bottom]
                'truncated': truncated,
                'occluded': occluded,
                'alpha': alpha,
                'dimensions': [height, width, length],
                'location': [x, y, z],
                'rotation_y': rotation_y
            }
            
            objects.append(obj)
    
    return objects

def convert_kitti_to_yolo(objects, img_width, img_height):
    """
    Convert KITTI format annotations to YOLO format.
    
    Args:
        objects: List of KITTI format objects
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of YOLO format annotations [class_id, x_center, y_center, width, height]
    """
    yolo_annotations = []
    
    for obj in objects:
        # Get class ID
        class_id = obj['class_id']
        
        # Get bounding box in KITTI format [x1, y1, x2, y2]
        x1, y1, x2, y2 = obj['bbox']
        
        # Convert to YOLO format [x_center, y_center, width, height] (normalized)
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Add to annotations
        yolo_annotations.append([class_id, x_center, y_center, width, height])
    
    return yolo_annotations

def convert_yolo_to_kitti(yolo_box, class_id, img_width, img_height):
    """
    Convert YOLO format annotations to KITTI format.
    
    Args:
        yolo_box: YOLO format bounding box [x_center, y_center, width, height] (normalized)
        class_id: Class ID
        img_width: Image width
        img_height: Image height
        
    Returns:
        KITTI format annotation [class_name, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, rotation_y]
    """
    # Get class name
    class_name = [k for k, v in KITTI_CLASSES.items() if v == class_id][0]
    
    # Extract normalized coordinates
    x_center, y_center, width, height = yolo_box
    
    # Convert to KITTI format [x1, y1, x2, y2] (absolute pixels)
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    
    # Default values for other fields
    truncated = 0
    occluded = 0
    alpha = -10
    h = 0
    w = 0
    l = 0
    x = 0
    y = 0
    z = 0
    rotation_y = 0
    
    # Create KITTI format annotation
    kitti_annotation = f"{class_name} {truncated} {occluded} {alpha} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {rotation_y:.2f}"
    
    return kitti_annotation

def create_kitti_label_file(objects, output_path):
    """
    Create a KITTI label file from objects.
    
    Args:
        objects: List of KITTI format objects
        output_path: Path to save the label file
        
    Returns:
        Path to the created label file
    """
    with open(output_path, 'w') as f:
        for obj in objects:
            class_name = obj['class']
            truncated = obj['truncated']
            occluded = obj['occluded']
            alpha = obj['alpha']
            x1, y1, x2, y2 = obj['bbox']
            h, w, l = obj['dimensions']
            x, y, z = obj['location']
            rotation_y = obj['rotation_y']
            
            line = f"{class_name} {truncated} {int(occluded)} {alpha} {x1} {y1} {x2} {y2} {h} {w} {l} {x} {y} {z} {rotation_y}\n"
            f.write(line)
    
    return output_path

def create_yolo_label_file(yolo_annotations, output_path):
    """
    Create a YOLO label file from annotations.
    
    Args:
        yolo_annotations: List of YOLO format annotations [class_id, x_center, y_center, width, height]
        output_path: Path to save the label file
        
    Returns:
        Path to the created label file
    """
    with open(output_path, 'w') as f:
        for ann in yolo_annotations:
            class_id, x_center, y_center, width, height = ann
            line = f"{int(class_id)} {x_center} {y_center} {width} {height}\n"
            f.write(line)
    
    return output_path

def convert_dataset_to_yolo(kitti_image_dir, kitti_label_dir, output_dir):
    """
    Convert an entire KITTI dataset to YOLO format.
    
    Args:
        kitti_image_dir: Directory containing KITTI images
        kitti_label_dir: Directory containing KITTI labels
        output_dir: Directory to save YOLO format files
        
    Returns:
        Dictionary with statistics about the conversion
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(kitti_image_dir) if f.endswith('.png')]
    
    # Initialize statistics
    stats = {
        'total_images': len(image_files),
        'converted_images': 0,
        'converted_objects': 0,
        'objects_per_class': defaultdict(int)
    }
    
    # Process each image
    for img_file in image_files:
        # Get corresponding label file
        label_file = img_file.replace('.png', '.txt')
        img_path = os.path.join(kitti_image_dir, img_file)
        label_path = os.path.join(kitti_label_dir, label_file)
        
        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            continue
        
        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Parse KITTI label
        objects = parse_kitti_label(label_path)
        
        # Skip if no valid objects
        if not objects:
            continue
        
        # Convert to YOLO format
        yolo_annotations = convert_kitti_to_yolo(objects, img_width, img_height)
        
        # Skip if no valid annotations
        if not yolo_annotations:
            continue
        
        # Save YOLO label file
        output_label_path = os.path.join(output_dir, 'labels', label_file)
        create_yolo_label_file(yolo_annotations, output_label_path)
        
        # Copy image (or create symbolic link to save space)
        output_img_path = os.path.join(output_dir, 'images', img_file)
        if not os.path.exists(output_img_path):
            os.symlink(os.path.abspath(img_path), output_img_path)
        
        # Update statistics
        stats['converted_images'] += 1
        stats['converted_objects'] += len(yolo_annotations)
        for obj in objects:
            stats['objects_per_class'][obj['class']] += 1
    
    return stats

def parse_kitti_calibration(calib_path):
    """
    Parse KITTI calibration file.
    
    Args:
        calib_path: Path to KITTI calibration file
        
    Returns:
        Dictionary of calibration matrices
    """
    calibs = {}
    
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(':')
            if len(parts) != 2:
                continue
                
            key = parts[0].strip()
            values = parts[1].strip().split()
            values = [float(v) for v in values]
            
            # Reshape to matrix format
            if key == 'P0' or key == 'P1' or key == 'P2' or key == 'P3':
                # Projection matrices P0, P1, P2, P3 (3x4)
                calibs[key] = np.array(values).reshape(3, 4)
            elif key == 'R0_rect':
                # Rectification matrix R0_rect (3x3)
                calibs[key] = np.array(values).reshape(3, 3)
            elif key == 'Tr_velo_to_cam' or key == 'Tr_imu_to_velo':
                # Transformation matrices (3x4)
                calibs[key] = np.array(values).reshape(3, 4)
    
    return calibs

def project_3d_to_2d(points_3d, calib_matrix):
    """
    Project 3D points to 2D image plane.
    
    Args:
        points_3d: 3D points as numpy array (Nx3)
        calib_matrix: Camera calibration matrix (3x4)
        
    Returns:
        2D points as numpy array (Nx2)
    """
    # Add homogeneous coordinate
    if points_3d.shape[1] == 3:
        points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    else:
        points_3d_hom = points_3d
    
    # Project to 2D
    points_2d_hom = np.dot(calib_matrix, points_3d_hom.T).T
    
    # Normalize by Z
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
    
    return points_2d

def get_class_statistics(label_dir):
    """
    Calculate statistics on object classes in a dataset.
    
    Args:
        label_dir: Directory containing KITTI label files
        
    Returns:
        Dictionary of class statistics
    """
    stats = {
        'total_objects': 0,
        'classes': defaultdict(int),
        'truncated': 0,
        'occluded': 0,
        'dimensions': {
            'width': {'min': float('inf'), 'max': 0, 'mean': 0, 'values': []},
            'height': {'min': float('inf'), 'max': 0, 'mean': 0, 'values': []}
        }
    }
    
    # Get list of label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    # Process each label file
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        
        # Parse KITTI label
        objects = parse_kitti_label(label_path)
        
        # Update statistics
        stats['total_objects'] += len(objects)
        
        for obj in objects:
            class_name = obj['class']
            stats['classes'][class_name] += 1
            
            if obj['truncated'] > 0:
                stats['truncated'] += 1
                
            if obj['occluded'] > 0:
                stats['occluded'] += 1
                
            # Calculate box dimensions
            x1, y1, x2, y2 = obj['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Update width statistics
            stats['dimensions']['width']['min'] = min(stats['dimensions']['width']['min'], width)
            stats['dimensions']['width']['max'] = max(stats['dimensions']['width']['max'], width)
            stats['dimensions']['width']['values'].append(width)
            
            # Update height statistics
            stats['dimensions']['height']['min'] = min(stats['dimensions']['height']['min'], height)
            stats['dimensions']['height']['max'] = max(stats['dimensions']['height']['max'], height)
            stats['dimensions']['height']['values'].append(height)
    
    # Calculate means
    if stats['dimensions']['width']['values']:
        stats['dimensions']['width']['mean'] = np.mean(stats['dimensions']['width']['values'])
        stats['dimensions']['width']['median'] = np.median(stats['dimensions']['width']['values'])
        stats['dimensions']['width']['std'] = np.std(stats['dimensions']['width']['values'])
        # Remove raw values to save memory
        del stats['dimensions']['width']['values']
        
    if stats['dimensions']['height']['values']:
        stats['dimensions']['height']['mean'] = np.mean(stats['dimensions']['height']['values'])
        stats['dimensions']['height']['median'] = np.median(stats['dimensions']['height']['values'])
        stats['dimensions']['height']['std'] = np.std(stats['dimensions']['height']['values'])
        # Remove raw values to save memory
        del stats['dimensions']['height']['values']
    
    return stats