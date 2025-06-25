"""
Visualization utilities for KITTI object detection.
Includes functions for drawing bounding boxes, plotting predictions,
and creating visual reports on model performance.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
from PIL import Image
import sys
from typing import List, Dict, Tuple, Union

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import KITTI_CLASSES, IMG_SIZE, MEAN, STD

# Create inverse class mapping for label to name conversion
INV_CLASSES = {v: k for k, v in KITTI_CLASSES.items() if v != -1}

# Define colors for different classes
COLORS = {
    'Car': (0, 0, 255),         # Red
    'Van': (0, 165, 255),       # Orange
    'Truck': (0, 255, 255),     # Yellow
    'Pedestrian': (0, 255, 0),  # Green
    'Person_sitting': (255, 0, 0),  # Blue
    'Cyclist': (255, 0, 255),   # Purple
    'Tram': (255, 255, 0),      # Cyan
    'Misc': (128, 128, 128)     # Gray
}

def denormalize_image(image):
    """
    Denormalize an image from normalized tensor to numpy uint8 format.
    
    Args:
        image: Normalized image tensor [C, H, W]
        
    Returns:
        Denormalized image as numpy array [H, W, C] with uint8 values
    """
    # Check if image is a tensor
    if torch.is_tensor(image):
        # Move to CPU if on GPU
        image = image.cpu().clone()
        
        # If image is [C, H, W], convert to [H, W, C]
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
            
        # Convert to numpy
        image = image.numpy()
    
    # Denormalize
    image = image * np.array(STD) + np.array(MEAN)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image

def draw_boxes(image, boxes, labels, scores=None, thickness=2):
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Image as numpy array [H, W, C]
        boxes: Bounding boxes in [x_center, y_center, width, height] format (normalized)
        labels: Class labels for each box
        scores: Optional confidence scores for each box
        thickness: Line thickness for bounding boxes
        
    Returns:
        Image with drawn bounding boxes
    """
    # Make a copy of the image to avoid modifying the original
    img_with_boxes = image.copy()
    h, w = image.shape[:2]
    
    # Draw each box
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Convert from YOLO format [x_center, y_center, width, height] to corner format
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Get class name and color
        class_name = INV_CLASSES.get(int(label), "Unknown")
        color = COLORS.get(class_name, (255, 255, 255))  # White for unknown classes
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Add label with score if provided
        label_text = class_name
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
            
        # Define text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        
        # Draw text background
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_with_boxes,
            label_text,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness
        )
        
    return img_with_boxes

def visualize_dataset_samples(dataset, num_samples=5, figsize=(15, 10)):
    """
    Visualize random samples from a dataset with their annotations.
    
    Args:
        dataset: KITTIDataset instance
        num_samples: Number of samples to visualize
        figsize: Figure size
        
    Returns:
        Matplotlib figure with visualized samples
    """
    import random
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Plot each sample
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        
        # Denormalize image if needed
        if torch.is_tensor(image):
            image = denormalize_image(image)
        
        # Get boxes and labels
        boxes = target["boxes"]
        labels = target["labels"]
        
        # Draw boxes on image
        img_with_boxes = draw_boxes(image, boxes, labels)
        
        # Display image
        axes[i].imshow(img_with_boxes)
        axes[i].set_title(f"Sample {idx}: {target['img_name']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_predictions(image_path, results, confidence_threshold=0.25, figsize=(12, 8)):
    """
    Visualize YOLOv5 model predictions on an image.
    
    Args:
        image_path: Path to image file
        results: YOLOv5 results object
        confidence_threshold: Minimum confidence score to display
        figsize: Figure size
        
    Returns:
        Matplotlib figure with visualized predictions
    """
    # Load original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    pred = results.xyxyn[0].cpu().numpy()  # First image, normalized xywh
    
    # Filter by confidence
    confident_detections = pred[pred[:, 4] >= confidence_threshold]
    
    # Convert to [x_center, y_center, width, height] format
    boxes = []
    for x1, y1, x2, y2, _, _ in confident_detections:
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        boxes.append([x_center, y_center, width, height])
    
    # Extract labels and scores
    labels = confident_detections[:, 5].astype(int)
    scores = confident_detections[:, 4]
    
    # Draw boxes
    img_with_boxes = draw_boxes(original_image, boxes, labels, scores)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img_with_boxes)
    ax.set_title(f"Detections: {len(confident_detections)}")
    ax.axis('off')
    
    return fig

def plot_precision_recall_curve(precision, recall, average_precision, class_name='', figsize=(8, 6)):
    """
    Plot precision-recall curve for a given class.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
        average_precision: Average precision value
        class_name: Name of the class
        figsize: Figure size
        
    Returns:
        Matplotlib figure with precision-recall curve
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.step(recall, precision, where='post', color='b', alpha=0.2)
    ax.fill_between(recall, precision, alpha=0.2, color='b')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f'Precision-Recall Curve for {class_name}: AP={average_precision:.4f}')
    
    return fig

def create_evaluation_report(metrics, save_path=None):
    """
    Create a visual report of model evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report (optional)
        
    Returns:
        Matplotlib figure with evaluation metrics
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot mAP by class
    class_aps = metrics.get('class_ap', {})
    if class_aps:
        classes = list(class_aps.keys())
        values = list(class_aps.values())
        
        axs[0, 0].bar(classes, values, color='skyblue')
        axs[0, 0].set_title('AP by Class')
        axs[0, 0].set_xlabel('Class')
        axs[0, 0].set_ylabel('Average Precision (AP)')
        axs[0, 0].set_ylim([0, 1.05])
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # Annotate values
        for i, v in enumerate(values):
            axs[0, 0].text(i, v + 0.02, f"{v:.3f}", ha='center')
            
    # Plot precision-recall curve for all classes
    if 'all_precision' in metrics and 'all_recall' in metrics:
        axs[0, 1].step(
            metrics['all_recall'], 
            metrics['all_precision'], 
            where='post', 
            color='blue', 
            alpha=0.2
        )
        axs[0, 1].fill_between(
            metrics['all_recall'], 
            metrics['all_precision'], 
            alpha=0.2, 
            color='blue'
        )
        axs[0, 1].set_title(f"Precision-Recall Curve: mAP={metrics.get('mAP', 0):.4f}")
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Precision')
        axs[0, 1].set_ylim([0.0, 1.05])
        axs[0, 1].set_xlim([0.0, 1.0])
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        axs[1, 0].matshow(cm, cmap='Blues')
        axs[1, 0].set_title('Confusion Matrix')
        axs[1, 0].set_xlabel('Predicted')
        axs[1, 0].set_ylabel('True')
        
        # Add class names
        classes = list(INV_CLASSES.values())
        axs[1, 0].set_xticks(range(len(classes)))
        axs[1, 0].set_yticks(range(len(classes)))
        axs[1, 0].set_xticklabels(classes, rotation=45, ha='right')
        axs[1, 0].set_yticklabels(classes)
        
        # Add values to cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[1, 0].text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', 
                             color='black' if cm[i, j] < 0.5 else 'white')
    
    # Plot additional metrics
    additional_metrics = {}
    for k, v in metrics.items():
        if k not in ['class_ap', 'all_precision', 'all_recall', 'confusion_matrix', 'mAP'] and not isinstance(v, dict):
            additional_metrics[k] = v
    
    if additional_metrics:
        # Convert to lists
        metric_names = list(additional_metrics.keys())
        metric_values = list(additional_metrics.values())
        
        axs[1, 1].bar(metric_names, metric_values, color='lightgreen')
        axs[1, 1].set_title('Additional Metrics')
        axs[1, 1].set_xlabel('Metric')
        axs[1, 1].set_ylabel('Value')
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        # Annotate values
        for i, v in enumerate(metric_values):
            axs[1, 1].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation report saved to {save_path}")
    
    return fig