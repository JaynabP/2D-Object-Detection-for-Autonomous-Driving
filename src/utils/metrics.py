"""
Evaluation metrics for object detection tasks.
Implements IoU, precision/recall, mAP, and other metrics.
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional

def compute_iou(boxes1, boxes2):
    """
    Compute IoU (Intersection over Union) between two sets of bounding boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in format [x_center, y_center, width, height]
        boxes2: Array of shape (M, 4) in format [x_center, y_center, width, height]
        
    Returns:
        IoU matrix of shape (N, M)
    """
    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    def xywh_to_xyxy(boxes):
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return np.stack([x1, y1, x2, y2], axis=1)
    
    boxes1 = xywh_to_xyxy(boxes1)
    boxes2 = xywh_to_xyxy(boxes2)
    
    # Get number of boxes
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Initialize IoU matrix
    iou_matrix = np.zeros((N, M))
    
    # Compute IoU for each pair of boxes
    for i in range(N):
        box1 = boxes1[i]
        
        # Calculate areas of boxes1
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        for j in range(M):
            box2 = boxes2[j]
            
            # Calculate areas of boxes2
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # Calculate intersection coordinates
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            # Calculate intersection area
            if x_right < x_left or y_bottom < y_top:
                intersection = 0
            else:
                intersection = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union area
            union = area1 + area2 - intersection
            
            # Calculate IoU
            iou = intersection / union if union > 0 else 0
            iou_matrix[i, j] = iou
    
    return iou_matrix

def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision and recall at different confidence thresholds.
    
    Args:
        predictions: List of dictionaries with keys 'boxes', 'labels', 'scores'
        ground_truths: List of dictionaries with keys 'boxes', 'labels'
        iou_threshold: IoU threshold for considering a detection correct
        
    Returns:
        precision: List of precision values
        recall: List of recall values
        average_precision: Average precision
    """
    # Sort predictions by confidence score
    sorted_indices = np.argsort([-p['score'] for p in predictions])
    sorted_predictions = [predictions[i] for i in sorted_indices]
    
    # Count total number of ground truths
    total_gt = len(ground_truths)
    
    # Initialize arrays to keep track of true positives and false positives
    tp = np.zeros(len(sorted_predictions))
    fp = np.zeros(len(sorted_predictions))
    
    # Keep track of matched ground truths
    matched_gt = set()
    
    # Process each prediction
    for i, pred in enumerate(sorted_predictions):
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            # Skip already matched ground truths
            if j in matched_gt:
                continue
                
            # Skip if labels don't match
            if pred['label'] != gt['label']:
                continue
                
            # Calculate IoU
            iou = compute_iou(
                np.array([pred['box']]), 
                np.array([gt['box']])
            )[0, 0]
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # If IoU is above threshold, it's a true positive
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1
    
    # Calculate cumulative true positives and false positives
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    
    # Calculate precision and recall
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    recall = cumulative_tp / total_gt if total_gt > 0 else np.zeros_like(cumulative_tp)
    
    # Append sentinel values to ensure curves start at 0 recall and end at 0 precision
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    
    # Ensure precision is monotonically decreasing (for PR curve)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate average precision (area under PR curve)
    # Find indices where recall changes
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return precision, recall, average_precision

def calculate_map(predictions, ground_truths, iou_thresholds=None, class_names=None):
    """
    Calculate mean Average Precision (mAP) at different IoU thresholds.
    
    Args:
        predictions: List of dictionaries with keys 'boxes', 'labels', 'scores' for each image
        ground_truths: List of dictionaries with keys 'boxes', 'labels' for each image
        iou_thresholds: List of IoU thresholds for mAP calculation
        class_names: Dictionary mapping class indices to class names
        
    Returns:
        mAP: Mean average precision across all classes
        ap_per_class: Dictionary mapping class indices to AP values
    """
    if iou_thresholds is None:
        # Default: mAP@0.5
        iou_thresholds = [0.5]
        
    # Group predictions and ground truths by class
    class_predictions = {}
    class_ground_truths = {}
    
    # Process all predictions
    for img_idx, preds in enumerate(predictions):
        for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
            if label not in class_predictions:
                class_predictions[label] = []
            
            class_predictions[label].append({
                'img_idx': img_idx,
                'box': box,
                'score': score,
                'label': label
            })
    
    # Process all ground truths
    for img_idx, gts in enumerate(ground_truths):
        for box, label in zip(gts['boxes'], gts['labels']):
            if label not in class_ground_truths:
                class_ground_truths[label] = []
            
            class_ground_truths[label].append({
                'img_idx': img_idx,
                'box': box,
                'label': label
            })
    
    # Calculate AP for each class
    ap_per_class = {}
    for label in class_ground_truths:
        if label not in class_predictions:
            # No predictions for this class
            ap_per_class[label] = 0
            continue
        
        # Calculate AP for each IoU threshold
        aps = []
        for iou_threshold in iou_thresholds:
            precision, recall, ap = calculate_precision_recall(
                class_predictions[label],
                class_ground_truths[label],
                iou_threshold
            )
            aps.append(ap)
        
        # Average AP across IoU thresholds
        ap_per_class[label] = np.mean(aps)
    
    # Calculate mAP across all classes
    mAP = np.mean([ap for ap in ap_per_class.values()])
    
    # Convert class indices to class names if provided
    if class_names is not None:
        ap_per_class_named = {class_names[k]: v for k, v in ap_per_class.items() if k in class_names}
        return mAP, ap_per_class_named
    else:
        return mAP, ap_per_class

def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0

def calculate_confusion_matrix(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """
    Calculate confusion matrix for object detection.
    
    Args:
        predictions: List of dictionaries with keys 'boxes', 'labels', 'scores' for each image
        ground_truths: List of dictionaries with keys 'boxes', 'labels' for each image
        num_classes: Number of classes
        iou_threshold: IoU threshold for considering a detection correct
        
    Returns:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Process each image
    for img_idx in range(len(predictions)):
        # Get predictions and ground truths for this image
        preds = predictions[img_idx]
        gts = ground_truths[img_idx]
        
        # Skip if no predictions or ground truths
        if len(preds['boxes']) == 0 or len(gts['boxes']) == 0:
            continue
        
        # Calculate IoU between predictions and ground truths
        ious = compute_iou(
            np.array(preds['boxes']),
            np.array(gts['boxes'])
        )
        
        # Process each prediction
        for pred_idx, pred_label in enumerate(preds['labels']):
            # Find best matching ground truth
            best_iou = np.max(ious[pred_idx])
            best_gt_idx = np.argmax(ious[pred_idx])
            
            # If IoU is above threshold, update confusion matrix
            if best_iou >= iou_threshold:
                gt_label = gts['labels'][best_gt_idx]
                confusion_matrix[gt_label, pred_label] += 1
            else:
                # False positive (no matching ground truth)
                # Some implementations add this to a background class
                pass
    
    # Normalize confusion matrix (by row)
    for i in range(confusion_matrix.shape[0]):
        row_sum = np.sum(confusion_matrix[i])
        if row_sum > 0:
            confusion_matrix[i] /= row_sum
    
    return confusion_matrix