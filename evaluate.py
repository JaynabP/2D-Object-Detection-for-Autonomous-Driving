"""
Evaluation script for KITTI 2D object detection using YOLO.
Computes metrics including mAP, precision, recall, and confusion matrix.
"""
import os
import sys
import argparse
import yaml
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import project modules
from src.data.dataset import KITTIDataset
from src.data.augmentation import get_validation_augmentations
from src.models.yolo import KITTIYOLOModel
from src.utils.metrics import compute_iou, calculate_precision_recall, calculate_map
from src.utils.visualization import create_evaluation_report, visualize_predictions
from src.config import (
    KITTI_TRAIN_IMAGES, KITTI_TRAIN_LABELS, KITTI_TEST_IMAGES,
    KITTI_CLASSES, IMG_SIZE, YOLO_CONFIG
)

def evaluate_model(model, dataset, iou_threshold=0.5, conf_threshold=0.25, max_dets=100):
    """
    Evaluate model on a dataset.
    
    Args:
        model: KITTIYOLOModel instance
        dataset: Dataset to evaluate on
        iou_threshold: IoU threshold for considering a detection correct
        conf_threshold: Confidence threshold for detections
        max_dets: Maximum number of detections to consider per image
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize metrics
    metrics = {
        'detections': 0,
        'ground_truths': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'images_processed': 0,
    }
    
    # Confusion matrix (normalized)
    num_classes = sum(1 for v in KITTI_CLASSES.values() if v != -1)
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    # Prepare containers for precision-recall curve
    class_predictions = defaultdict(list)  # Store predictions by class
    class_ground_truths = defaultdict(int)  # Count ground truths by class
    
    # Evaluation loop
    pbar = tqdm(dataset, desc="Evaluating")
    for i, (image, target) in enumerate(pbar):
        # Get image path
        img_name = target['img_name']
        img_path = os.path.join(KITTI_TRAIN_IMAGES, img_name)
        
        # Get original image size for scaling
        orig_size = target['orig_size']
        
        # Get ground truth boxes and labels
        gt_boxes = target['boxes']  # normalized YOLO format
        gt_labels = target['labels']
        
        # Count ground truths (by class)
        metrics['ground_truths'] += len(gt_boxes)
        for label in gt_labels:
            class_ground_truths[int(label)] += 1
        
        # Run inference
        results = model.predict(img_path, conf_thres=conf_threshold, iou_thres=YOLO_CONFIG['iou_thres'])
        
        # Get predictions
        pred = results.xyxyn[0].cpu().numpy()  # First image, normalized xyxy
        
        # Count detections
        pred_boxes = []
        pred_labels = []
        pred_scores = []
        
        # Extract boxes, labels, and scores
        if len(pred) > 0:
            # Sort by confidence and get top max_dets
            pred = pred[pred[:, 4].argsort()[::-1][:max_dets]]
            
            # Get predictions in YOLO format [x_center, y_center, width, height]
            for x1, y1, x2, y2, conf, cls in pred:
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2
                
                pred_boxes.append([x_center, y_center, width, height])
                pred_labels.append(int(cls))
                pred_scores.append(conf)
                
                # Store for precision-recall curve (class, confidence, matched)
                class_predictions[int(cls)].append({
                    'confidence': conf,
                    'matched': False  # Will be updated if matched to a ground truth
                })
        
        # Count predictions
        metrics['detections'] += len(pred_boxes)
        
        # Match predictions to ground truths
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Calculate IoU between predictions and ground truths
            ious = compute_iou(
                np.array(pred_boxes),
                np.array(gt_boxes)
            )
            
            # Match each prediction to ground truth with highest IoU
            matched_gt = set()  # Keep track of matched ground truths
            
            # Sort predictions by confidence
            sorted_indices = np.argsort([score for score in pred_scores])[::-1]
            
            for pred_idx in sorted_indices:
                # Find best matching ground truth
                best_gt_idx = np.argmax(ious[pred_idx])
                best_iou = ious[pred_idx, best_gt_idx]
                
                # Match if IoU is above threshold and not already matched
                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    matched_gt.add(best_gt_idx)
                    metrics['true_positives'] += 1
                    
                    # Mark as matched for precision-recall curve
                    pred_class = pred_labels[pred_idx]
                    for p in class_predictions[pred_class]:
                        if p['confidence'] == pred_scores[pred_idx] and not p['matched']:
                            p['matched'] = True
                            break
                    
                    # Update confusion matrix
                    pred_label = pred_labels[pred_idx]
                    gt_label = gt_labels[best_gt_idx]
                    confusion_matrix[gt_label, pred_label] += 1
                else:
                    metrics['false_positives'] += 1
        else:
            # All predictions are false positives if no ground truths
            metrics['false_positives'] += len(pred_boxes)
            
        # Count false negatives (ground truths with no matching prediction)
        metrics['false_negatives'] += len(gt_boxes) - len(matched_gt) if 'matched_gt' in locals() else len(gt_boxes)
        
        # Update number of processed images
        metrics['images_processed'] += 1
        
        # Update progress bar
        pbar.set_postfix({
            'TP': metrics['true_positives'],
            'FP': metrics['false_positives'],
            'FN': metrics['false_negatives']
        })
    
    # Calculate precision and recall (overall)
    if metrics['true_positives'] + metrics['false_positives'] > 0:
        metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
    else:
        metrics['precision'] = 0
    
    if metrics['true_positives'] + metrics['false_negatives'] > 0:
        metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
    else:
        metrics['recall'] = 0
    
    # Calculate F1 score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    # Calculate precision-recall curve and AP for each class
    class_ap = {}
    all_precisions = []
    all_recalls = []
    
    for cls in sorted(class_ground_truths.keys()):
        # Skip if no ground truths for this class
        if class_ground_truths[cls] == 0:
            class_ap[cls] = 0
            continue
            
        # Get predictions for this class
        predictions = class_predictions[cls]
        
        if len(predictions) == 0:
            class_ap[cls] = 0
            continue
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall
        tp = 0
        fp = 0
        precision = []
        recall = []
        
        for pred in predictions:
            if pred['matched']:
                tp += 1
            else:
                fp += 1
                
            # Calculate precision and recall at this threshold
            curr_precision = tp / (tp + fp)
            curr_recall = tp / class_ground_truths[cls]
            
            precision.append(curr_precision)
            recall.append(curr_recall)
        
        # Calculate average precision
        ap = 0
        for i in range(11):  # 11-point interpolation
            threshold = i / 10
            
            # Find all recalls >= threshold
            indices = [j for j, r in enumerate(recall) if r >= threshold]
            
            if indices:
                max_precision = max([precision[j] for j in indices])
                ap += max_precision
        
        ap /= 11
        class_ap[cls] = ap
        
        # Store for overall curve
        all_precisions.extend(precision)
        all_recalls.extend(recall)
    
    # Calculate mAP
    metrics['mAP'] = sum(class_ap.values()) / len(class_ap) if class_ap else 0
    metrics['class_ap'] = class_ap
    
    # Store precision-recall curves
    metrics['all_precision'] = all_precisions
    metrics['all_recall'] = all_recalls
    
    # Normalize confusion matrix (by ground truths)
    for i in range(confusion_matrix.shape[0]):
        if np.sum(confusion_matrix[i]) > 0:
            confusion_matrix[i] = confusion_matrix[i] / np.sum(confusion_matrix[i])
    
    metrics['confusion_matrix'] = confusion_matrix
    
    return metrics

def main(args):
    """Main evaluation function"""
    print("=" * 80)
    print("KITTI 2D Object Detection Evaluation")
    print("=" * 80)
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('evaluation_results', Path(args.weights).stem)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine data source
    if args.test_dir:
        img_dir = args.test_dir
        label_dir = None
        is_test = True
        print(f"Evaluating on test set: {img_dir}")
    else:
        img_dir = KITTI_TRAIN_IMAGES
        label_dir = KITTI_TRAIN_LABELS
        is_test = False
        print(f"Evaluating on validation set: {img_dir}")
    
    # Load model
    print("\nLoading model...")
    num_classes = sum(1 for v in KITTI_CLASSES.values() if v != -1)
    model = KITTIYOLOModel(num_classes=num_classes)
    model.load_model(args.weights)
    
    # Create dataset
    print("\nPreparing dataset...")
    dataset = KITTIDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=get_validation_augmentations(),
        is_test=is_test
    )
    print(f"Dataset size: {len(dataset)} images")
    
    # Run evaluation
    if not is_test:
        print("\nRunning evaluation...")
        start_time = time.time()
        metrics = evaluate_model(
            model=model,
            dataset=dataset,
            iou_threshold=args.iou_threshold,
            conf_threshold=args.conf_threshold
        )
        eval_time = time.time() - start_time
        
        print("\nEvaluation Results:")
        print(f"- mAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
        print(f"- Precision: {metrics['precision']:.4f}")
        print(f"- Recall: {metrics['recall']:.4f}")
        print(f"- F1 Score: {metrics['f1_score']:.4f}")
        print(f"- Images processed: {metrics['images_processed']}")
        print(f"- Evaluation time: {eval_time:.2f} seconds")
        
        # Generate evaluation report
        print("\nGenerating evaluation report...")
        fig = create_evaluation_report(metrics, save_path=os.path.join(output_dir, 'evaluation_report.png'))
        
        # Save metrics as JSON
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            # Convert numpy arrays to lists
            for key in metrics:
                if isinstance(metrics[key], np.ndarray):
                    metrics[key] = metrics[key].tolist()
                    
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {os.path.join(output_dir, 'metrics.json')}")
    
    # Run inference on sample images
    print("\nRunning inference on sample images...")
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Get sample images (limited number)
    if args.num_samples > 0:
        sample_indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    else:
        sample_indices = range(len(dataset))
    
    for i in sample_indices:
        image, target = dataset[i]
        img_path = os.path.join(img_dir, target['img_name'])
        
        # Run inference
        results = model.predict(img_path, conf_thres=args.conf_threshold)
        
        # Visualize predictions
        fig = visualize_predictions(
            image_path=img_path,
            results=results,
            confidence_threshold=args.conf_threshold
        )
        
        # Save visualization
        fig_path = os.path.join(output_dir, 'samples', f"sample_{i}.png")
        fig.savefig(fig_path)
        plt.close(fig)
    
    print(f"Sample visualizations saved to {os.path.join(output_dir, 'samples')}")
    
    print("\nEvaluation complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on KITTI dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--test_dir", type=str, default=None, help="Test images directory")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sample images for visualization")
    
    args = parser.parse_args()
    main(args)