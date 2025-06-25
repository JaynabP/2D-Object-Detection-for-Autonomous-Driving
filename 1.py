import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns

def plot_precision_recall_curves(predictions_file, ground_truth_file, class_names, output_path):
    """
    Generate precision-recall curves for each object class.
    
    Args:
        predictions_file: Path to the YOLOv5 predictions file (can be from val.py output)
        ground_truth_file: Path to the ground truth annotations
        class_names: List of class names
        output_path: Path to save the output figure
    """
    # Load predictions and ground truth
    # This assumes your predictions are in the format: [image_id, class_id, confidence, x1, y1, x2, y2]
    # You may need to adapt this to match your data format
    preds = pd.read_csv(predictions_file, header=None, sep=' ')
    gt = pd.read_csv(ground_truth_file, header=None, sep=' ')
    
    plt.figure(figsize=(12, 10))
    
    # Create colormap for different classes
    colors = plt.cm.jet(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        # Extract predictions and ground truth for this class
        class_preds = preds[preds[1] == i]
        class_gt = gt[gt[1] == i]
        
        # Get all images with this class
        all_images = set(class_preds[0].unique()).union(set(class_gt[0].unique()))
        
        # Prepare arrays for precision-recall calculation
        y_true = []  # Binary array indicating if detection is correct
        y_scores = []  # Confidence scores
        
        # For each image
        for img_id in all_images:
            img_preds = class_preds[class_preds[0] == img_id]
            img_gt = class_gt[class_gt[0] == img_id]
            
            # Get ground truth boxes
            gt_boxes = img_gt.iloc[:, 3:7].values if len(img_gt) > 0 else np.array([])
            
            # For each prediction in this image
            for _, pred in img_preds.iterrows():
                conf_score = pred[2]
                pred_box = pred[3:7].values
                
                # Check if this prediction matches any ground truth
                is_match = False
                for gt_box in gt_boxes:
                    # Calculate IoU
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= 0.5:  # IoU threshold
                        is_match = True
                        break
                
                y_true.append(int(is_match))
                y_scores.append(conf_score)
            
            # Add false negatives (ground truth with no matching prediction)
            # This is simplified and might need adjustment for your exact evaluation protocol
            for _ in range(len(img_gt) - sum([1 for _, row in img_preds.iterrows() if row[1] == i])):
                if len(img_preds) > 0:  # Only if we have predictions to compare with
                    y_true.append(1)
                    y_scores.append(0.0)  # Lowest confidence
        
        # Calculate precision-recall curve
        if len(y_true) > 0 and sum(y_true) > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            # Plot precision-recall curve
            plt.plot(recall, precision, lw=2, color=colors[i], 
                    label=f'{class_name} (AP: {ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Class')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-recall curves saved to {output_path}")

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    # Determine coordinates of intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

# Example usage:
# class_names = ["Car", "Pedestrian", "Cyclist", "Truck", "Traffic Light", "Traffic Sign"]
# plot_precision_recall_curves("predictions.txt", "ground_truth.txt", class_names, "precision_recall_curves.png")