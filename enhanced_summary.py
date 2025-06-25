
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter

def main():
    # Create the summary directory
    summary_dir = 'results/enhanced_summary'
    os.makedirs(summary_dir, exist_ok=True)
    
    # Find all prediction files
    pred_files = glob.glob('results/colorful_detections/*.txt')
    print(f"Found {len(pred_files)} prediction files")
    
    # Collect statistics
    class_counts = Counter()
    confidences = []
    boxes_by_class = {}
    class_confidence = {}
    
    # Load model class names from COCO (for pre-trained model)
    try:
        from ultralytics import YOLO
        model = YOLO('yolov5su.pt')
        class_names = model.names
    except:
        # Fallback to COCO class names
        class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
            79: 'toothbrush'
        }
    
    # Process prediction files
    for pred_file in pred_files:
        if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            class_id = int(float(parts[0]))
                            class_name = class_names.get(class_id, f"Class {class_id}")
                            x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            conf = float(parts[5])
                            
                            # Update class count
                            class_counts[class_name] += 1
                            
                            # Collect confidence
                            confidences.append(conf)
                            
                            # Store confidence by class
                            if class_name not in class_confidence:
                                class_confidence[class_name] = []
                            class_confidence[class_name].append(conf)
                            
                        except Exception as e:
                            print(f"Error processing line: {line} - {e}")
    
    # Convert data for plotting
    if class_counts:
        # Sort by count
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_labels = [item[0] for item in sorted_counts]
        counts = [item[1] for item in sorted_counts]
        
        # Create DataFrame for plotting
        df_counts = pd.DataFrame({'Class': class_labels, 'Count': counts})
        
        # Plot class distribution with enhanced style
        plt.figure(figsize=(14, 8))
        
        # Create bar plot with pandas/matplotlib instead of directly with seaborn
        ax = plt.bar(range(len(df_counts)), df_counts['Count'], color=sns.color_palette("husl", len(df_counts)))
        plt.xticks(range(len(df_counts)), df_counts['Class'], rotation=45, ha='right')
        
        # Add count labels on top of bars
        for i, count in enumerate(counts):
            plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
            
        plt.title('Object Detection Distribution by Class', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Number of Detections', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(summary_dir, 'class_distribution.png'), dpi=300)
        plt.close()
        
        # Plot confidence distribution without KDE to avoid version issues
        plt.figure(figsize=(12, 7))
        
        # Create histogram
        plt.hist(confidences, bins=20, color='#3182bd', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add mean and median lines
        avg_conf = np.mean(confidences)
        median_conf = np.median(confidences)
        plt.axvline(x=avg_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_conf:.3f}')
        plt.axvline(x=median_conf, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_conf:.3f}')
        
        plt.title('Detection Confidence Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(summary_dir, 'confidence_distribution.png'), dpi=300)
        plt.close()
        
        # Confidence by class boxplot if we have enough data
        if class_confidence and any(len(confs) >= 5 for confs in class_confidence.values()):
            # Create DataFrame
            conf_data = []
            for cls, confs in class_confidence.items():
                if len(confs) >= 5:  # Only include classes with at least 5 detections
                    for conf in confs:
                        conf_data.append({'Class': cls, 'Confidence': conf})
            
            if conf_data:
                df_conf = pd.DataFrame(conf_data)
                
                # Create simple boxplot with matplotlib
                plt.figure(figsize=(14, 8))
                
                # Group by class and create boxplot
                df_pivot = df_conf.pivot(columns='Class', values='Confidence')
                df_pivot.boxplot(grid=True, figsize=(14,8))
                
                plt.title('Confidence Score Distribution by Class', fontsize=16, fontweight='bold')
                plt.xlabel('Class', fontsize=14)
                plt.ylabel('Confidence Score', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, 'confidence_by_class.png'), dpi=300)
                plt.close()
        
        # Print summary statistics
        total = sum(counts)
        print("\n" + "="*50)
        print("DETECTION SUMMARY STATISTICS")
        print("="*50)
        print(f"Total objects detected: {total}")
        
        for i, (cls_name, count) in enumerate(sorted_counts):
            pct = 100 * count / total
            avg_conf = np.mean(class_confidence.get(cls_name, [0])) if cls_name in class_confidence else 0
            print(f"  {cls_name}: {count} detections ({pct:.1f}%) - Avg conf: {avg_conf:.3f}")
        
        print("\nConfidence Statistics:")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Median confidence: {np.median(confidences):.3f}")
        print(f"  Min confidence: {min(confidences):.3f}")
        print(f"  Max confidence: {max(confidences):.3f}")
        print("="*50)
    else:
        print("No detections found in the provided files.")
    
    print(f"\nEnhanced summary visualizations saved to {summary_dir}/")
    return 0

if __name__ == "__main__":
    main()
