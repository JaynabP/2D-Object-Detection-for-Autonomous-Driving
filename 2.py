import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_confidence_distribution(predictions_file, class_names, output_path, bins=20):
    """
    Generate confidence score distribution histogram.
    
    Args:
        predictions_file: Path to the YOLOv5 predictions file
        class_names: List of class names
        output_path: Path to save the output figure
        bins: Number of bins for the histogram
    """
    # Load predictions
    preds = pd.read_csv(predictions_file, header=None, sep=' ')
    
    plt.figure(figsize=(12, 8))
    
    # Plot overall confidence distribution
    plt.subplot(2, 1, 1)
    sns.histplot(preds[2], bins=bins, kde=True, color='blue')
    plt.title('Overall Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot confidence distribution by class
    plt.subplot(2, 1, 2)
    
    # Create colormap for different classes
    colors = plt.cm.jet(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        # Get predictions for this class
        class_preds = preds[preds[1] == i]
        if len(class_preds) > 0:
            sns.kdeplot(class_preds[2], label=class_name, color=colors[i], fill=True, alpha=0.3)
    
    plt.title('Confidence Score Distribution by Class')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confidence distribution histogram saved to {output_path}")

# Example usage:
# class_names = ["Car", "Pedestrian", "Cyclist", "Truck", "Traffic Light", "Traffic Sign"]
# plot_confidence_distribution("predictions.txt", class_names, "confidence_distribution.png")