
import os
import sys
import argparse
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random

def get_color_palette(num_classes):
    """Generate a beautiful color palette for classes"""
    # Bright, distinct colors
    base_colors = [
        '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', 
        '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', 
        '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', 
        '#FF95C8', '#FF37C7'
    ]
    
    # If we have more classes than base colors, generate additional ones
    if num_classes <= len(base_colors):
        return base_colors[:num_classes]
    else:
        # Generate additional colors using HSV color space for better distinction
        colors = base_colors.copy()
        for i in range(len(base_colors), num_classes):
            hue = i / num_classes
            saturation = 0.8 + random.uniform(-0.1, 0.1)  # High saturation with small variation
            value = 0.9 + random.uniform(-0.1, 0.1)  # High value with small variation
            colors.append(plt.cm.hsv(hue))
        return colors

def process_image(model, image_path, conf_threshold=0.25, class_colors=None, output_dir=None, show=False):
    """Run inference on a single image and visualize results with multi-color boxes"""
    print(f"Processing image: {image_path}")
    start_time = time.time()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return None
    
    # Run inference
    try:
        results = model(image_path, conf=conf_threshold)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")
        
        # Get number of detections
        num_detections = len(results[0].boxes) if results and hasattr(results[0], 'boxes') else 0
        print(f"Found {num_detections} objects")
        
        # Create visualization
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 9), dpi=100)
        ax.imshow(img)
        
        # Class names for model (using COCO classes for pre-trained model)
        class_names = model.names
        
        # Draw beautiful boxes with different colors per class
        if num_detections > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Use class color
                color = class_colors[cls_id % len(class_colors)]
                
                # Draw rectangle with a bit of transparency
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1, 
                    linewidth=3, 
                    edgecolor=color,
                    facecolor='none',
                    alpha=0.9
                )
                ax.add_patch(rect)
                
                # Create fancy label background
                class_name = class_names[cls_id] if cls_id in class_names else f"Class {cls_id}"
                text = f"{class_name} {conf:.2f}"
                text_width, text_height = ax.transData.inverted().transform(
                    ax.transAxes.transform((1, 0)) - 
                    ax.transAxes.transform((0, 0))
                )[0] * len(text) * 0.55, 25
                
                # Add label background with rounded corners
                label_rect = patches.Rectangle(
                    (x1, y1 - text_height - 2),
                    text_width,
                    text_height,
                    facecolor=color,
                    alpha=0.85,
                    transform=ax.transData
                )
                ax.add_patch(label_rect)
                
                # Add label text
                ax.text(
                    x1 + 5, y1 - 7,
                    text,
                    color='white',
                    fontsize=12,
                    fontweight='bold',
                    va='center'
                )
        
        # Remove axis
        ax.set_axis_off()
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Add fancy title with model info and stats
        plt.figtext(
            0.5, 0.01,
            f"KITTI Object Detection | {os.path.basename(image_path)} | {num_detections} objects detected | {inference_time:.3f}s",
            ha="center", 
            fontsize=12,
            bbox={"facecolor":"black", "alpha":0.7, "pad":5, "edgecolor":"white"},
            color="white"
        )
        
        # Save output if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            
            # Save raw predictions to text file
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_path, "w") as f:
                if num_detections > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        f.write(f"{cls_id} {x1} {y1} {x2} {y2} {conf}\n")
            
            print(f"Output saved to {output_path}")
        
        # Show result if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return results
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Run beautiful object detection with color-coded boxes")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='results/colorful_detections', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--max_images', type=int, default=30, help='Maximum number of images to process')
    parser.add_argument('--show', action='store_true', help='Show results')
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("KITTI Object Detection with Beautiful Visualization")
    print("=" * 80)
    print(f"Model weights: {args.weights}")
    print(f"Input path: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Max images: {args.max_images}")
    print("=" * 80)
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        print(f"Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Generate color palette for classes
    num_classes = 80  # COCO has 80 classes for pre-trained model
    class_colors = get_color_palette(num_classes)
    
    # Determine input type (image or directory)
    if os.path.isdir(args.input):
        # Process all images in directory
        print(f"\nProcessing directory: {args.input}")
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.input, f"*.{ext}")))
        
        print(f"Found {len(image_files)} images")
        
        # Process a limited number of images by default
        max_images = min(len(image_files), args.max_images)
        for i, image_path in enumerate(image_files[:max_images]):
            print(f"Processing image {i+1}/{max_images}: {image_path}")
            process_image(
                model=model,
                image_path=image_path,
                conf_threshold=args.conf,
                class_colors=class_colors,
                output_dir=output_dir,
                show=args.show
            )
    else:
        # Process single image
        process_image(
            model=model,
            image_path=args.input,
            conf_threshold=args.conf,
            class_colors=class_colors,
            output_dir=output_dir,
            show=args.show
        )
    
    print("\nInference complete!")
    print(f"Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
