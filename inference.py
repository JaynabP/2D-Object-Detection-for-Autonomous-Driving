"""
Inference script for KITTI 2D object detection using YOLO.
Run model on new images and visualize results.
"""
import os
import sys
import argparse
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import project modules
from src.models.yolo import KITTIYOLOModel
from src.utils.visualization import visualize_predictions
from src.config import KITTI_CLASSES, IMG_SIZE, YOLO_CONFIG

def process_image(model, image_path, conf_threshold=0.25, output_dir=None, show=True):
    """
    Run inference on a single image and visualize results.
    
    Args:
        model: Trained YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        output_dir: Directory to save output visualization
        show: Whether to display the result
        
    Returns:
        Tuple of (results, visualization)
    """
    print(f"Processing image: {image_path}")
    start_time = time.time()
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return None, None
    
    # Run inference
    try:
        results = model.predict(image_path, conf_thres=conf_threshold)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f} seconds")
        
        # Get number of detections (handle new results format)
        if results is None:
            print("No results returned from model.")
            return None, None
            
        # Process the results from the Ultralytics YOLO model
        num_detections = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
        print(f"Found {num_detections} objects")
        
        # Create a figure to visualize results
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw boxes
        if num_detections > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Draw rectangle
                class_names = list(KITTI_CLASSES.keys())
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                  edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                ax.text(x1, y1-5, f"{class_name} {conf:.2f}", 
                        color='white', fontsize=10, 
                        bbox=dict(facecolor='red', alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save output if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
            fig.savefig(output_path, bbox_inches='tight')
            print(f"Output saved to {output_path}")
        
        # Show result if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return results, fig
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_video(model, video_path, conf_threshold=0.25, output_path=None, fps=30):
    """
    Process a video file frame by frame.
    
    Args:
        model: Trained YOLO model
        video_path: Path to input video
        conf_threshold: Confidence threshold for detections
        output_path: Path to save output video
        fps: Frames per second for output video
        
    Returns:
        Path to output video
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist")
        return None
    
    print(f"Processing video: {video_path}")
    start_time = time.time()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer if requested
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame to temporary file (YOLOv5 expects a file path)
        temp_frame_path = 'temp_frame.jpg'
        cv2.imwrite(temp_frame_path, frame)
        
        # Run inference
        try:
            results = model.predict(temp_frame_path, conf_thres=conf_threshold)
            
            # Process results with new format
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                # Draw boxes on frame
                for box in boxes:
                    # Get box coordinates (convert to int for drawing)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Get class name
                    class_names = list(KITTI_CLASSES.keys())
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
        
        # Write frame to output video
        if output_path:
            out.write(frame)
        
        # Increment frame counter
        frame_idx += 1
        
        # Print progress
        if frame_idx % 10 == 0:
            progress = frame_idx / total_frames * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}, {elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
    
    # Clean up
    cap.release()
    if output_path:
        out.release()
    
    # Remove temporary file
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"Video processing complete: {total_frames} frames in {total_time:.1f}s ({total_frames/total_time:.1f} FPS)")
    
    return output_path

def main(args):
    """Main inference function"""
    print("=" * 80)
    print("KITTI 2D Object Detection Inference")
    print("=" * 80)
    
    # Set up output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
    
    # Load model
    print("\nLoading model...")
    num_classes = sum(1 for v in KITTI_CLASSES.values() if v != -1)
    model = KITTIYOLOModel(num_classes=num_classes)
    model.load_model(args.weights)
    
    # Determine input type (image, directory, or video)
    if os.path.isdir(args.input):
        # Process all images in directory
        print(f"\nProcessing directory: {args.input}")
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(args.input, f"*.{ext}")))
        
        print(f"Found {len(image_files)} images")
        
        for image_path in image_files:
            process_image(
                model=model,
                image_path=image_path,
                conf_threshold=args.conf_threshold,
                output_dir=output_dir,
                show=args.show
            )
    elif any(args.input.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov']):
        # Process video
        if output_dir:
            output_path = os.path.join(output_dir, os.path.basename(args.input))
        else:
            output_path = None
            
        process_video(
            model=model,
            video_path=args.input,
            conf_threshold=args.conf_threshold,
            output_path=output_path,
            fps=args.fps
        )
    else:
        # Process single image
        process_image(
            model=model,
            image_path=args.input,
            conf_threshold=args.conf_threshold,
            output_dir=output_dir,
            show=args.show
        )
    
    print("\nInference complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--input", type=str, required=True, help="Path to input image, video, or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--fps", type=int, default=30, help="FPS for video output")
    
    args = parser.parse_args()
    main(args)