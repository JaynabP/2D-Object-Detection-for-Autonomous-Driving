
import os
import glob
import random
from datetime import datetime

def main():
    # Create report directory
    report_dir = 'results/beautiful_report'
    os.makedirs(report_dir, exist_ok=True)
    
    # Find detection images
    detection_images = glob.glob('results/colorful_detections/*_detection.jpg')
    
    # Check if we have any images
    if not detection_images:
        print("No detection images found in 'results/colorful_detections/'")
        return 1
    
    # Count total objects detected
    total_objects = 0
    class_counts = {}
    
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
    
    for img_path in glob.glob('results/colorful_detections/*.txt'):
        try:
            with open(img_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id = int(float(parts[0]))
                        class_name = class_names[class_id] if class_id in class_names else f"Class {class_id}"
                        
                        total_objects += 1
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get some sample images for the gallery
    samples = random.sample(detection_images, min(12, len(detection_images)))
    
    # Create the HTML report with modern design
    with open(os.path.join(report_dir, 'index.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KITTI Object Detection Results</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3b82f6;
            --secondary-color: #10b981;
            --background-color: #f8fafc;
            --text-color: #334155;
            --card-bg: #ffffff;
            --accent-color: #6366f1;
            --border-radius: 8px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            color: white;
            padding: 30px 20px;
            border-radius: var(--border-radius);
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        h2 {
            font-size: 1.8rem;
            color: var(--primary-color);
            margin: 40px 0 20px 0;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
        }
        
        h3 {
            font-size: 1.4rem;
            color: var(--secondary-color);
            margin: 20px 0 15px 0;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background-color: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .stat-card h3 {
            font-size: 1.2rem;
            margin-bottom: 10px;
            color: var(--text-color);
        }
        
        .stat-card .value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .stat-card .label {
            font-size: 0.9rem;
            color: #64748b;
        }
        
        .chart-container {
            background-color: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: var(--border-radius);
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .gallery-item {
            background-color: var(--card-bg);
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
        }
        
        .gallery-item:hover {
            transform: scale(1.03);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        
        .gallery-item img {
            width: 100%;
            display: block;
            aspect-ratio: 16/9;
            object-fit: cover;
        }
        
        .gallery-item .caption {
            padding: 15px;
            font-size: 0.9rem;
        }
        
        .class-distribution {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .class-item {
            background-color: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .class-item .class-name {
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .class-item .class-count {
            font-size: 1.8rem;
            font-weight: bold;
        }
        
        .class-item .class-percent {
            font-size: 0.9rem;
            color: #64748b;
        }
        
        footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.9rem;
            color: #64748b;
            padding: 20px;
            border-top: 1px solid #e2e8f0;
        }
        
        .badge {
            background-color: var(--primary-color);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .gallery {
                grid-template-columns: repeat(2, 1fr);
            }
            
            h1 {
                font-size: 2rem;
            }
        }
        
        @media (max-width: 480px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .gallery {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>KITTI Object Detection Results</h1>
            <p>Advanced visualization of object detection on the KITTI dataset</p>
            <p>Generated on ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        </header>
        
        <section>
            <h2>Detection Overview</h2>
            
            <div class="dashboard">
                <div class="stat-card">
                    <h3>Total Detections</h3>
                    <div class="value">''' + str(total_objects) + '''</div>
                    <div class="label">Objects detected</div>
                </div>
                
                <div class="stat-card">
                    <h3>Images Processed</h3>
                    <div class="value">''' + str(len(detection_images)) + '''</div>
                    <div class="label">Test images</div>
                </div>
                
                <div class="stat-card">
                    <h3>Most Common Class</h3>
                    <div class="value">''' + (sorted_classes[0][0] if sorted_classes else "N/A") + '''</div>
                    <div class="label">''' + (f"{sorted_classes[0][1]} detections" if sorted_classes else "") + '''</div>
                </div>
                
                <div class="stat-card">
                    <h3>Classes Detected</h3>
                    <div class="value">''' + str(len(class_counts)) + '''</div>
                    <div class="label">Different object types</div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Class Distribution</h2>
            
            <div class="chart-container">
                <img src="../enhanced_summary/class_distribution.png" alt="Class Distribution Chart">
            </div>
            
            <div class="class-distribution">''')
        
        # Add class distribution cards
        for class_name, count in sorted_classes:
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            f.write(f'''
                <div class="class-item">
                    <div class="class-name">{class_name}</div>
                    <div class="class-count">{count}</div>
                    <div class="class-percent">{percentage:.1f}% of total</div>
                </div>''')
        
        f.write('''
            </div>
        </section>
        
        <section>
            <h2>Confidence Analysis</h2>
            
            <div class="chart-container">
                <img src="../enhanced_summary/confidence_distribution.png" alt="Confidence Distribution">
            </div>''')
        
        # Add confidence by class if it exists
        if os.path.exists('results/enhanced_summary/confidence_by_class.png'):
            f.write('''
            <div class="chart-container">
                <img src="../enhanced_summary/confidence_by_class.png" alt="Confidence by Class">
            </div>''')
        
        f.write('''
        </section>
        
        <section>
            <h2>Detection Gallery</h2>
            <p>Sample of detected objects across various test images</p>
            
            <div class="gallery">''')
        
        # Add sample images
        for img_path in samples:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(os.path.splitext(img_name)[0])[0]  # Remove _detection.jpg
            rel_path = os.path.relpath(img_path, report_dir)
            
            # Count objects in this image
            count = 0
            txt_path = os.path.join('results/colorful_detections', f"{base_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as txt_file:
                    count = len(txt_file.readlines())
            
            f.write(f'''
                <div class="gallery-item">
                    <img src="../colorful_detections/{img_name}" alt="{base_name}">
                    <div class="caption">
                        <strong>{base_name}</strong><br>
                        {count} objects detected
                    </div>
                </div>''')
        
        f.write('''
            </div>
        </section>
        
        <footer>
            <p>KITTI Object Detection Report | Generated with Ultralytics YOLOv5</p>
        </footer>
    </div>
</body>
</html>''')
    
    print(f"Beautiful HTML report created at {os.path.join(report_dir, 'index.html')}")
    return 0

if __name__ == "__main__":
    main()
