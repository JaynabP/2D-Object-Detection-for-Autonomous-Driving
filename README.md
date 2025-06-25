# KITTI 2D Object Detection with YOLOv5

This project implements object detection on the KITTI dataset using YOLOv5. It provides a complete pipeline for training, evaluating, and deploying models for autonomous vehicle perception.

## Project Structure

```
machine-learning-project/
├── data/                   # Dataset storage
│   └── kitti/              # KITTI dataset
│       ├── training/       # Training data (images, labels, calibration)
│       └── testing/        # Testing data
├── models/                 # Model storage
│   └── checkpoints/        # Saved model weights
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code
│   ├── data/               # Data handling modules
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── train.py                # Training script
├── evaluate.py             # Evaluation script
└── inference.py            # Inference script
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kitti-object-detection.git
cd kitti-object-detection
```

2. Create and activate a virtual environment:
```bash
python3 -m venv kitti-env
source kitti-env/bin/activate  # On Windows: kitti-env\Scripts\activate
```

3. Install dependencies:
```bash
python3 -m pip install "torch>=1.10.0" "torchvision>=0.11.0"
python3 -m pip install "numpy>=1.20.0" "pandas>=1.3.0" "pillow>=8.3.0" "opencv-python>=4.5.3" "albumentations>=1.0.0"
python3 -m pip install "matplotlib>=3.4.0" "seaborn>=0.11.0"
python3 -m pip install "scikit-learn>=0.24.0" "pycocotools>=2.0.2"
python3 -m pip install "tqdm>=4.62.0" "pyyaml>=6.0" "tensorboard>=2.6.0"
python3 -m pip install "jupyterlab>=3.0.0"
```

4. Download the KITTI dataset:
   - Visit [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php)
   - Register and download:
     * Left color images of object data set (12 GB)
     * Training labels of object data set (5 MB)
     * Camera calibration matrices (16 MB)
   - Extract to the `data/kitti/` directory

## Data Preparation

Prepare the KITTI dataset for training:

```bash
python -m src.data.preprocessing --output-dir data/processed
```

This script:
- Splits data into training and validation sets
- Converts KITTI format annotations to YOLO format
- Creates the necessary directory structure
- Generates a YAML configuration file for YOLOv5

## Training

Train the YOLOv5 model on the KITTI dataset:

```bash
python train.py --batch_size 16 --epochs 50
```

Optional arguments:
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--output_dir`: Output directory for saving results

## Evaluation

Evaluate a trained model on the validation set:

```bash
python evaluate.py --weights path/to/best.pt
```

Optional arguments:
- `--weights`: Path to model weights (required)
- `--output_dir`: Directory to save evaluation results
- `--iou_threshold`: IoU threshold for evaluation (default: 0.5)
- `--conf_threshold`: Confidence threshold (default: 0.25)
- `--num_samples`: Number of sample images for visualization (default: 10)

## Inference

Run inference on new images:

```bash
python inference.py --weights path/to/best.pt --input path/to/image.jpg
```

Optional arguments:
- `--weights`: Path to model weights (required)
- `--input`: Path to input image, video, or directory (required)
- `--output`: Output directory
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--show`: Show results (default: False)

## Notebooks

Explore the dataset and analyze results using Jupyter notebooks:

```bash
jupyter lab notebooks/
```

Available notebooks:
- `data_exploration.ipynb`: Explore the KITTI dataset
- `model_evaluation.ipynb`: Analyze model performance

## Results

The model performance is evaluated using the following metrics:
- Mean Average Precision (mAP@0.5)
- Precision
- Recall
- F1 Score

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- [YOLOv5](https://github.com/ultralytics/yolov5)