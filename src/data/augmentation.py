"""
Data augmentation utilities for the KITTI object detection dataset.
Uses albumentations library for efficient image augmentations.
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import IMG_SIZE, MEAN, STD, AUG_PARAMS

def get_training_augmentations():
    """
    Returns the augmentation pipeline for training data.
    
    Includes:
    - Resizing
    - Random horizontal flip
    - Random brightness and contrast
    - Random blur
    - Random rotation
    - Normalization
    - Conversion to PyTorch tensor
    """
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE[1], width=IMG_SIZE[0], always_apply=True),
            A.HorizontalFlip(p=AUG_PARAMS['horizontal_flip']),
            A.RandomBrightnessContrast(
                brightness_limit=AUG_PARAMS['brightness_contrast'],
                contrast_limit=AUG_PARAMS['brightness_contrast'],
                p=0.5
            ),
            A.Blur(blur_limit=AUG_PARAMS['blur_limit'], p=0.3),
            A.Rotate(limit=AUG_PARAMS['rotate_limit'], p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels']
        )
    )

def get_validation_augmentations():
    """
    Returns the augmentation pipeline for validation data.
    
    Includes only:
    - Resizing
    - Normalization
    - Conversion to PyTorch tensor
    """
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE[1], width=IMG_SIZE[0], always_apply=True),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels']
        )
    )

def get_test_augmentations():
    """
    Returns the augmentation pipeline for test data.
    
    Same as validation augmentations, but without bbox params
    since test data might not have labels.
    """
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE[1], width=IMG_SIZE[0], always_apply=True),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]
    )

def visualize_augmentations(image, boxes, labels, augment_fn, num_samples=5):
    """
    Visualizes the effect of augmentations on images with bounding boxes.
    
    Args:
        image: Original image (numpy array)
        boxes: Bounding boxes in YOLO format
        labels: Class labels for each box
        augment_fn: Augmentation function to apply
        num_samples: Number of augmented samples to generate
        
    Returns:
        list of augmented images with drawn bounding boxes
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches
    
    augmented_images = []
    
    for i in range(num_samples):
        # Apply augmentations
        augmented = augment_fn(image=image.copy(), bboxes=boxes.copy(), class_labels=labels.copy())
        aug_image = augmented['image']
        aug_boxes = augmented['bboxes']
        aug_labels = augmented['class_labels']
        
        # Convert tensor to numpy for visualization
        if torch.is_tensor(aug_image):
            aug_image = aug_image.permute(1, 2, 0).cpu().numpy()
            
            # Un-normalize
            aug_image = aug_image * np.array(STD) + np.array(MEAN)
            aug_image = np.clip(aug_image, 0, 1)
        
        # Create figure for visualization
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(aug_image)
        
        # Convert YOLO format to [x1, y1, w, h] for matplotlib
        for box, label in zip(aug_boxes, aug_labels):
            x_center, y_center, width, height = box
            x1 = (x_center - width / 2) * IMG_SIZE[0]
            y1 = (y_center - height / 2) * IMG_SIZE[1]
            w = width * IMG_SIZE[0]
            h = height * IMG_SIZE[1]
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), w, h, 
                linewidth=2, 
                edgecolor='r', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x1, y1, 
                f"Class {label}", 
                color='white', 
                fontsize=12, 
                backgroundcolor='red'
            )
        
        ax.set_title(f"Augmented Sample {i+1}")
        augmented_images.append(fig)
        
    return augmented_images