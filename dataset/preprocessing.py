import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from utils.logger import get_dataset_logger


logger = get_dataset_logger()


def get_transforms(image_size=(224, 224)):
    """
    Get train and validation transforms

    Args:
        image_size: Size to resize images to

    Returns:
        Dictionary containing train and val transforms
    """
    # Standard transforms for CLIP
    train_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ToTensorV2(),
    ])

    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform  # Same as val for testing
    }


def preprocess_image(image_path, transform=None):
    """
    Preprocess a single image

    Args:
        image_path: Path to the image
        transform: Albumentations transform to apply

    Returns:
        Preprocessed image tensor
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        # Return a dummy image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform:
        augmented = transform(image=image)
        image = augmented['image']

    return image


def clean_text(text):
    """
    Clean and preprocess text

    Args:
        text: Input text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)

    # Basic cleaning
    text = text.strip()
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text