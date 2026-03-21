"""Albumentations-based augmentation pipeline for synthetic OCR images."""

import albumentations as A
import numpy as np


def get_augmentation_pipeline(
    blur_limit: int = 5,
    noise_var_limit: float = 25.0,
    jpeg_quality_lower: int = 50,
    rotation_limit: int = 5,
    perspective_scale: float = 0.05,
    brightness_limit: float = 0.3,
    contrast_limit: float = 0.3,
) -> A.Compose:
    """Build the augmentation pipeline.

    All transforms are applied with moderate probability so that many samples
    remain clean (easier) while others get heavy augmentation (harder).
    """
    return A.Compose([
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0),
            A.MotionBlur(blur_limit=(3, blur_limit), p=1.0),
        ], p=0.3),

        # Noise
        A.GaussNoise(std_range=(0, noise_var_limit / 255.0), p=0.2),

        # JPEG compression
        A.ImageCompression(quality_range=(jpeg_quality_lower, 100), p=0.2),

        # Geometric
        A.Affine(
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            scale=(0.95, 1.05),
            rotate=(-rotation_limit, rotation_limit),
            mode=0,  # constant border
            p=0.3,
        ),
        A.Perspective(scale=perspective_scale, p=0.2),

        # Color / brightness
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.3,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.2,
        ),

        # Shadow simulation (random rectangular darkening)
        A.RandomShadow(
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            shadow_roi=(0, 0, 1, 1),
            p=0.15,
        ),
    ])


def apply_augmentation(
    image: np.ndarray,
    pipeline: A.Compose,
) -> np.ndarray:
    """Apply augmentation pipeline to an image (HWC uint8 numpy array)."""
    result = pipeline(image=image)
    return result["image"]
