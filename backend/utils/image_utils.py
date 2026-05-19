"""
utils/image_utils.py
====================
Preprocessing utility for the leaf disease CNN model input.

Adjust IMAGE_SIZE if your model uses a different input resolution.
"""

import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)  # Must match the model's expected input shape


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalise a PIL image for CNN prediction.

    Args:
        image: PIL Image in RGB mode.

    Returns:
        np.ndarray of shape (1, HEIGHT, WIDTH, 3), values in [0, 1].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = image.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
