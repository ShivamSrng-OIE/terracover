"""Deep learning inference module for TerraCover.

This module provides the neural network model and inference pipeline
for satellite imagery semantic segmentation.
"""

import ssl
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .config import log, LANDCOVER

# Disable SSL verification for model weight downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Model constants
_ENCODER_NAME = "resnet50"
_ENCODER_WEIGHTS = "imagenet"
_INPUT_SIZE = 512


def check_device() -> torch.device:
    """Detect and configure the compute device.

    Returns:
        torch.device configured for CUDA if available, otherwise CPU.
    """
    log("=" * 50)
    log("SYSTEM CONFIGURATION")
    log(f"  PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        log("  Device: CPU")

    log("=" * 50)
    return device


class SatelliteLandCoverModel:
    """U-Net segmentation model with ResNet50 encoder.

    This model performs pixel-wise classification of satellite imagery
    using transfer learning from ImageNet pretrained weights.

    Attributes:
        device: Compute device for inference.
        num_classes: Number of output classification classes.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize the model container.

        Args:
            device: Target compute device (CPU or CUDA).
        """
        self.device = device
        self.num_classes = len(LANDCOVER)
        self._model: smp.Unet = None

    def load_model(self) -> None:
        """Build and load the U-Net model with pretrained weights.

        Raises:
            RuntimeError: If model initialization fails.
        """
        try:
            self._model = smp.Unet(
                encoder_name=_ENCODER_NAME,
                encoder_weights=_ENCODER_WEIGHTS,
                in_channels=3,
                classes=self.num_classes,
                activation=None,
            )
            self._model.to(self.device)
            self._model.eval()
            log(f"Model initialized: U-Net ({_ENCODER_NAME})")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image: RGB image array of shape (H, W, 3).

        Returns:
            Normalized tensor of shape (1, 3, 512, 512).
        """
        resized = cv2.resize(image, (_INPUT_SIZE, _INPUT_SIZE))
        tensor = torch.from_numpy(resized).float().div(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict_probabilities(self, image: np.ndarray) -> np.ndarray:
        """Generate class probability maps with test-time augmentation.

        Performs inference on the original image plus horizontal and
        vertical flips, then averages predictions for improved robustness.

        Args:
            image: RGB image array of shape (H, W, 3).

        Returns:
            Probability array of shape (num_classes, H, W) with values [0, 1].
        """
        h, w = image.shape[:2]

        # Prepare augmented batch
        images = [
            image,
            np.fliplr(image).copy(),
            np.flipud(image).copy(),
        ]
        batch = torch.cat([self._preprocess(img) for img in images], dim=0)

        # Inference
        with torch.no_grad():
            logits = self._model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        # Reverse augmentations and average
        probs[1] = probs[1, :, :, ::-1]  # Undo horizontal flip
        probs[2] = probs[2, :, ::-1, :]  # Undo vertical flip
        avg_probs = probs.mean(axis=0)

        # Resize to original resolution
        result = np.array([
            cv2.resize(avg_probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
            for c in range(self.num_classes)
        ])

        return result
