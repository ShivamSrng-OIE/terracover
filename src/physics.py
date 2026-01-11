"""Physics-based color and texture analysis module for TerraCover.

This module implements probability estimation using HSV color space
analysis and texture features for land cover classification.
"""

import cv2
import numpy as np

from .config import LANDCOVER, PHYSICS

# Texture analysis constants
_TEXTURE_KERNEL_SIZE = 7
_TEXTURE_NORMALIZATION_FACTOR = 50.0


def get_color_probabilities(image: np.ndarray) -> np.ndarray:
    """Generate probability maps from color and texture analysis.

    Analyzes the input image in HSV color space and computes per-pixel
    probabilities for each land cover class based on configured thresholds.

    Args:
        image: RGB image array of shape (H, W, 3).

    Returns:
        Probability array of shape (num_classes, H, W) with values [0, 1].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Compute texture (local standard deviation)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (_TEXTURE_KERNEL_SIZE, _TEXTURE_KERNEL_SIZE))
    sqr_mean = cv2.blur(gray ** 2, (_TEXTURE_KERNEL_SIZE, _TEXTURE_KERNEL_SIZE))
    texture = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0))

    # Normalize features to [0, 1]
    h_norm = h.astype(np.float32)
    s_norm = s.astype(np.float32) / 255.0
    v_norm = v.astype(np.float32) / 255.0
    tex_norm = np.clip(texture / _TEXTURE_NORMALIZATION_FACTOR, 0, 1)

    # Initialize probability maps
    num_classes = len(LANDCOVER)
    probs = np.zeros((num_classes, image.shape[0], image.shape[1]), dtype=np.float32)

    # Compute per-class probabilities
    probs[LANDCOVER["Water"]["id"]] = _water_probability(h_norm, s_norm, v_norm, tex_norm)
    probs[LANDCOVER["Trees_Woodland"]["id"]] = _vegetation_probability(h_norm, s_norm)
    probs[LANDCOVER["Roads_Paths"]["id"]] = _road_probability(s_norm, v_norm, tex_norm)
    probs[LANDCOVER["Human_Habitat"]["id"]] = _habitat_probability(h_norm, s_norm, v_norm, tex_norm)

    return probs


def _water_probability(h: np.ndarray, s: np.ndarray, v: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Compute water probability based on blue hue and smoothness."""
    cfg = PHYSICS["water"]

    hue_score = np.exp(-((h - cfg["hue_center"]) ** 2) / (2 * cfg["hue_width"] ** 2))
    sat_score = 1 / (1 + np.exp(-12 * (s - cfg["min_saturation"])))
    val_score = 1 - (1 / (1 + np.exp(-10 * (v - cfg["max_brightness"]))))

    strictness = cfg.get("texture_strictness", 1.0)
    tex_score = np.clip(1.2 - texture * strictness, 0, 1)

    return hue_score * sat_score * val_score * tex_score


def _vegetation_probability(h: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Compute vegetation probability based on green hue."""
    cfg = PHYSICS["trees"]

    hue_score = np.exp(-((h - cfg["hue_center"]) ** 2) / (2 * cfg["hue_width"] ** 2))
    sat_score = 1 / (1 + np.exp(-5 * (s - cfg["min_saturation"])))

    return hue_score * sat_score * 0.9


def _road_probability(s: np.ndarray, v: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Compute road probability based on low saturation and smoothness."""
    cfg = PHYSICS["roads"]

    sat_score = np.exp(-(s ** 2) / (2 * cfg["max_saturation"] ** 2))
    val_score = np.exp(-((v - cfg["val_center"]) ** 2) / (2 * cfg["val_width"] ** 2))
    tex_score = 1 - texture

    return sat_score * val_score * tex_score * 0.8


def _habitat_probability(h: np.ndarray, s: np.ndarray, v: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Compute habitat probability using building colors and texture filtering."""
    cfg = PHYSICS["human_habitat"]

    # Red/orange brick detection
    red_score = np.maximum(
        np.exp(-((h - cfg["red_brick_center1"]) ** 2) / (2 * cfg["hue_width"] ** 2)),
        np.exp(-((h - cfg["red_brick_center2"]) ** 2) / (2 * cfg["hue_width"] ** 2)),
    )

    # Concrete detection (low saturation, high value)
    concrete_score = (1 - s) * v
    base_score = np.maximum(red_score, concrete_score)

    # Texture filter: high texture = structure, low texture = bare soil
    min_texture = cfg.get("min_texture", 0.35)
    tex_modifier = np.where(texture < min_texture, 0.2, 0.5 + 0.5 * texture)

    return base_score * tex_modifier
