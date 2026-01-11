"""Visualization and utility functions for TerraCover.

This module provides area calculation, visualization generation,
and file I/O utilities for the analysis pipeline.
"""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .config import CONFIG, LANDCOVER, log

# Visualization constants
_FIGURE_SIZE = (26, 15)
_DPI = 150
_OVERLAY_ALPHA = 0.45
_BACKGROUND_COLOR = 200


def save_probability_heatmaps(
    image: np.ndarray,
    deep_probs: np.ndarray,
    color_probs: np.ndarray,
    output_dir: Path,
) -> np.ndarray:
    """Combine and save probability heatmaps for each class.

    Args:
        image: Original RGB image of shape (H, W, 3).
        deep_probs: Deep learning probabilities of shape (C, H, W).
        color_probs: Physics-based probabilities of shape (C, H, W).
        output_dir: Directory for output files.

    Returns:
        Combined and normalized probabilities of shape (C, H, W).
    """
    prob_dir = output_dir / "probabilities"
    prob_dir.mkdir(exist_ok=True)

    # Weighted ensemble
    w_deep = CONFIG["ensemble_weights"]["deep_learning"]
    w_color = CONFIG["ensemble_weights"]["color_physics"]
    combined = w_deep * deep_probs + w_color * color_probs

    # Normalize to sum to 1
    combined = combined / (combined.sum(axis=0, keepdims=True) + 1e-8)

    # Save per-class heatmaps
    for name, info in LANDCOVER.items():
        if name == "Background":
            continue

        prob_map = combined[info["id"]]

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.imshow(prob_map, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.title(f"{name} Confidence", fontsize=16, fontweight="bold")
        plt.colorbar(label="Probability")
        plt.savefig(prob_dir / f"{name}.png", bbox_inches="tight", dpi=100)
        plt.close()

    log(f"Probability maps saved: {prob_dir}")
    return combined


def calculate_areas(segmentation: np.ndarray, meters_per_pixel: float) -> Dict[str, Dict[str, Any]]:
    """Calculate real-world area statistics from segmentation.

    Args:
        segmentation: Class ID mask of shape (H, W).
        meters_per_pixel: Ground sampling distance.

    Returns:
        Dictionary mapping class names to statistics including
        pixel count, percentage, hectares, and display color.
    """
    total_pixels = segmentation.size
    pixel_area_sqm = meters_per_pixel ** 2
    areas = {}

    for name, info in LANDCOVER.items():
        count = int((segmentation == info["id"]).sum())
        if count > 0:
            areas[name] = {
                "pixels": count,
                "percentage": 100.0 * count / total_pixels,
                "area_hectares": count * pixel_area_sqm / 10000.0,
                "color": info["color"],
            }

    return areas


def create_comprehensive_visualization(
    image: np.ndarray,
    segmentation: np.ndarray,
    areas: Dict[str, Dict[str, Any]],
    save_path: Path,
    title: str,
    meters_per_pixel: float,
) -> None:
    """Generate the analysis dashboard visualization.

    Args:
        image: Original RGB image of shape (H, W, 3).
        segmentation: Class ID mask of shape (H, W).
        areas: Area statistics from calculate_areas().
        save_path: Output file path.
        title: Visualization title.
        meters_per_pixel: Ground sampling distance.
    """
    h, w = segmentation.shape

    # Create colored classification map
    colored = np.full((h, w, 3), _BACKGROUND_COLOR, dtype=np.uint8)
    for name, info in LANDCOVER.items():
        colored[segmentation == info["id"]] = info["color"]

    # Create figure
    fig = plt.figure(figsize=_FIGURE_SIZE)

    # Panel 1: Original
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=16, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Classification
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(colored)
    ax2.set_title("Classification", fontsize=16, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Overlay
    ax3 = fig.add_subplot(2, 3, 3)
    overlay = ((1 - _OVERLAY_ALPHA) * image + _OVERLAY_ALPHA * colored).astype(np.uint8)
    ax3.imshow(overlay)
    ax3.set_title("Overlay", fontsize=16, fontweight="bold")
    ax3.axis("off")

    # Panel 4: Methodology
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis("off")
    _render_methodology(ax4, w, h, meters_per_pixel)

    # Panel 5: Distribution chart
    ax5 = fig.add_subplot(2, 3, 5)
    _render_chart(ax5, areas)

    # Panel 6: Highlights
    ax6 = fig.add_subplot(2, 3, 6)
    _render_highlights(ax6, image, segmentation)

    # Title and save
    total_ha = (h * w * meters_per_pixel ** 2) / 10000.0
    plt.suptitle(f"{title}\nTotal Area: {total_ha:,.0f} Hectares", fontsize=22, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close()

    log(f"Visualization saved: {save_path}")


def _render_methodology(ax: plt.Axes, w: int, h: int, mpp: float) -> None:
    """Render methodology panel."""
    pixel_area = mpp ** 2
    total_ha = (h * w * pixel_area) / 10000.0

    text = f"""ANALYSIS PARAMETERS
{'─' * 40}
Resolution: {w} x {h} pixels
Scale: {mpp:.2f} m/pixel

AREA CALCULATION
{'─' * 40}
1 pixel = {pixel_area:.2f} sq.m
Total: {total_ha:,.1f} hectares"""

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f5f5f5", edgecolor="#cccccc"))


def _render_chart(ax: plt.Axes, areas: Dict[str, Dict[str, Any]]) -> None:
    """Render distribution bar chart."""
    if not areas:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    sorted_names = sorted(areas.keys(), key=lambda x: areas[x]["percentage"], reverse=True)
    pcts = [areas[n]["percentage"] for n in sorted_names]
    has = [areas[n]["area_hectares"] for n in sorted_names]
    colors = [tuple(c / 255.0 for c in areas[n]["color"]) for n in sorted_names]

    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, pcts, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title("Distribution", fontsize=16, fontweight="bold")

    for i, (pct, ha) in enumerate(zip(pcts, has)):
        ax.text(pct + 1, i, f"{pct:.1f}% ({ha:,.0f} ha)", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, max(pcts) * 1.35 if pcts else 100)


def _render_highlights(ax: plt.Axes, image: np.ndarray, segmentation: np.ndarray) -> None:
    """Render water and vegetation highlight view."""
    dimmed = (image * 0.4).astype(np.uint8)

    water_mask = segmentation == LANDCOVER["Water"]["id"]
    trees_mask = segmentation == LANDCOVER["Trees_Woodland"]["id"]

    dimmed[water_mask] = [0, 150, 255]
    dimmed[trees_mask] = [50, 205, 50]

    ax.imshow(dimmed)
    ax.set_title("Water & Vegetation", fontsize=16, fontweight="bold")
    ax.axis("off")
