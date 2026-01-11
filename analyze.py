"""TerraCover: Satellite imagery land cover analysis tool.

This is the main entry point for the analysis pipeline. It supports
both local file analysis and coordinate-based imagery fetching.

Usage:
    python analyze.py "path/to/image.png"
    python analyze.py --lat 40.78 --lon -73.96 --radius 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import CONFIG, LANDCOVER, log
from src.core import check_device, SatelliteLandCoverModel
from src.physics import get_color_probabilities
from src.utils import (
    calculate_areas,
    create_comprehensive_visualization,
    save_probability_heatmaps,
)
from src.fetcher import fetch_satellite_image

# Analysis constants
_DEFAULT_LATITUDE_FOR_SCALE = 40.7  # NYC latitude for default scale estimation
_SHADOW_CONFIDENCE_THRESHOLD = 0.35


def analyze_image(
    image_path: str,
    zoom_level: Optional[int] = None,
    real_width_meters: Optional[float] = None,
    rotations: int = 1,
) -> None:
    """Execute the complete land cover analysis pipeline.

    Args:
        image_path: Path to the input satellite image.
        zoom_level: Optional zoom level for scale estimation.
        real_width_meters: Optional known ground width in meters.
        rotations: Number of rotation angles for ensemble analysis.
    """
    device = check_device()
    path = Path(image_path)

    if not path.exists():
        log(f"Image not found: {path}", level="ERROR")
        sys.exit(1)

    # Setup output directory
    output_dir = Path("outputs") / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    log(f"Loading image: {path}")
    image = np.array(Image.open(path).convert("RGB"))
    h, w = image.shape[:2]
    log(f"Image size: {w} x {h} pixels")

    # Determine scale
    scale_mpp = _determine_scale(w, zoom_level, real_width_meters)

    # Save original
    Image.fromarray(image).save(output_dir / "original.png")

    # Run analysis
    if rotations > 1:
        final_probs = _rotational_ensemble(image, rotations, output_dir, device)
    else:
        final_probs = _standard_analysis(image, output_dir, device)

    # Generate segmentation
    log("Generating segmentation")
    segmentation = final_probs.argmax(axis=0).astype(np.uint8)

    # Apply post-processing
    _apply_confidence_filter(segmentation, final_probs)
    _apply_shadow_filter(segmentation, final_probs)

    # Calculate areas
    log("Computing area statistics")
    areas = calculate_areas(segmentation, scale_mpp)

    for name, stats in areas.items():
        log(f"  {name}: {stats['percentage']:.1f}% ({stats['area_hectares']:.1f} ha)")

    # Generate visualization
    log("Creating visualization")
    create_comprehensive_visualization(
        image, segmentation, areas,
        output_dir / "landcover_analysis.png",
        f"Analysis: {path.stem}",
        scale_mpp,
    )

    log(f"Analysis complete: {output_dir}")


def _determine_scale(width: int, zoom: Optional[int], real_width: Optional[float]) -> float:
    """Determine the ground sampling distance."""
    if real_width:
        mpp = real_width / width
        log(f"Scale from width: {mpp:.2f} m/px")
    elif zoom:
        mpp = (156543.03 * 0.758) / (2 ** zoom)
        log(f"Scale from zoom {zoom}: {mpp:.2f} m/px")
    else:
        mpp = CONFIG["scale"].get("default_mpp", 2.39)
        log(f"Using default scale: {mpp:.2f} m/px")
    return mpp


def _standard_analysis(image: np.ndarray, output_dir: Path, device) -> np.ndarray:
    """Run standard single-pass analysis."""
    model = SatelliteLandCoverModel(device)
    model.load_model()

    log("Running deep learning inference")
    deep_probs = model.predict_probabilities(image)

    log("Running physics analysis")
    color_probs = get_color_probabilities(image)

    log("Combining probability maps")
    return save_probability_heatmaps(image, deep_probs, color_probs, output_dir)


def _rotational_ensemble(
    image: np.ndarray,
    num_rotations: int,
    output_dir: Path,
    device,
) -> np.ndarray:
    """Run rotational ensemble analysis."""
    log(f"Starting rotational ensemble ({num_rotations} angles)")

    h, w = image.shape[:2]
    num_classes = len(LANDCOVER)

    accum_probs = np.zeros((num_classes, h, w), dtype=np.float32)
    accum_counts = np.zeros((h, w), dtype=np.float32) + 1e-8

    rot_dir = output_dir / "rotations"
    rot_dir.mkdir(exist_ok=True)

    model = SatelliteLandCoverModel(device)
    model.load_model()

    base_pil = Image.fromarray(image)
    w_deep = CONFIG["ensemble_weights"]["deep_learning"]
    w_color = CONFIG["ensemble_weights"]["color_physics"]

    for i in range(num_rotations):
        angle = i * (360.0 / num_rotations)
        log(f"  Processing angle: {angle:.0f}°")

        rot_pil = base_pil.rotate(angle, resample=Image.BICUBIC, expand=False)
        rot_pil.save(rot_dir / f"view_{i:02d}_{int(angle)}deg.png")
        rot_img = np.array(rot_pil)

        deep_probs = model.predict_probabilities(rot_img)
        color_probs = get_color_probabilities(rot_img)
        view_probs = deep_probs * w_deep + color_probs * w_color

        # Rotate probabilities back
        back_probs = np.zeros_like(view_probs)
        for c in range(num_classes):
            c_pil = Image.fromarray(view_probs[c])
            back_probs[c] = np.array(c_pil.rotate(-angle, resample=Image.BICUBIC, expand=False))

        # Create and rotate validity mask
        mask = np.array(Image.new("L", (w, h), 255).rotate(angle, resample=Image.NEAREST, expand=False)) > 128
        mask_back = np.array(Image.fromarray((mask * 255).astype(np.uint8)).rotate(-angle, resample=Image.NEAREST, expand=False)) > 128

        for c in range(num_classes):
            accum_probs[c] += back_probs[c] * mask_back
        accum_counts += mask_back

    log("Rotational ensemble complete")
    return accum_probs / accum_counts


def _apply_confidence_filter(segmentation: np.ndarray, probs: np.ndarray) -> None:
    """Reclassify low-confidence pixels as background."""
    threshold = CONFIG["post_processing"].get("global_unknown_threshold", 0.25)
    low_conf_mask = probs.max(axis=0) < threshold

    if low_conf_mask.sum() > 0:
        segmentation[low_conf_mask] = LANDCOVER["Background"]["id"]
        log(f"  Reclassified {low_conf_mask.sum():,} low-confidence pixels")


def _apply_shadow_filter(segmentation: np.ndarray, probs: np.ndarray) -> None:
    """Remove small low-confidence water regions (likely shadows)."""
    water_id = LANDCOVER["Water"]["id"]
    water_mask = (segmentation == water_id).astype(np.uint8)

    if water_mask.sum() == 0:
        return

    min_blob_size = CONFIG["post_processing"].get("min_water_blob_size", 500)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water_mask, connectivity=8)
    water_probs = probs[water_id]

    removed = 0
    for i in range(1, num_labels):
        size = stats[i, cv2.CC_STAT_AREA]
        if size >= min_blob_size:
            continue

        blob_mask = labels == i
        if water_probs[blob_mask].mean() < _SHADOW_CONFIDENCE_THRESHOLD:
            segmentation[blob_mask] = LANDCOVER["Background"]["id"]
            removed += size

    if removed > 0:
        log(f"  Removed {removed:,} shadow pixels")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TerraCover: Satellite Imagery Land Cover Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py "data/image.png"
  python analyze.py --lat 40.78 --lon -73.96 --radius 1000
  python analyze.py --lat 40.78 --lon -73.96 --zoom 18 --rotations 12
        """,
    )

    parser.add_argument("image_path", nargs="?", help="Path to satellite image")
    parser.add_argument("--lat", type=float, help="Latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, help="Longitude (decimal degrees)")
    parser.add_argument("--radius", type=float, default=500, help="Radius in meters (default: 500)")
    parser.add_argument("--width", type=float, help="Known ground width in meters")
    parser.add_argument("--zoom", type=int, help="Force zoom level")
    parser.add_argument("--rotations", type=int, default=1, help="Rotation angles for ensemble")

    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        api_key = CONFIG.get("api", {}).get("google_maps_key")
        path, zoom, width = fetch_satellite_image(args.lat, args.lon, args.radius, api_key, args.zoom)
        analyze_image(path, real_width_meters=width, rotations=args.rotations)

    elif args.image_path:
        default_zoom = CONFIG["scale"].get("default_zoom_level", 16)
        analyze_image(
            args.image_path,
            zoom_level=args.zoom or default_zoom,
            real_width_meters=args.width,
            rotations=args.rotations,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
