"""Satellite imagery acquisition module for TerraCover.

This module handles fetching satellite imagery from Esri World Imagery
(free) and Google Static Maps API providers.
"""

import io
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image

from .config import CONFIG, log

# Tile server constants
_ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
_GOOGLE_API_URL = "https://maps.googleapis.com/maps/api/staticmap"
_TILE_SIZE = 256
_MAX_ZOOM = 19
_MAX_OUTPUT_SIZE = 2000
_MAX_TILES = 225
_USER_AGENT = "TerraCover/1.0"

# Earth geometry constant
_METERS_PER_PIXEL_AT_EQUATOR = 156543.03


def fetch_satellite_image(
    lat: float,
    lon: float,
    radius: float,
    api_key: Optional[str] = None,
    forced_zoom: Optional[int] = None,
) -> Tuple[str, int, float]:
    """Fetch satellite imagery for a geographic location.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius: Capture radius in meters.
        api_key: Optional Google Maps API key.
        forced_zoom: Optional zoom level override.

    Returns:
        Tuple of (filepath, zoom_level, actual_width_meters).
    """
    provider = CONFIG.get("api", {}).get("provider", "esri")

    if provider == "google":
        return _fetch_google(lat, lon, radius, api_key, forced_zoom)
    return _fetch_esri(lat, lon, radius, forced_zoom)


def _meters_per_pixel(lat: float, zoom: int) -> float:
    """Calculate ground sampling distance for a latitude and zoom level."""
    return (_METERS_PER_PIXEL_AT_EQUATOR * math.cos(math.radians(lat))) / (2 ** zoom)


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert geographic coordinates to XYZ tile indices."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _fetch_esri(
    lat: float,
    lon: float,
    radius: float,
    forced_zoom: Optional[int] = None,
) -> Tuple[str, int, float]:
    """Fetch imagery from Esri World Imagery tile server."""
    log("Fetching imagery from Esri World Imagery")
    log(f"  Location: {lat:.6f}, {lon:.6f}")
    log(f"  Radius: {radius}m")

    # Determine zoom level
    if forced_zoom:
        zoom = min(forced_zoom, _MAX_ZOOM)
        log(f"  Zoom: {zoom} (forced)")
    else:
        target_mpp = 1.0
        zoom = int(round(math.log2(_METERS_PER_PIXEL_AT_EQUATOR * math.cos(math.radians(lat)) / target_mpp)))
        zoom = min(zoom, _MAX_ZOOM)
        log(f"  Zoom: {zoom} (auto)")

    mpp = _meters_per_pixel(lat, zoom)
    pixels_needed = int((radius * 2) / mpp)

    # Limit output size
    if pixels_needed > _MAX_OUTPUT_SIZE:
        log(f"  Clamping size to {_MAX_OUTPUT_SIZE}px")
        pixels_needed = _MAX_OUTPUT_SIZE
        target_mpp = (radius * 2) / _MAX_OUTPUT_SIZE
        zoom = int(round(math.log2(_METERS_PER_PIXEL_AT_EQUATOR * math.cos(math.radians(lat)) / target_mpp)))
        zoom = min(zoom, _MAX_ZOOM)
        mpp = _meters_per_pixel(lat, zoom)

    # Calculate tile grid
    center_x, center_y = _latlon_to_tile(lat, lon, zoom)
    tiles_span = math.ceil(pixels_needed / _TILE_SIZE)
    if tiles_span % 2 == 0:
        tiles_span += 1
    if tiles_span ** 2 > _MAX_TILES:
        tiles_span = int(math.sqrt(_MAX_TILES))
        log(f"  Clamping grid to {tiles_span}x{tiles_span}", level="WARNING")

    offset = tiles_span // 2
    canvas_size = tiles_span * _TILE_SIZE
    canvas = Image.new("RGB", (canvas_size, canvas_size))

    log(f"  Downloading {tiles_span}x{tiles_span} tiles ({tiles_span ** 2} total)")

    # Fetch tiles
    headers = {"User-Agent": _USER_AGENT}
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            url = _ESRI_TILE_URL.format(z=zoom, y=center_y + dy, x=center_x + dx)
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content)).convert("RGB")
                    canvas.paste(tile, ((dx + offset) * _TILE_SIZE, (dy + offset) * _TILE_SIZE))
            except requests.RequestException as e:
                log(f"  Tile error ({center_x + dx}, {center_y + dy}): {e}", level="WARNING")

    # Crop to target size
    final_size = min(pixels_needed, canvas_size)
    margin = (canvas_size - final_size) // 2
    result = canvas.crop((margin, margin, margin + final_size, margin + final_size))

    # Save
    output_dir = Path("data/downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"esri_{lat}_{lon}_z{zoom}.png"
    result.save(filepath)

    actual_width = mpp * final_size
    log(f"  Saved: {filepath}")
    log(f"  Resolution: {mpp:.2f} m/px, Width: {actual_width:.0f}m")

    return str(filepath), zoom, actual_width


def _fetch_google(
    lat: float,
    lon: float,
    radius: float,
    api_key: Optional[str],
    forced_zoom: Optional[int] = None,
) -> Tuple[str, int, float]:
    """Fetch imagery from Google Static Maps API."""
    if not api_key or "YOUR_API_KEY" in api_key:
        log("Google API key not configured", level="ERROR")
        sys.exit(1)

    log("Fetching imagery from Google Static Maps")
    log(f"  Location: {lat:.6f}, {lon:.6f}")

    img_size = CONFIG.get("api", {}).get("image_size", 640)

    if forced_zoom:
        zoom = forced_zoom
    else:
        target_mpp = (radius * 2) / img_size
        zoom = int(round(math.log2(_METERS_PER_PIXEL_AT_EQUATOR * math.cos(math.radians(lat)) / target_mpp)))

    mpp = _meters_per_pixel(lat, zoom)
    actual_width = mpp * img_size

    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{img_size}x{img_size}",
        "maptype": "satellite",
        "key": api_key,
        "scale": 1,
    }

    response = requests.get(_GOOGLE_API_URL, params=params, timeout=30)
    if response.status_code != 200:
        log(f"Google API error: {response.text}", level="ERROR")
        sys.exit(1)

    output_dir = Path("data/downloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"google_{lat}_{lon}.png"

    with open(filepath, "wb") as f:
        f.write(response.content)

    log(f"  Saved: {filepath}")
    return str(filepath), zoom, actual_width
