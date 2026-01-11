# TerraCover - Quick Start Guide

## What It Does
Analyzes satellite images to measure land coverage: **Water** (blue), **Trees** (green), **Buildings** (red), **Roads** (grey). Calculates exact area in Hectares.

---

## Installation

**Prerequisite:** Python 3.8+ from https://python.org (check "Add to PATH")

```
cd path/to/terracover
python install.py
```
Wait for "Installation Complete" (5-10 min first time).

---

## Usage

**Step 1: Activate** (every time)
```
Windows:   .\venv\Scripts\activate
Mac/Linux: source venv/bin/activate
```

**Step 2: Run**
```
python analyze.py --lat 40.78 --lon -73.96 --radius 1000
```

---

## Command Reference

| Option | Description |
|--------|-------------|
| `image_path` | Path to local image file |
| `--lat` | Latitude (decimal degrees) |
| `--lon` | Longitude (decimal degrees) |
| `--radius` | Area size in meters (default: 500) |
| `--zoom` | Force resolution level (see below) |
| `--rotations` | Multi-angle analysis (default: 1) |

**Zoom Levels Explained:**
- Zoom 15: ~5 m/pixel (city overview)
- Zoom 16: ~2.5 m/pixel (neighborhood)
- Zoom 17: ~1.2 m/pixel (buildings visible)
- Zoom 18: ~0.6 m/pixel (high detail)
- Zoom 19: ~0.3 m/pixel (maximum detail)

Higher zoom = more detail but smaller area covered.

---

## Examples

```bash
# Analyze by coordinates (easiest)
python analyze.py --lat 40.78 --lon -73.96 --radius 1000

# Analyze local file
python analyze.py "data/image.png"

# High-resolution analysis (zoom 18 = 0.6 m/pixel)
python analyze.py --lat 40.78 --lon -73.96 --zoom 18

# Maximum accuracy (12 rotation angles)
python analyze.py --lat 40.78 --lon -73.96 --rotations 12
```

---

## Results

Open `outputs/[name]/landcover_analysis.png` for the visual report with area breakdown.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "python not recognized" | Reinstall Python, check "Add to PATH" |
| "No module named..." | Run `python install.py` again |
| Buildings look like land | Add `--zoom 18` for more detail |
