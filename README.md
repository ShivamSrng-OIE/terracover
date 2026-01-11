# TerraCover: Satellite Land Cover Analysis

A professional tool for semantic segmentation and area measurement of satellite imagery. Calculates exact area (in Hectares) of **Water, Trees, Human Habitat, Roads,** and **Grass** using **Deep Learning (U-Net)** combined with **Physics-Based Color and Texture Analysis**.

![Sample Analysis Result](assets/sample_result.png)

---

## Features

- **Hybrid Analysis**: U-Net neural network + HSV color/texture physics
- **Texture Filtering**: Distinguishes buildings from bare soil
- **Precision Measurement**: Converts pixels to real-world area in Hectares
- **Shadow Removal**: Geometric filtering prevents shadow misclassification
- **Free Imagery**: Downloads satellite images from Esri (no API key required)
- **Docker Support**: Containerized deployment with GPU acceleration

---

## Quick Start

### Installation

```bash
cd terracover
python install.py
```

### Activate Environment

```bash
# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Run Analysis

```bash
# By coordinates
python analyze.py --lat 40.78 --lon -73.96 --radius 1000

# Local file
python analyze.py "data/image.png"
```

---

## Command Reference

```
usage: analyze.py [-h] [--lat LAT] [--lon LON] [--radius RADIUS]
                  [--width WIDTH] [--zoom ZOOM] [--rotations ROTATIONS]
                  [image_path]

TerraCover: Satellite Imagery Land Cover Analysis

positional arguments:
  image_path            Path to satellite image file

options:
  -h, --help            Show help message
  --lat LAT             Latitude (decimal degrees)
  --lon LON             Longitude (decimal degrees)
  --radius RADIUS       Capture radius in meters (default: 500)
  --width WIDTH         Known ground width in meters
  --zoom ZOOM           Force resolution level (see below)
  --rotations ROTATIONS Number of rotation angles (default: 1)
```

### Zoom Levels and Resolution

The `--zoom` parameter controls image resolution (detail level):

| Zoom | Resolution | Best For |
|------|------------|----------|
| 15 | ~5 m/pixel | City overview, large areas |
| 16 | ~2.5 m/pixel | Neighborhoods (default) |
| 17 | ~1.2 m/pixel | Individual buildings visible |
| 18 | ~0.6 m/pixel | High detail analysis |
| 19 | ~0.3 m/pixel | Maximum detail (small area) |

**Higher zoom = more detail but smaller coverage area.**

### Examples

```bash
# Standard analysis (auto zoom)
python analyze.py --lat 40.78 --lon -73.96 --radius 1000

# High-resolution (zoom 18 = 0.6 m/pixel detail)
python analyze.py --lat 40.78 --lon -73.96 --radius 500 --zoom 18

# Maximum accuracy with 360-degree ensemble
python analyze.py --lat 40.78 --lon -73.96 --rotations 12
```

---

## Docker

```bash
# Build
docker build -t terracover .

# Run (CPU)
docker run -v $(pwd)/data:/app/data terracover "data/image.png"

# Run (GPU)
docker run --gpus all -v $(pwd)/data:/app/data terracover --lat 40.78 --lon -73.96
```

---

## Configuration

Edit `config.yaml` to tune parameters:

| Section | Description |
|---------|-------------|
| `ensemble_weights` | Balance deep learning vs physics (default: 60/40) |
| `physics` | Color thresholds per class |
| `post_processing` | Shadow removal, confidence thresholds |

---

## Project Structure

```
terracover/
├── analyze.py          # Main entry point
├── install.py          # Automated installer
├── config.yaml         # Configuration
├── requirements.txt    # Dependencies
├── Dockerfile          # Container definition
├── src/
│   ├── config.py       # Logging and configuration
│   ├── core.py         # Deep learning model
│   ├── physics.py      # Color/texture analysis
│   ├── utils.py        # Visualization
│   └── fetcher.py      # Imagery acquisition
└── data/               # Input images
```

---

## License

This project is not for sale or distribution.

Developed by Shivam Manish Sarang
