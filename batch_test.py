import os
import shutil
import glob
import subprocess
from pathlib import Path

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. Clean Outputs
    if Path("outputs").exists():
        print("Cleaning outputs directory...")
        shutil.rmtree("outputs")
    Path("outputs").mkdir()

    # 2. Run on Local Files
    print("--- Processing Local Files ---")
    local_files = glob.glob("data/*.png") + glob.glob("data/*.jpg")
    for f in local_files:
        if "google_" in f or "esri_" in f: continue # Skip cached downloads if present in root
        run_command(f"python analyze.py \"{f}\"")

    # 3. Run on Diverse Lat/Lon Locations
    print("\n--- Fetching Diverse Terrains ---")
    locations = [
        ("Cairo Edge (Habitat vs Desert)", 30.0444, 31.2357),
        ("Grand Canyon Village (Habitat vs Rock)", 36.0544, -112.1401),
        ("Kyoto Edge (Habitat vs Forest)", 35.0116, 135.7681),
        ("Santorini (White Habitat vs Water)", 36.4167, 25.4333),
        ("Mumbai Coast (Dense Habitat vs Water)", 19.0760, 72.8777)
    ]

    for name, lat, lon in locations:
        print(f"\nProcessing {name} ({lat}, {lon})...")
        # Use radius 1000m to get a decent area
        run_command(f"python analyze.py --lat {lat} --lon {lon} --radius 1000")

    print("\nBatch Test Complete. Check outputs/ folder.")

if __name__ == "__main__":
    main()
