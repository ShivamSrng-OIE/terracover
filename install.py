"""
TerraCover Installation Script.

This script automates the setup process by creating a virtual environment
and installing all required dependencies including GPU-accelerated PyTorch.

Usage:
    python install.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def print_header(msg: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


def print_status(msg: str) -> None:
    """Print a status message."""
    print(f"[SETUP] {msg}", flush=True)


def run_command(cmd: list, description: str) -> None:
    """
    Execute a subprocess command with real-time output.

    Args:
        cmd: Command and arguments as a list.
        description: Human-readable description of the operation.

    Raises:
        SystemExit: If the command fails.
    """
    print_status(description)
    print("-" * 40)
    
    try:
        # Show output in real-time (no PIPE)
        result = subprocess.run(cmd, check=True)
        print("-" * 40)
        print_status(f"{description} - COMPLETE\n")
    except subprocess.CalledProcessError as e:
        print_status(f"ERROR: {description} failed with code {e.returncode}")
        sys.exit(1)


def main() -> None:
    """Main installation routine."""
    print_header("TerraCover Installation")
    
    print_status(f"Python Version: {sys.version.split()[0]}")
    print_status(f"Platform: {platform.system()} {platform.release()}")
    
    if sys.version_info < (3, 8):
        print_status("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    
    venv_dir = Path("venv")
    
    if not venv_dir.exists():
        print_status("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print_status("Virtual environment created")
    else:
        print_status("Virtual environment already exists - reusing")
    
    if platform.system() == "Windows":
        pip_exe = venv_dir / "Scripts" / "pip.exe"
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python"
    
    if not pip_exe.exists():
        print_status(f"ERROR: pip not found at {pip_exe}")
        sys.exit(1)
    
    print("")
    run_command(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        "Step 1/3: Upgrading pip"
    )
    
    run_command(
        [str(pip_exe), "install", "torch", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        "Step 2/3: Installing PyTorch (this may take several minutes)"
    )
    
    run_command(
        [str(pip_exe), "install", "-r", "requirements.txt"],
        "Step 3/3: Installing remaining dependencies"
    )
    
    print_header("Installation Complete")
    
    if platform.system() == "Windows":
        activate_cmd = ".\\venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"""
To get started:

  1. Activate the virtual environment:
     {activate_cmd}

  2. Run an analysis:
     python analyze.py "data/your_image.png"

  3. Or fetch imagery by coordinates:
     python analyze.py --lat 40.78 --lon -73.96 --radius 1000

For more options:
     python analyze.py --help
""")


if __name__ == "__main__":
    main()
