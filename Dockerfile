# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with specific CUDA support (User Request)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
# Use --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create outputs directory
RUN mkdir -p outputs

# Set entrypoint
ENTRYPOINT ["python", "analyze.py"]

# Default command (can be overridden)
CMD ["--help"]
