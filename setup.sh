#!/bin/bash

# Setup script for SAM2 project
echo "Setting up SAM2 environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv sam2_env

# Activate virtual environment
echo "Activating virtual environment..."
source sam2_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install typing_extensions>=4.8.0
pip install torch torchvision
pip install numpy Pillow matplotlib opencv-python tqdm

# Install SAM2
echo "Installing SAM2..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git

echo ""
echo "Setup complete!"
echo ""
echo "IMPORTANT: You need to download a SAM2 checkpoint file:"
echo "  1. Visit: https://github.com/facebookresearch/segment-anything-2/releases"
echo "  2. Download one of: sam2_hiera_large.pt, sam2_hiera_base_plus.pt, or sam2_hiera_small.pt"
echo "  3. Place the .pt file in this directory"
echo ""
echo "To activate the environment, run:"
echo "  source sam2_env/bin/activate"
echo ""
echo "Then run the script:"
echo "  python sam2_train_test.py"
echo ""
