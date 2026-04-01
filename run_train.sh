#!/bin/bash

# Activate virtual environment and run training script
cd "$(dirname "$0")"

echo "Activating virtual environment..."
source sam2_env/bin/activate

echo "Starting SAM2 training..."
python sam2_train_test.py

deactivate
