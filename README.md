# SAM2 Training and Testing

This script uses SAM2 (Segment Anything Model 2) for segmentation on your custom dataset.

## Dataset Structure

```
cc/
├── train/
│   ├── images/       # Training images (9,901 samples)
│   ├── masks/        # Training masks
│   └── _classes.csv  # Class definitions (background, object)
└── test/
    ├── images/       # Test images (1,098 samples)
    ├── masks/        # Test masks
    └── _classes.csv
```

## Setup

### Option 1: Automatic Setup (Recommended)

```bash
./setup.sh
```

This will create a virtual environment and install all dependencies.

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv sam2_env
source sam2_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install typing_extensions>=4.8.0
pip install torch torchvision numpy Pillow matplotlib opencv-python

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Download SAM2 Checkpoint

**Important**: The model checkpoint files are large (300MB+) and must be downloaded separately. They are NOT included in this repository.

Download one of the pre-trained SAM2 models from the [SAM2 GitHub releases](https://github.com/facebookresearch/segment-anything-2/releases):

- **sam2_hiera_large.pt** (recommended for best performance)
- **sam2_hiera_base_plus.pt** (balanced, ~309MB)
- **sam2_hiera_small.pt** (faster)

Place the downloaded checkpoint file in this project directory.

### Update the Script

Edit `sam2_train_test.py` and set the checkpoint path (line 242):

```python
CHECKPOINT = 'sam2_hiera_large.pt'  # Path to your downloaded checkpoint
```

## Usage

### Run the Script

```bash
# If using virtual environment
source sam2_env/bin/activate
python sam2_train_test.py

# Or use the run script
./run.sh
```

The script will:
1. Load your train/test datasets
2. Initialize SAM2 model
3. Test on sample images and save visualizations
4. Evaluate on test set and report metrics

### Output

- **result_0.png, result_1.png, etc.**: Visualization of predictions
- **Console output**: IoU metrics on test set

## How It Works

The script uses **point-based prompting** for SAM2:
1. Samples random points from ground truth masks as prompts
2. Feeds points to SAM2 to generate predictions
3. Compares predictions with ground truth masks
4. Reports IoU (Intersection over Union) metrics

## Customization

### Change Number of Point Prompts

```python
point_coords, point_labels = trainer.generate_point_prompts_from_mask(gt_mask, num_points=10)
```

### Change Model Configuration

```python
MODEL_CFG = 'sam2_hiera_b+.yaml'  # Use base+ model instead
```

### Evaluate on More Samples

```python
metrics = trainer.evaluate_dataset(test_dataset, num_samples=500)  # Evaluate 500 samples
```

## Classes

- **SegmentationDataset**: Loads images and masks from folders
- **SAM2Trainer**: Handles SAM2 model initialization, prediction, and evaluation

## Notes

- This is a **binary segmentation** task (background vs object)
- SAM2 works in **promptable mode** (using points, boxes, or masks as prompts)
- For fine-tuning SAM2 on your dataset, you'll need additional training code
