#!/usr/bin/env python3
"""
Generate accuracy report for the best SAM2 model.

This script loads the best model checkpoint and evaluates it on the test set
to generate a comprehensive accuracy report.
"""

import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from datetime import datetime

from sam2_train_test import SAM2Trainer, CrackDataset


def format_report(metrics, model_info):
    """Format the evaluation metrics into a readable report."""
    report = []
    report.append("=" * 70)
    report.append("SAM2 CRACK DETECTION - BEST MODEL ACCURACY REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("MODEL INFORMATION")
    report.append("-" * 70)
    report.append(f"Model Architecture:    {model_info['architecture']}")
    report.append(f"Best Checkpoint Epoch: {model_info['epoch']}")
    report.append(f"Dataset:               {model_info['dataset']}")
    report.append(f"Test Samples:          {model_info['test_samples']}")
    report.append(f"Image Size:            {model_info['image_size']}x{model_info['image_size']}")
    report.append(f"Batch Size:            {model_info['batch_size']}")
    report.append("")

    report.append("SEGMENTATION ACCURACY METRICS")
    report.append("-" * 70)
    report.append(f"Average IoU Score:     {metrics['avg_iou']:.6f} ({metrics['avg_iou']*100:.2f}%)")
    report.append(f"Average Dice Score:    {metrics['avg_dice']:.6f} ({metrics['avg_dice']*100:.2f}%)")
    report.append("")

    report.append("LOSS COMPONENTS")
    report.append("-" * 70)
    report.append(f"Total Loss:            {metrics['total']:.6f}")
    report.append(f"Dice Loss:             {metrics['dice']:.6f}")
    report.append(f"Focal Loss:            {metrics['focal']:.6f}")
    report.append(f"IoU Loss:              {metrics['iou']:.6f}")
    report.append("")

    report.append("PERFORMANCE INTERPRETATION")
    report.append("-" * 70)

    # IoU interpretation
    iou_pct = metrics['avg_iou'] * 100
    if iou_pct >= 70:
        iou_quality = "Excellent"
    elif iou_pct >= 60:
        iou_quality = "Good"
    elif iou_pct >= 50:
        iou_quality = "Moderate"
    else:
        iou_quality = "Needs Improvement"

    # Dice interpretation
    dice_pct = metrics['avg_dice'] * 100
    if dice_pct >= 80:
        dice_quality = "Excellent"
    elif dice_pct >= 70:
        dice_quality = "Good"
    elif dice_pct >= 60:
        dice_quality = "Moderate"
    else:
        dice_quality = "Needs Improvement"

    report.append(f"IoU Score:             {iou_quality} ({iou_pct:.2f}%)")
    report.append(f"Dice Score:            {dice_quality} ({dice_pct:.2f}%)")
    report.append("")

    report.append("NOTES")
    report.append("-" * 70)
    report.append("- IoU (Intersection over Union): Measures overlap between predicted")
    report.append("  and ground truth segmentation masks. Higher is better.")
    report.append("- Dice Score: Measures similarity between predicted and ground truth.")
    report.append("  Values closer to 1.0 indicate better segmentation accuracy.")
    report.append("- This model was trained on 5% of the full training dataset.")
    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    """Main function to generate the accuracy report."""

    # Configuration
    DATA_ROOT = "cc"
    MODEL_CFG = "sam2_hiera_b+"
    CHECKPOINT = "checkpoints/sam2_hiera_base_plus.pt"
    BEST_MODEL_PATH = "checkpoints/best_model.pt"
    BATCH_SIZE = 2
    IMAGE_SIZE = 1024
    DATA_FRACTION = 0.05

    print("Initializing SAM2 Model for Evaluation...")
    print("-" * 70)

    # Check if best model exists
    if not Path(BEST_MODEL_PATH).exists():
        print(f"Error: Best model checkpoint not found at {BEST_MODEL_PATH}")
        print("Please train the model first using sam2_train_test.py")
        sys.exit(1)

    # Create test dataset
    print("Loading test dataset...")
    test_dataset_full = CrackDataset(DATA_ROOT, split="Test", image_size=IMAGE_SIZE)

    # Use same fraction as training
    test_size = int(len(test_dataset_full) * DATA_FRACTION)
    test_indices = list(range(test_size))
    test_dataset = Subset(test_dataset_full, test_indices)

    print(f"Test dataset: {len(test_dataset)} images ({DATA_FRACTION*100}% of full dataset)")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Initialize trainer
    print("\nInitializing SAM2 trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SAM2Trainer(MODEL_CFG, CHECKPOINT, device=device)

    # Load best model checkpoint
    print(f"Loading best model checkpoint from {BEST_MODEL_PATH}...")
    trainer.load_checkpoint(BEST_MODEL_PATH)

    # Evaluate on test set
    print("\nEvaluating model on test set...")
    print("This may take a few minutes...\n")
    test_metrics = trainer.test(test_loader)

    # Prepare model info
    model_info = {
        'architecture': 'SAM2 Hiera Base+',
        'epoch': trainer.current_epoch,
        'dataset': 'Crack Detection Dataset',
        'test_samples': len(test_dataset),
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE
    }

    # Generate report
    report = format_report(test_metrics, model_info)

    # Display report
    print("\n" + report)

    # Save report to file
    report_file = "model_accuracy_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    # Also save as JSON for programmatic access
    import json
    json_file = "model_accuracy_report.json"
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'model_info': model_info,
        'metrics': {
            'avg_iou': float(test_metrics['avg_iou']),
            'avg_dice': float(test_metrics['avg_dice']),
            'total_loss': float(test_metrics['total']),
            'dice_loss': float(test_metrics['dice']),
            'focal_loss': float(test_metrics['focal']),
            'iou_loss': float(test_metrics['iou'])
        }
    }

    with open(json_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"Report also saved as JSON: {json_file}")


if __name__ == "__main__":
    main()
