#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class CrackDataset(Dataset):
    """Dataset for crack detection with image and mask pairs."""

    def __init__(self, root_dir: str, split: str = "Train", image_size: int = 1024):
        """
        Args:
            root_dir: Root directory containing Train/Test folders
            split: Either "Train" or "Test"
            image_size: Target image size for SAM2
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        self.images_dir = self.root_dir / split / "images"
        self.masks_dir = self.root_dir / split / "masks"

        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"Loaded {len(self.image_files)} images from {split} set")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - image: (3, H, W) tensor
                - mask: (1, H, W) tensor with binary values
                - original_size: (H, W) tuple
        """
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # Binary threshold

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "original_size": original_size,
            "image_path": str(img_path)
        }


class SAM2LossFunctions:
    """Collection of loss functions for SAM2 training."""

    @staticmethod
    def dice_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred_masks: (B, 1, H, W) predicted masks
            target_masks: (B, 1, H, W) ground truth masks
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice loss value
        """
        pred_masks = torch.sigmoid(pred_masks)

        intersection = (pred_masks * target_masks).sum(dim=(2, 3))
        union = pred_masks.sum(dim=(2, 3)) + target_masks.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    @staticmethod
    def focal_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        Compute Focal loss for handling class imbalance.

        Args:
            pred_masks: (B, 1, H, W) predicted masks (logits)
            target_masks: (B, 1, H, W) ground truth masks
            alpha: Weighting factor
            gamma: Focusing parameter

        Returns:
            Focal loss value
        """
        pred_sigmoid = torch.sigmoid(pred_masks)
        ce_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction='none')

        p_t = pred_sigmoid * target_masks + (1 - pred_sigmoid) * (1 - target_masks)
        alpha_t = alpha * target_masks + (1 - alpha) * (1 - target_masks)

        focal_loss = alpha_t * ((1 - p_t) ** gamma) * ce_loss
        return focal_loss.mean()

    @staticmethod
    def iou_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Compute IoU loss.

        Args:
            pred_masks: (B, 1, H, W) predicted masks
            target_masks: (B, 1, H, W) ground truth masks
            smooth: Smoothing factor

        Returns:
            IoU loss value
        """
        pred_masks = torch.sigmoid(pred_masks)

        intersection = (pred_masks * target_masks).sum(dim=(2, 3))
        union = pred_masks.sum(dim=(2, 3)) + target_masks.sum(dim=(2, 3)) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return 1.0 - iou.mean()

    @staticmethod
    def combined_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor,
                     dice_weight: float = 1.0, focal_weight: float = 20.0,
                     iou_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with individual components.

        Returns:
            Dictionary with 'total', 'dice', 'focal', 'iou' losses
        """
        dice = SAM2LossFunctions.dice_loss(pred_masks, target_masks)
        focal = SAM2LossFunctions.focal_loss(pred_masks, target_masks)
        iou = SAM2LossFunctions.iou_loss(pred_masks, target_masks)

        total_loss = dice_weight * dice + focal_weight * focal + iou_weight * iou

        return {
            'total': total_loss,
            'dice': dice,
            'focal': focal,
            'iou': iou
        }


class SAM2Trainer:
    """Trainer class for SAM2 model."""

    def __init__(self, model_cfg: str, checkpoint_path: str, device: str = "cuda"):
        """
        Args:
            model_cfg: Path to model config file
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to use for training
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Build SAM2 model
        self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        # Loss functions
        self.loss_fn = SAM2LossFunctions()

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for SAM2 model (normalize and process)."""
        # SAM2 expects images in range [0, 1] normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        return (image - mean) / std

    def _encode_prompts(self, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create point prompts for SAM2."""
        # Use center point as positive prompt
        point_coords = torch.zeros(batch_size, 1, 2, device=self.device)
        point_coords[:, 0, 0] = W // 2  # x coordinate
        point_coords[:, 0, 1] = H // 2  # y coordinate

        point_labels = torch.ones(batch_size, 1, device=self.device)  # positive prompt

        return point_coords, point_labels

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   dice_weight: float = 1.0, focal_weight: float = 20.0,
                   iou_weight: float = 1.0) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with average losses
        """
        self.sam2_model.train()

        total_losses = {'total': 0.0, 'dice': 0.0, 'focal': 0.0, 'iou': 0.0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch}")

        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            optimizer.zero_grad()

            # Prepare images
            images_prepared = self._prepare_image(images)

            batch_size, _, H, W = images.shape

            # Encode image features
            with torch.set_grad_enabled(True):
                # Get image embeddings
                backbone_out = self.sam2_model.forward_image(images_prepared)

                # Prepare backbone features (similar to predictor)
                _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)

                # Add no_mem_embed if needed
                if self.sam2_model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed

                # Get backbone feature sizes
                feat_sizes = [(256, 256), (128, 128), (64, 64)]  # For SAM2 with 1024x1024 input

                # Process features similar to predictor
                feats = [
                    feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
                ][::-1]

                image_embed = feats[-1]  # Lowest resolution
                high_res_feats = feats[:-1]  # Higher resolution features

                # Create point prompts
                point_coords, point_labels = self._encode_prompts(batch_size, H, W)

                # Prepare sparse and dense embeddings
                sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )

                # Decode masks
                low_res_masks, iou_predictions, _, _ = self.sam2_model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats,
                )

                # Resize to target resolution
                pred_masks = F.interpolate(
                    low_res_masks,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )

            # Compute loss
            losses = self.loss_fn.combined_loss(
                pred_masks, masks,
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                iou_weight=iou_weight
            )

            # Backward pass
            losses['total'].backward()
            optimizer.step()

            # Accumulate losses
            for key in total_losses.keys():
                total_losses[key] += losses[key].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'dice': f"{losses['dice'].item():.4f}",
                'focal': f"{losses['focal'].item():.4f}",
                'iou': f"{losses['iou'].item():.4f}"
            })

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.current_epoch += 1

        return avg_losses

    @torch.no_grad()
    def test(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Test the model.

        Returns:
            Dictionary with test metrics
        """
        self.sam2_model.eval()

        total_losses = {'total': 0.0, 'dice': 0.0, 'focal': 0.0, 'iou': 0.0}
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Testing")

        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Prepare images
            images_prepared = self._prepare_image(images)

            batch_size, _, H, W = images.shape

            # Get image embeddings
            backbone_out = self.sam2_model.forward_image(images_prepared)

            # Prepare backbone features
            _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)

            if self.sam2_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed

            feat_sizes = [(256, 256), (128, 128), (64, 64)]

            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
            ][::-1]

            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            # Create point prompts
            point_coords, point_labels = self._encode_prompts(batch_size, H, W)

            # Prepare sparse and dense embeddings
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )

            # Decode masks
            low_res_masks, iou_predictions, _, _ = self.sam2_model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )

            # Resize to target resolution
            pred_masks = F.interpolate(
                low_res_masks,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

            # Compute losses
            losses = self.loss_fn.combined_loss(pred_masks, masks)

            for key in total_losses.keys():
                total_losses[key] += losses[key].item()

            # Compute metrics
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            intersection = (pred_binary * masks).sum(dim=(2, 3))
            union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection

            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2.0 * intersection + 1e-6) / (pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + 1e-6)

            total_iou += iou.mean().item()
            total_dice += dice.mean().item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'IoU': f"{iou.mean().item():.4f}",
                'Dice': f"{dice.mean().item():.4f}"
            })

        # Average metrics
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses['avg_iou'] = total_iou / num_batches
        avg_losses['avg_dice'] = total_dice / num_batches

        return avg_losses

    def save_checkpoint(self, save_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.sam2_model.state_dict(),
            'best_loss': self.best_loss,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.sam2_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")


def main():
    """Main training and testing pipeline."""

    # Configuration
    DATA_ROOT = "cc"
    MODEL_CFG = "sam2_hiera_b+"  # Config name without .yaml extension
    CHECKPOINT = "checkpoints/sam2_hiera_base_plus.pt"
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-5
    IMAGE_SIZE = 1024

    # Auto-download checkpoint if it doesn't exist
    checkpoint_path = Path(CHECKPOINT)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {CHECKPOINT}")
        print("Please download the checkpoint first by running:")
        print("  python download_checkpoint.py --model-size base_plus")
        print("\nOr download manually from:")
        print("  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt")
        print(f"  Save it to: {checkpoint_path.absolute()}")
        return

    # Loss weights
    DICE_WEIGHT = 1.0
    FOCAL_WEIGHT = 20.0
    IOU_WEIGHT = 1.0

    # Create datasets
    train_dataset = CrackDataset(DATA_ROOT, split="Train", image_size=IMAGE_SIZE)
    test_dataset = CrackDataset(DATA_ROOT, split="Test", image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize trainer
    trainer = SAM2Trainer(MODEL_CFG, CHECKPOINT, device="cuda")

    # Setup optimizer
    optimizer = torch.optim.AdamW(trainer.sam2_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_losses = trainer.train_epoch(
            train_loader, optimizer,
            dice_weight=DICE_WEIGHT,
            focal_weight=FOCAL_WEIGHT,
            iou_weight=IOU_WEIGHT
        )

        print(f"\nTrain Losses - Total: {train_losses['total']:.4f}, "
              f"Dice: {train_losses['dice']:.4f}, "
              f"Focal: {train_losses['focal']:.4f}, "
              f"IoU: {train_losses['iou']:.4f}")

        # Test
        test_metrics = trainer.test(test_loader)

        print(f"\nTest Metrics - Total Loss: {test_metrics['total']:.4f}, "
              f"Avg IoU: {test_metrics['avg_iou']:.4f}, "
              f"Avg Dice: {test_metrics['avg_dice']:.4f}")

        # Save best model
        if test_metrics['total'] < trainer.best_loss:
            trainer.best_loss = test_metrics['total']
            trainer.save_checkpoint("checkpoints/best_model.pt", optimizer)
            print("Saved new best model!")

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch+1}.pt", optimizer)

        # Update learning rate
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    print("\n" + "="*50)
    print("Training Completed!")
    print("="*50)


if __name__ == "__main__":
    main()
