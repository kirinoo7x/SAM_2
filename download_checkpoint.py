#!/usr/bin/env python3
"""Download SAM2 checkpoint from official sources."""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_sam2_checkpoint(model_size="base_plus"):
    """
    Download SAM2 checkpoint.

    Args:
        model_size: One of ["tiny", "small", "base_plus", "large"]
    """
    # Create checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # SAM2 checkpoint URLs
    checkpoint_urls = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }

    if model_size not in checkpoint_urls:
        raise ValueError(f"Invalid model_size. Choose from {list(checkpoint_urls.keys())}")

    url = checkpoint_urls[model_size]
    checkpoint_name = f"sam2_hiera_{model_size}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    # Check if already downloaded
    if checkpoint_path.exists():
        print(f"Checkpoint already exists at {checkpoint_path}")
        return str(checkpoint_path)

    print(f"Downloading SAM2 {model_size} checkpoint...")
    print(f"URL: {url}")
    print(f"Saving to: {checkpoint_path}")

    try:
        download_url(url, checkpoint_path)
        print(f"\nDownload completed! Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")
        if checkpoint_path.exists():
            checkpoint_path.unlink()  # Remove partial download
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SAM2 checkpoint")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base_plus",
        choices=["tiny", "small", "base_plus", "large"],
        help="Model size to download"
    )

    args = parser.parse_args()

    checkpoint_path = download_sam2_checkpoint(args.model_size)
    print(f"\nCheckpoint ready at: {checkpoint_path}")
