# src/data/loader.py

import torch
from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset
from src.data.transform import train_transform, test_transform

# Paths to your processed datasets
TRAIN_DIR = "datas/processed/train"
TEST_DIR = "datas/processed/test"

def get_train_loader(batch_size=4, num_workers=0, frames_per_video=8):
    """
    Returns a DataLoader for the training set.
    Includes dry-run checks to ensure batches are valid.
    """
    dataset = VideoDataset(root_dir=TRAIN_DIR, frames_per_video=frames_per_video, transform=train_transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Dry-run check
    sample_frames, sample_labels = next(iter(loader))
    assert sample_frames.dim() == 5, f"Expected 5D frames [B, T, C, H, W], got {sample_frames.shape}"
    assert sample_frames.size(1) == frames_per_video, f"Expected {frames_per_video} frames per video, got {sample_frames.size(1)}"
    assert sample_labels.size(0) == sample_frames.size(0), "Mismatch between frames and labels batch size"

    print(f"✅ Train loader ready: {len(loader)} batches, sample frames shape: {sample_frames.shape}")
    return loader


def get_test_loader(batch_size=4, num_workers=0, frames_per_video=8):
    """
    Returns a DataLoader for the test/validation set.
    Includes dry-run checks to ensure batches are valid.
    """
    dataset = VideoDataset(root_dir=TEST_DIR, frames_per_video=frames_per_video, transform=test_transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Dry-run check
    sample_frames, sample_labels = next(iter(loader))
    assert sample_frames.dim() == 5, f"Expected 5D frames [B, T, C, H, W], got {sample_frames.shape}"
    assert sample_frames.size(1) == frames_per_video, f"Expected {frames_per_video} frames per video, got {sample_frames.size(1)}"
    assert sample_labels.size(0) == sample_frames.size(0), "Mismatch between frames and labels batch size"

    print(f"✅ Test loader ready: {len(loader)} batches, sample frames shape: {sample_frames.shape}")
    return loader
