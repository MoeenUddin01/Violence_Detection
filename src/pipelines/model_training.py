# src/pipelines/model_training.py

import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import get_train_loader, get_test_loader
from src.models.cnn import CNN
from src.models.train import Trainer
import wandb

def main():
    # --------------------------
    # Device setup
    # --------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --------------------------
    # Hyperparameters
    # --------------------------
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-3
    FRAMES_PER_VIDEO = 8

    # --------------------------
    # Checkpoint folder
    # --------------------------
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------
    # Initialize W&B
    # --------------------------
    wandb.init(
        project="Violence-Detection-CNN",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "frames_per_video": FRAMES_PER_VIDEO,
            "device": DEVICE
        },
        name=f"Run-{datetime.now().strftime('%d_%m_%Y_%H_%M')}"
    )

    # --------------------------
    # Load data
    # --------------------------
    train_loader = get_train_loader(BATCH_SIZE, num_workers=0, frames_per_video=FRAMES_PER_VIDEO)
    test_loader = get_test_loader(BATCH_SIZE, num_workers=0, frames_per_video=FRAMES_PER_VIDEO)

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    # Dry-run batch check
    sample_frames, sample_labels = next(iter(train_loader))
    print("Sample frames shape:", sample_frames.shape)
    print("Sample labels shape:", sample_labels.shape)
    assert len(sample_frames.shape) == 5, "Frames should be 5D: [batch, frames, C, H, W]"
    assert sample_frames.size(1) == FRAMES_PER_VIDEO, "Frames per video mismatch!"

    # --------------------------
    # Model, Trainer
    # --------------------------
    model = CNN(num_classes=2).to(DEVICE)
    trainer = Trainer(model=model, learning_rate=LR, device=DEVICE)

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(EPOCHS):
        print(f"\n🔥 Starting epoch {epoch+1}/{EPOCHS}")
        torch.cuda.empty_cache()  # Free GPU memory

        running_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} batches"):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            # Forward + backward
            trainer.optimizer.zero_grad()
            outputs = trainer.model(frames)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        # Train metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # W&B logging
        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "GPU Memory (MB)": torch.cuda.memory_allocated() / 1e6
        })

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
        trainer.save_model(checkpoint_path)
        print(f"✅ Saved model at {checkpoint_path}")
        wandb.save(checkpoint_path)

    print("\n🏁 Training complete!")

if __name__ == "__main__":
    main()
