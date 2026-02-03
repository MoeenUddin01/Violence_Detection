# src/pipelines/model_training.py

import os
from datetime import datetime

import torch
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
    BATCH_SIZE = 1
    EPOCHS = 10
    LR = 1e-3
    FRAMES_PER_VIDEO = 2

    # --------------------------
    # Checkpoint directory
    # --------------------------
    checkpoint_dir = "/content/drive/MyDrive/Violence_Detection/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")

    # --------------------------
    # Initialize W&B (resume previous run)
    # --------------------------
    wandb.init(
        project="Violence-Detection-CNN",
        id="2p476n8w",       # previous run ID
        resume="must",       # resume this run
        name=f"Run-{datetime.now().strftime('%d_%m_%Y_%H_%M')}",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "frames_per_video": FRAMES_PER_VIDEO,
            "device": DEVICE
        },
        settings=wandb.Settings(start_method="thread")  # enable system logging
    )

    # --------------------------
    # Load data
    # --------------------------
    train_loader = get_train_loader(
        batch_size=BATCH_SIZE,
        num_workers=0,
        frames_per_video=FRAMES_PER_VIDEO
    )

    val_loader = get_test_loader(
        batch_size=BATCH_SIZE,
        num_workers=0,
        frames_per_video=FRAMES_PER_VIDEO
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------
    # Dry-run batch check
    # --------------------------
    sample_frames, sample_labels = next(iter(train_loader))
    print("Sample frames shape:", sample_frames.shape)
    print("Sample labels shape:", sample_labels.shape)

    # --------------------------
    # Model & Trainer
    # --------------------------
    model = CNN(num_classes=2).to(DEVICE)
    trainer = Trainer(model=model, learning_rate=LR, device=DEVICE)

    # --------------------------
    # Enable W&B system logging
    # --------------------------
    wandb.watch(trainer.model, log="all", log_freq=10)

    # --------------------------
    # Resume training if checkpoint exists
    # --------------------------
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔁 Resuming training from epoch {start_epoch}")
    else:
        print("🆕 Starting fresh training")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🔥 Starting epoch {epoch + 1}/{EPOCHS}")
        torch.cuda.empty_cache()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            trainer.optimizer.zero_grad()

            try:
                outputs = trainer.model(frames)
                loss = trainer.criterion(outputs, labels)
                loss.backward()
                trainer.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                print(f"⚠️ RuntimeError in batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue

        # --------------------------
        # Epoch metrics
        # --------------------------
        average_epoch_training_loss = running_loss / len(train_loader)
        epoch_training_acc = 100 * correct / total

        # --------------------------
        # Validation loop
        # --------------------------
        trainer.model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc="Validation"):
                frames = frames.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = trainer.model(frames)
                loss = trainer.criterion(outputs, labels)

                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        average_epoch_validation_loss = val_running_loss / len(val_loader)
        epoch_validation_acc = 100 * val_correct / val_total

        # --------------------------
        # ✅ W&B logging: only epoch-level metrics
        # --------------------------
        wandb.log(
            {
                "Training Loss": average_epoch_training_loss,
                "Validation Loss": average_epoch_validation_loss,
                "Epoch": epoch + 1,
                "Training Accuracy": epoch_training_acc,
                "Validation Accuracy": epoch_validation_acc
            }
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {average_epoch_training_loss:.4f} | "
            f"Val Loss: {average_epoch_validation_loss:.4f} | "
            f"Train Acc: {epoch_training_acc:.2f}% | "
            f"Val Acc: {epoch_validation_acc:.2f}%"
        )

        # --------------------------
        # Save checkpoint
        # --------------------------
        torch.save({
            "epoch": epoch,
            "model_state": trainer.model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict()
        }, checkpoint_path)

        print(f"💾 Checkpoint saved to {checkpoint_path}")
        wandb.save(checkpoint_path)

    print("\n🏁 Training complete!")


if __name__ == "__main__":
    main()
