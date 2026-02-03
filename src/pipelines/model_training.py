# src/pipelines/model_training.py

import os
from datetime import datetime
import torch
from tqdm import tqdm
from src.data.loader import get_train_loader
from src.models.cnn import CNN
from src.models.train import Trainer
import wandb

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Hyperparameters
    BATCH_SIZE = 2      # Safe for Colab GPU
    EPOCHS = 10
    LR = 1e-3
    FRAMES_PER_VIDEO = 4

    # Checkpoint directory on Drive
    checkpoint_dir = "/content/drive/MyDrive/Violence_Detection/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")

    # Initialize W&B
    wandb.init(
        project="Violence-Detection-CNN",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "frames_per_video": FRAMES_PER_VIDEO,
            "device": DEVICE
            
        },
        id="2p476n8w",
        resume="must",
        
        name=f"Run-{datetime.now().strftime('%d_%m_%Y_%H_%M')}"
    )

    # Load data
    train_loader = get_train_loader(batch_size=BATCH_SIZE, frames_per_video=FRAMES_PER_VIDEO)
    print("Train batches:", len(train_loader))

    # Dry-run check
    sample_frames, sample_labels = next(iter(train_loader))
    print("Sample frames shape:", sample_frames.shape)
    print("Sample labels shape:", sample_labels.shape)

    # Model & Trainer
    model = CNN(num_classes=2).to(DEVICE)
    trainer = Trainer(model=model, learning_rate=LR, device=DEVICE)

    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"🔁 Resuming training from epoch {start_epoch}")
    else:
        print("🆕 Starting fresh training")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🔥 Starting epoch {epoch + 1}/{EPOCHS}")
        torch.cuda.empty_cache()

        train_loss, train_acc = trainer.train_one_epoch(train_loader)

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # W&B logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1e6
        })

        # Save checkpoint
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
