# src/pipelines/model_training.py
import sys
import os


import torch
from torch.utils.data import DataLoader


from src.data.loader import get_train_loader, get_test_loader
from src.models.cnn import CNN         # <-- use your existing cnn.py
from src.models.train import Trainer
from src.models.evaluation import Evaluator

import wandb
from datetime import datetime
import os

def main():
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 1e-3
    FRAMES_PER_VIDEO = 16

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
        name=f"Run-{datetime.now().strftime('%d_%m_%Y_%H_%M')}"
    )

    # DataLoaders
    train_loader = get_train_loader(BATCH_SIZE, num_workers=2, frames_per_video=FRAMES_PER_VIDEO)
    test_loader = get_test_loader(BATCH_SIZE, num_workers=2, frames_per_video=FRAMES_PER_VIDEO)

    # Model and Trainer
    model = CNN(num_classes=2)
    trainer = Trainer(model=model, learning_rate=LR, device=DEVICE)
    evaluator = Evaluator(model=model, data_loader=test_loader, device=DEVICE)

    best_acc = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = evaluator.evaluate()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Log to W&B
        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc
        })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            path = trainer.save_model(f"best_model_epoch{epoch+1}.pth")
            print(f"Saved best model at {path}")
            wandb.save(path)

if __name__ == "__main__":
    wandb.login()
    main()
