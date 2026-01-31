# src/pipelines/model_training.py

import torch
from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset
from src.models.cnn import CNN
from src.models.train import Trainer
import wandb
import os
from datetime import datetime

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # ------------------ W&B ------------------
    wandb.init(
        project="Violence-Detection-CNN",
        config={
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE
        },
        name=f"Experiment-{datetime.now().strftime('%d_%m_%Y_%H_%M')}"
    )

    # ------------------ Dataset ------------------
    train_dataset = VideoDataset(root_dir="datas/processed/train")
    test_dataset = VideoDataset(root_dir="datas/processed/test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ------------------ Model & Trainer ------------------
    model = CNN().to(DEVICE)
    trainer = Trainer(model=model, learning_rate=LEARNING_RATE, device=DEVICE)

    # ------------------ Training Loop ------------------
    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = trainer.train_one_epoch(epoch, train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        wandb.log({
            "Epoch": epoch+1,
            "Training Loss": train_loss,
            "Training Accuracy": train_acc
        })

        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            path = trainer.save_model(f"best_model_epoch{epoch+1}.pth")
            print(f"Saved model at {path}")
            wandb.save(path)

if __name__ == "__main__":
    wandb.login()
    main()
