# src/pipelines/model_training.py
import sys
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.data.loader import get_train_loader, get_test_loader
from src.models.cnn import CNN
from src.models.train import Trainer
from src.models.evaluation import Evaluator

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 1e-3
    FRAMES_PER_VIDEO = 8
    EARLY_STOPPING_PATIENCE = 5

    # W&B init
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

    # Load data
    train_loader = get_train_loader(BATCH_SIZE, num_workers=0, frames_per_video=FRAMES_PER_VIDEO)
    test_loader = get_test_loader(BATCH_SIZE, num_workers=0, frames_per_video=FRAMES_PER_VIDEO)
    sample_frames, sample_labels = next(iter(train_loader))
    print("Sample frames shape:", sample_frames.shape)

    # Model setup
    model = CNN(num_classes=2)
    trainer = Trainer(model=model, learning_rate=LR, device=DEVICE)
    evaluator = Evaluator(model=model, data_loader=test_loader, device=DEVICE)

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")
        torch.cuda.empty_cache()
        running_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} batches"):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            trainer.optimizer.zero_grad()
            outputs = trainer.model(frames)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        val_loss, val_acc, precision, recall, f1 = evaluator.evaluate()

        # Log all metrics
        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "GPU Memory (MB)": torch.cuda.memory_allocated() / 1e6
        })

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | F1: {f1:.4f}")

        # Best model saving + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = trainer.save_model(f"best_model_epoch{epoch+1}.pth")
            print(f"✅ Saved best model at {path}")
            wandb.save(path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

    print("\n🏁 Training complete!")

if __name__ == "__main__":
    main()
