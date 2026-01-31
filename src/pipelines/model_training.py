# src/pipelines/model_training.py

import torch
from src.data.loader import get_train_loader, get_test_loader
from src.models.cnn import CNN
from src.models.train import Trainer
from src.models.evaluation import Evaluator
import wandb
import os
from datetime import datetime

def main():
    try:
        # -----------------------------
        # Training Configuration
        # -----------------------------
        EPOCHS = 10
        BATCH_SIZE = 16
        LEARNING_RATE = 0.001
        DEVICE = "cuda"                                                                             
        print("Using device:", DEVICE)

        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": "CNN"
        }

        # -----------------------------
        # Initialize W&B
        # -----------------------------
        wandb.init(
            project="Violence-Detection-CNN",
            config=config,
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
        )

        # -----------------------------
        # Model Initialization
        # -----------------------------
        model = CNN().to(DEVICE)

        # -----------------------------
        # Load Data (num_workers=0 for Colab GPU)
        # -----------------------------
        train_loader = get_train_loader(batch_size=BATCH_SIZE, num_workers=0)
        test_loader = get_test_loader(batch_size=BATCH_SIZE, num_workers=0)

        # -----------------------------
        # Trainer & Evaluator
        # -----------------------------
        trainer = Trainer(model=model, learning_rate=LEARNING_RATE, device=DEVICE)
        evaluator = Evaluator(data=test_loader, model=model, device=DEVICE)

        BEST_ACCURACY = 0

        # -----------------------------
        # Epoch Loop
        # -----------------------------
        for epoch in range(EPOCHS):
            # Train one epoch
            train_loss, train_acc = trainer.train_one_epoch(epoch, train_loader)

            # Evaluate
            try:
                val_loss, val_acc = evaluator.evaluate()
            except Exception as e:
                print(f"[WARNING] Evaluation failed at epoch {epoch}: {e}")
                val_loss, val_acc = 0.0, 0.0

            # -----------------------------
            # Logging to W&B
            # -----------------------------
            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })

            # -----------------------------
            # Save Best Model
            # -----------------------------
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                saved_model_path = f"violence_detection_epoch{epoch+1}.pth"
                trainer.save_model(saved_model_path)
                print(f"Best Model Saved with Accuracy: {val_acc:.2f}%")

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise e

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Login to W&B (optional)
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()
