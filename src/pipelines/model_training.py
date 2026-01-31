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
        BATCH_SIZE = 4           # Reduce batch size for video data to avoid memory issues
        LEARNING_RATE = 0.001
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        print("Using device:", DEVICE)

        # -----------------------------
        # W&B Config
        # -----------------------------
        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": "CNN"
        }

        wandb.init(
            project="Violence-Detection-CNN",
            config=config,
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
        )

        # -----------------------------
        # Model Initialization
        # -----------------------------
        model = CNN().to(DEVICE)
        torch.set_default_device(DEVICE)  # Ensure PyTorch uses GPU

        # -----------------------------
        # Load Data
        # -----------------------------
        # Set num_workers=0 on Colab to avoid 'cuda/cpu generator' error
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
            train_loss_total = 0.0
            train_correct_total = 0
            train_samples = 0

            # -----------------------------
            # Training Loop
            # -----------------------------
            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # Corrected method: train_one_batch (not train_batch)
                    loss, correct = trainer.train_one_batch(images, labels)

                    train_loss_total += loss
                    train_correct_total += correct
                    train_samples += labels.size(0)

                except Exception as e:
                    print(f"[WARNING] Skipped batch {batch_idx} due to error: {e}")

            train_loss = train_loss_total / max(1, len(train_loader))
            train_acc = 100.0 * train_correct_total / max(1, train_samples)

            # -----------------------------
            # Evaluation Loop
            # -----------------------------
            try:
                val_loss, val_acc = evaluator.evaluate()
            except Exception as e:
                print(f"[WARNING] Evaluation failed at epoch {epoch}: {e}")
                val_loss, val_acc = 0.0, 0.0

            # -----------------------------
            # Logging
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
                saved_model_path = trainer.save_model()
                if saved_model_path:
                    print(f"Model with Accuracy {val_acc:.4f} Saved Successfully")
                    wandb.save(saved_model_path)

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise e

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Colab environment: provide W&B key via environment variable or manually
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()
