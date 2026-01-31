# src/pipelines/model_training.py

import torch
from src.data.loader import get_train_loader, get_test_loader   # Your data loaders
from src.models.cnn import CNN                         # Your CNN model
from src.models.train import Trainer                   # Training loop class
from src.models.evaluation import Evaluator           # Evaluation class
import wandb                                         # Weights & Biases for experiment tracking
import os
from datetime import datetime

def main():
    try:
        # -----------------------------
        # Training Configuration
        # -----------------------------
        EPOCHS = 10              # Number of times the model will see the entire dataset
        BATCH_SIZE = 16             # Number of samples per batch
        LEARNING_RATE = 0.001       # Step size for optimizer
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

        # Store config in a dictionary for logging in W&B
        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": CNN
        }

        # -----------------------------
        # Initialize Weights & Biases
        # -----------------------------
        # W&B is a tool to track experiments: loss, accuracy, model checkpoints, etc.
        wandb.init(
            project="Violence-Detection-CNN",  # Name of your W&B project
            config=config,                      # Pass training config
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'  # Unique experiment name
        )

        # -----------------------------
        # Model Initialization
        # -----------------------------
        model = CNN().to(DEVICE)            # Move model to GPU if available
        print("Using device:", DEVICE)
        torch.set_default_device(DEVICE)   # Optional: set default device for tensors

        # -----------------------------
        # Initialize Trainer and Evaluator
        # -----------------------------
        trainer = Trainer(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data=train_loader,
            model=model,
            model_path="artifacts",  # Folder to save model checkpoints
            device=DEVICE
        )

        evaluator = Evaluator(
            batch_size=BATCH_SIZE,
            data=test_loader,
            model=model,
            device=DEVICE
        )

        BEST_ACCURACY = 0  # Keep track of the best validation accuracy to save best model

        # -----------------------------
        # Epoch Loop (Training + Validation)
        # -----------------------------
        for epoch in range(EPOCHS):
            # 1️⃣ Training step
            train_loss, _, train_acc = trainer.start_training_loop(epoch)

            # 2️⃣ Validation step
            val_loss, _, val_acc = evaluator.start_evaluation_loop(epoch)

            # 3️⃣ Log metrics to W&B
            # This will create interactive plots in the W&B dashboard
            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })

            # 4️⃣ Save model if validation accuracy improves
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                saved_model_path = trainer.save_model()  # Save model checkpoint
                if saved_model_path:
                    print(f"Model with Accuracy {val_acc:.4f} Saved Successfully")
                    # Log saved model to W&B for easy versioning
                    wandb.log_model(
                        saved_model_path,
                        "violence_detection_cnn",      # Model name in W&B
                        aliases=[f"epoch-{epoch+1}"]  # Label the checkpoint by epoch
                    )

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise e  # Raise exception for debugging

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Login to W&B using API key from environment variables
    # Make sure you have WANDB_API_KEY set in your system or .env file
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()
