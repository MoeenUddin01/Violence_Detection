# src/pipelines/model_training.py

import torch
from src.data.loader import get_train_loader, get_test_loader  # Your data loader functions
from src.models.cnn import CNN                                  # Your CNN model
from src.models.train import Trainer                            # Training loop class
from src.models.evaluation import Evaluator                    # Evaluation class
import wandb                                                    # Weights & Biases for experiment tracking
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
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        wandb.init(
            project="Violence-Detection-CNN",
            config=config,
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
        )

        # -----------------------------
        # Model Initialization
        # -----------------------------
        model = CNN().to(DEVICE)
        print("Using device:", DEVICE)
        torch.set_default_device(DEVICE)

        # -----------------------------
        # Create DataLoaders (FIXED)
        # -----------------------------
        train_loader = get_train_loader(batch_size=BATCH_SIZE)
        test_loader = get_test_loader(batch_size=BATCH_SIZE)

        # -----------------------------
        # Initialize Trainer and Evaluator
        # -----------------------------
        trainer = Trainer(
            model=model,
            learning_rate=LEARNING_RATE,
            device=DEVICE
                        )

        evaluator = Evaluator(
            
            data=test_loader,
            model=model,
            device=DEVICE
        )

        BEST_ACCURACY = 0

        # -----------------------------
        # Epoch Loop
        # -----------------------------
        for epoch in range(EPOCHS):
            train_loss, _, train_acc = trainer.start_training_loop(epoch)
            val_loss, _, val_acc = evaluator.start_evaluation_loop(epoch)

            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })

            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                saved_model_path = trainer.save_model()
                if saved_model_path:
                    print(f"Model with Accuracy {val_acc:.4f} Saved Successfully")
                    wandb.log_model(
                        saved_model_path,
                        "violence_detection_cnn",
                        aliases=[f"epoch-{epoch+1}"]
                    )

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise e

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()
