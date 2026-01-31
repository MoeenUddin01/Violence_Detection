# src/models/train.py

import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
import os

# =========================
# Logging Configuration
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, model, learning_rate=0.001, device="cpu", save_dir="saved_models"):
        """
        model: your CNN model
        learning_rate: optimizer learning rate
        device: 'cuda' or 'cpu'
        save_dir: folder to save best models
        """
        try:
            self.device = torch.device(device)
            self.model = model.to(self.device)

            # Loss function
            self.criterion = nn.CrossEntropyLoss()

            # Optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Directory to save models
            self.save_dir = save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            logger.info("Trainer initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing Trainer: {e}")
            raise e

    # =========================
    # Training Loop (One Epoch)
    # =========================
    def train_one_epoch(self, epoch: int, train_loader: DataLoader):
        """
        Train the model for a single epoch.

        Returns:
            Tuple: (epoch_loss, epoch_accuracy)
        """
        try:
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            logger.info(f"Starting training for epoch {epoch}")

            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    # Move data to device
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Metrics
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                except Exception as e:
                    logger.warning(f"[WARNING] Skipped batch {batch_idx} due to error: {e}")

            epoch_loss = running_loss / max(1, len(train_loader))
            epoch_accuracy = 100.0 * correct / max(1, total)

            logger.info(
                f"Epoch {epoch} completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%"
            )

            return epoch_loss, epoch_accuracy

        except Exception as e:
            logger.error(f"Error during training at epoch {epoch}: {e}")
            raise e

    # =========================
    # Save Model
    # =========================
    def save_model(self, filename="best_model.pth"):
        """
        Save trained model weights.

        Returns:
            str: Path where the model was saved.
        """
        save_path = os.path.join(self.save_dir, filename)
        try:
            torch.save({"model_state_dict": self.model.state_dict()}, save_path)
            logger.info(f"Model saved at {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
