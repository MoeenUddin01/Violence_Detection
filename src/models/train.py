# src/models/train.py

import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader

# =========================
# Logging Configuration
# =========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    """
    Trainer class responsible for training the CNN model
    on video frames (violence / non-violence).
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        device: str
    ):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The CNN model to train.
            learning_rate (float): Learning rate for the optimizer.
            device (str): "cuda" or "cpu".
        """
        try:
            self.device = torch.device(device)
            self.model = model.to(self.device)

            # Loss function
            self.criterion = nn.CrossEntropyLoss()

            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )

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

        Args:
            epoch (int): Current epoch number.
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            Tuple: (epoch_loss, epoch_accuracy)
        """
        try:
            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            logger.info(f"Starting training for epoch {epoch}")

            for batch_idx, (images, labels) in enumerate(train_loader):  # ✅ Use passed DataLoader
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

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}"
                    )

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100.0 * correct / total if total > 0 else 0.0

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
    def save_model(self, save_path: str):
        """
        Save trained model weights.

        Args:
            save_path (str): Path to save the model.

        Returns:
            str: Path where the model was saved.
        """
        try:
            torch.save(
                {"model_state_dict": self.model.state_dict()},
                save_path
            )
            logger.info(f"Model saved at {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e
