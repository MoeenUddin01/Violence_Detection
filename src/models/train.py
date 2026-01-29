import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader

# =========================
# logging configuration
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
        train_loader: DataLoader,
        learning_rate: float,
        device: str
    ):
        try:
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.train_loader = train_loader

            self.criterion = nn.CrossEntropyLoss()
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
    def train_one_epoch(self, epoch: int):
        """
        Train the model for a single epoch
        """
        try:
            self.model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            logger.info(f"Starting training for epoch {epoch}")

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # ------------------
                # Forward pass
                # ------------------
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # ------------------
                # Backward pass
                # ------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ------------------
                # Metrics
                # ------------------
                running_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx} | "
                        f"Loss: {loss.item():.4f}"
                    )

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100.0 * correct / total if total > 0 else 0.0

            logger.info(
                f"Epoch {epoch} completed | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_accuracy:.2f}%"
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
        Save trained model weights
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
