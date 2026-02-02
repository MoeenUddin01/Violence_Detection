# src/models/evaluation.py

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import wandb
import numpy as np

class Evaluator:
    """
    Evaluator for video classification models.
    Computes loss, accuracy, confusion matrix, precision, recall, F1.
    Logs all metrics to W&B.
    """

    def __init__(self, model, data_loader, device='cuda'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for frames, labels in self.data_loader:
                frames, labels = frames.to(self.device), labels.to(self.device)
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Combine all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Metrics
        epoch_loss = running_loss / len(self.data_loader)
        epoch_acc = 100 * (all_preds == all_labels).sum().item() / len(all_labels)

        # Confusion matrix
        cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())

        # Precision, Recall, F1
        precision = precision_score(all_labels.numpy(), all_preds.numpy(), average='macro', zero_division=0)
        recall = recall_score(all_labels.numpy(), all_preds.numpy(), average='macro', zero_division=0)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro', zero_division=0)

        # Log metrics to W&B
        wandb.log({
            "Val Loss": epoch_loss,
            "Val Accuracy": epoch_acc,
            "Val Precision": precision,
            "Val Recall": recall,
            "Val F1": f1,
            "Confusion Matrix": wandb.Table(data=cm.tolist(), columns=[f"Pred_{i}" for i in range(cm.shape[0])])
        })

        return epoch_loss, epoch_acc, precision, recall, f1, cm
