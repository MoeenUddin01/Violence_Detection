# src/models/evaluation.py

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    """
    Evaluates a model on a validation/test set and logs metrics to W&B.
    """

    def __init__(self, model, data_loader, device='cuda'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0

        with torch.no_grad():
            for frames, labels in self.data_loader:
                frames, labels = frames.to(self.device), labels.to(self.device)
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        val_loss = running_loss / len(self.data_loader)
        val_acc = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.close()

        return val_loss, val_acc, precision, recall, f1
