# src/models/evaluator.py

import torch
from torch.utils.data import DataLoader

class Evaluator:
    """
    Evaluate CNNTemporal model on validation/test data
    """

    def __init__(self, model, data_loader: DataLoader, device='cuda'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for frames, labels in self.data_loader:
                frames, labels = frames.to(self.device), labels.to(self.device)
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = torch.max(outputs,1)
                total += labels.size(0)
                correct += (preds==labels).sum().item()

        avg_loss = total_loss/len(self.data_loader)
        accuracy = 100*correct/total
        return avg_loss, accuracy
