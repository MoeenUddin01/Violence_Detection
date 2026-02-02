# src/models/train.py

import torch
import torch.nn as nn

class Trainer:
    """
    Trainer class for CNNTemporal model with robust training.
    """

    def __init__(self, model, learning_rate=1e-3, device='cuda'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in train_loader:
            frames, labels = frames.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            try:
                outputs = self.model(frames)
            except RuntimeError as e:
                print("⚠️ RuntimeError in forward pass:", e)
                continue

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    def save_model(self, path="best_model.pth"):
        torch.save({"model_state_dict": self.model.state_dict()}, path)
        return path
