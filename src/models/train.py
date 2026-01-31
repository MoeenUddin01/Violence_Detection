# src/models/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, learning_rate, device):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_one_epoch(self, epoch, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def save_model(self, path="model.pth"):
        torch.save({"model_state_dict": self.model.state_dict()}, path)
        return path
