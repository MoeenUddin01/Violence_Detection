# src/models/cnn_temporal.py

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN + Temporal Average Pooling for video classification.
    Input: (batch_size, num_frames, 3, 224, 224)
    Output: class scores
    """

    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        # CNN backbone (applied to each frame)
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # After 3 pooling layers, 224x224 -> 28x28
        self.fc = nn.Sequential(
            nn.Linear(64*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape
        # Merge batch and time: (B*T, C, H, W)
        x = x.view(B*T, C, H, W)
        features = self.cnn(x)  # (B*T, 64,28,28)
        features = features.view(B*T, -1)  # Flatten
        # Pass through FC
        out = self.fc(features)  # (B*T, num_classes)
        # Reshape back: (B, T, num_classes)
        out = out.view(B, T, -1)
        # Temporal average pooling over frames
        out = out.mean(dim=1)  # (B, num_classes)
        return out
