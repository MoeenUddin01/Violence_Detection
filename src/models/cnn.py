# src/models/cnn.py

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN + Temporal Average Pooling for video classification.
    Input: (batch_size, num_frames, 3, H, W)
    Output: class scores
    """

    def __init__(self, num_classes=2, input_size=112):
        super(CNN, self).__init__()

        # CNN backbone (applied to each frame)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Dynamically compute flattened size for FC layer
        dummy = torch.zeros(1, 3, input_size, input_size)
        dummy = self.cnn(dummy)
        flattened_size = dummy.numel()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Merge batch and time: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Pass through CNN
        features = self.cnn(x)  # (B*T, 64, H_out, W_out)

        # Flatten
        features = features.view(B * T, -1)

        # Pass through fully connected layers
        out = self.fc(features)  # (B*T, num_classes)

        # Reshape back: (B, T, num_classes)
        out = out.view(B, T, -1)

        # Temporal average pooling over frames
        out = out.mean(dim=1)  # (B, num_classes)

        return out
