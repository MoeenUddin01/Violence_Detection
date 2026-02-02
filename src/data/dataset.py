# src/data/dataset.py

import os
import random
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """
    Video dataset that samples exactly `frames_per_video` frames per video.
    Uses uniform + random sampling. No repeated last frame.
    """
    def __init__(self, root_dir, frames_per_video=8, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video

        self.video_paths = []
        self.labels = []

        # Scan directories
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for video_file in os.listdir(cls_path):
                self.video_paths.append(os.path.join(cls_path, video_file))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames):
        num_frames = len(frames)
        if num_frames >= self.frames_per_video:
            # Uniform indices
            interval = num_frames / self.frames_per_video
            indices = [int(i*interval) for i in range(self.frames_per_video)]
        else:
            # If video too short, repeat random frames
            indices = [i % num_frames for i in range(self.frames_per_video)]
        sampled = [frames[i] for i in indices]
        return sampled

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)  # (T, C, H, W)
        return frames, torch.tensor(label, dtype=torch.long)
