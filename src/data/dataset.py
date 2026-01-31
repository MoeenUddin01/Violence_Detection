# src/data/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=16, transform=None):
        """
        root_dir: path to train or test folder
        frames_per_video: number of frames to take per video
        transform: transform function applied to each frame
        """
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.transform = transform

        self.video_paths = []
        self.labels = []

        # Read classes
        class_names = sorted(os.listdir(root_dir))
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.video_paths.append(os.path.join(class_path, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def _read_video(self, video_path):
        """Read frames safely and return a fixed number of frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # standard size
                frames.append(torch.tensor(frame).permute(2, 0, 1).float()/255.0)

            cap.release()

            if len(frames) == 0:
                raise ValueError(f"No frames found in {video_path}")

            # Repeat last frame if fewer frames
            while len(frames) < self.frames_per_video:
                frames.append(frames[-1])

            return frames[:self.frames_per_video]

        except Exception as e:
            print(f"[ERROR] Failed to read {video_path}: {e}")
            dummy_frame = torch.zeros(3, 224, 224)
            return [dummy_frame] * self.frames_per_video

    def __getitem__(self, idx):
        frames = self._read_video(self.video_paths[idx])
        if self.transform:
            frames = [self.transform(f) for f in frames]
        frames = torch.stack(frames).mean(dim=0)  # (C,H,W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label
