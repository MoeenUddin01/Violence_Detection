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
    Includes safeguards for corrupt or very short videos.
    """
    def __init__(self, root_dir, frames_per_video=8, transform=None, verbose=True):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.verbose = verbose

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

        if self.verbose:
            print(f"Found {len(self.video_paths)} videos in {root_dir}")

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        if not cap.isOpened():
            if self.verbose:
                print(f"⚠️ Could not open video: {path}")
            return frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Skipping frame in {path} due to: {e}")
        cap.release()
        return frames

    def _sample_frames(self, frames):
        num_frames = len(frames)
        if num_frames == 0:
            # Return dummy frames if video failed
            return [torch.zeros(3, 112, 112)] * self.frames_per_video

        if num_frames >= self.frames_per_video:
            # Uniform sampling
            interval = num_frames / self.frames_per_video
            indices = [int(i * interval) for i in range(self.frames_per_video)]
        else:
            # Repeat frames if too short
            indices = [i % num_frames for i in range(self.frames_per_video)]

        sampled = [frames[i] for i in indices]
        return sampled

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)

        if self.transform:
            try:
                frames = [self.transform(frame) for frame in frames]
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Transform failed for {video_path}: {e}")
                frames = [torch.zeros(3, 112, 112)] * self.frames_per_video

        frames = torch.stack(frames)  # (T, C, H, W)
        return frames, torch.tensor(label, dtype=torch.long)
