# src/data/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """
    Professional Video Dataset:
    - Samples exactly `frames_per_video` frames per video.
    - Uses uniform + repeat sampling.
    - Skips corrupted videos gracefully.
    - Uses logging instead of print.
    """

    def __init__(self, root_dir, frames_per_video=8, transform=None, logger=None):
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.logger = logger

        self.video_paths = []
        self.labels = []

        # Scan directories and map classes to indices
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for video_file in os.listdir(cls_path):
                self.video_paths.append(os.path.join(cls_path, video_file))
                self.labels.append(self.class_to_idx[cls])

        if self.logger:
            self.logger.info(f"Found {len(self.video_paths)} videos in {root_dir}")

    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, path):
        """
        Loads video frames with try-except.
        Returns None if video is corrupted.
        """
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                if self.logger:
                    self.logger.warning(f"Cannot open video: {path}")
                return None

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Skipping frame in {path}: {e}")
            cap.release()

            if len(frames) == 0:
                if self.logger:
                    self.logger.warning(f"No frames read from video: {path}")
                return None

            return frames

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load video {path}: {e}")
            return None

    def _sample_frames(self, frames):
        """
        Uniform + repeat frame sampling.
        """
        num_frames = len(frames)
        if num_frames >= self.frames_per_video:
            interval = num_frames / self.frames_per_video
            indices = [int(i * interval) for i in range(self.frames_per_video)]
        else:
            indices = [i % num_frames for i in range(self.frames_per_video)]
        sampled = [frames[i] for i in indices]
        return sampled

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video(video_path)
        if frames is None:
            return None  # Skip corrupted video

        frames = self._sample_frames(frames)

        if self.transform:
            try:
                frames = [self.transform(frame) for frame in frames]
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Transform failed for {video_path}: {e}")
                return None  # Skip video if transform fails

        frames = torch.stack(frames)  # (T, C, H, W)
        return frames, torch.tensor(label, dtype=torch.long)
