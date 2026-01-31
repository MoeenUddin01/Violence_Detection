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

        # Get class folders
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
        """Read video frames safely with try-except"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise ValueError(f"No frames in video: {video_path}")

            # Repeat last frame if video has fewer frames than required
            while len(frames) < self.frames_per_video:
                frames.append(frames[-1])

            return frames[:self.frames_per_video]

        except Exception as e:
            print(f"[ERROR] Failed to read {video_path}: {e}")
            # Return black frames if video fails
            dummy_frame = torch.zeros(3, 224, 224)
            return [dummy_frame] * self.frames_per_video

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            frames = self._read_video(video_path)

            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            frames = torch.stack(frames)  # (T, C, H, W)
            frames = frames.mean(dim=0)   # (C, H, W) for CNN

            label = torch.tensor(label, dtype=torch.long)
            return frames, label

        except Exception as e:
            print(f"[ERROR] Failed to load {video_path}: {e}")
            dummy_frame = torch.zeros(3, 224, 224)
            return dummy_frame, torch.tensor(-1, dtype=torch.long)
