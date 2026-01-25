import os
import cv2
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=16):
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video

        # Lists to store video paths and labels
        self.video_paths = []
        self.labels = []

        # Folder names = class names
        class_names = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_path, file_name)
                    self.video_paths.append(video_path)
                    self.labels.append(label)

    # Number of samples
    def __len__(self):
        return len(self.video_paths)

    # Read frames from a video
    def _read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        # If video is shorter, repeat last frame
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])

        frames = frames[:self.frames_per_video]
        return frames

    # Get one sample
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._read_video(video_path)

        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

        return frames, label
