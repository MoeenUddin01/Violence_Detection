# src/data/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """
    Dataset for video classification.
    Returns a sequence of frames per video with label.
    Output shape: (num_frames, 3, 224, 224)
    """

    def __init__(self, root_dir, frames_per_video=16, transform=None):
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.transform = transform

        self.video_paths = []
        self.labels = []

        # Collect video paths and labels
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
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame_tensor = torch.tensor(frame).permute(2,0,1).float()/255.0
                frames.append(frame_tensor)
            cap.release()
            
            if len(frames) == 0:
                raise ValueError(f"No frames in {video_path}")

            # Repeat last frame if not enough
            while len(frames) < self.frames_per_video:
                frames.append(frames[-1])
            # Sample evenly if too many frames
            if len(frames) > self.frames_per_video:
                indices = torch.linspace(0,len(frames)-1,steps=self.frames_per_video).long()
                frames = [frames[i] for i in indices]

            if self.transform:
                frames = [self.transform(f) for f in frames]

            return torch.stack(frames)

        except Exception as e:
            print(f"[ERROR] Failed to read {video_path}: {e}")
            dummy_frame = torch.zeros(3,224,224)
            return torch.stack([dummy_frame]*self.frames_per_video)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._read_video(video_path)
        return frames, torch.tensor(label,dtype=torch.long)
