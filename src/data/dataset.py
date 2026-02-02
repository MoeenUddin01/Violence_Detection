# src/data/dataset.py
import os
import random
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=8, transform=None):
        """
        Args:
            root_dir (str): path to folder with class subfolders
            frames_per_video (int): number of frames per video to sample
            transform (callable): optional transform to apply to each frame
        """
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video
        self.transform = transform

        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_folder = os.path.join(root_dir, cls)
            for file in os.listdir(cls_folder):
                if file.endswith((".mp4", ".avi")):
                    self.video_paths.append(os.path.join(cls_folder, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video frames
        frames = self._load_video(video_path)

        if self.transform:
            frames = [self.transform(f) for f in frames]

        # Stack frames to (T, C, H, W)
        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.long)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = self._sample_frames(total_frames)

        frames = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                # fallback: use last successfully read frame
                frame = frames[-1] if frames else np.zeros((224,224,3), np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        cap.release()
        return frames

    def _sample_frames(self, total_frames):
        """
        Uniform + random sampling:
        - Divide video into `frames_per_video` segments
        - Randomly pick one frame from each segment
        """
        if total_frames < self.frames_per_video:
            # Repeat some frames if video too short
            frame_ids = list(range(total_frames))
            while len(frame_ids) < self.frames_per_video:
                frame_ids.append(random.choice(frame_ids))
        else:
            seg_size = total_frames / self.frames_per_video
            frame_ids = [int(seg_size*i + random.uniform(0, seg_size)) for i in range(self.frames_per_video)]
            frame_ids = [min(fid, total_frames-1) for fid in frame_ids]
        return frame_ids
