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

        # Load video frames -> list of (H, W, 3)
        frames = self._load_video(video_path)

        processed_frames = []
        for frame in frames:
            # frame is numpy array (H, W, 3)
            if self.transform:
                frame = self.transform(frame)   # -> (3, 112, 112)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            processed_frames.append(frame)

        # Stack -> (T, 3, 112, 112)
        frames = torch.stack(processed_frames)

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
                # fallback frame (112x112 to match pipeline)
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()
        return frames

    def _sample_frames(self, total_frames):
        """
        Uniform + random sampling:
        - Divide video into `frames_per_video` segments
        - Randomly pick one frame from each segment
        """
        if total_frames < self.frames_per_video:
            frame_ids = list(range(total_frames))
            while len(frame_ids) < self.frames_per_video:
                frame_ids.append(random.choice(frame_ids))
        else:
            seg_size = total_frames / self.frames_per_video
            frame_ids = [
                int(seg_size * i + random.uniform(0, seg_size))
                for i in range(self.frames_per_video)
            ]
            frame_ids = [min(fid, total_frames - 1) for fid in frame_ids]

        return frame_ids
