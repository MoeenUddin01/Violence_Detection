# src/data/loader.py

from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset
from src.data.transform import train_transform, test_transform

train_dir = "datas/processed/train"
test_dir = "datas/processed/test"

def get_train_loader(batch_size=4, num_workers=0, frames_per_video=16):
    dataset = VideoDataset(root_dir=train_dir, frames_per_video=frames_per_video, transform=train_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def get_test_loader(batch_size=4, num_workers=0, frames_per_video=16):
    dataset = VideoDataset(root_dir=test_dir, frames_per_video=frames_per_video, transform=test_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
