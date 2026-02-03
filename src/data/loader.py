# src/data/loader.py
import logging
from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset
from src.data.transform import train_transform, test_transform

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Collate function to skip corrupted videos
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    videos, labels = zip(*batch)
    return torch.stack(videos), torch.tensor(labels)

train_dir = "datas/processed/train"
test_dir = "datas/processed/test"

def get_train_loader(batch_size=2, num_workers=0, frames_per_video=8):
    dataset = VideoDataset(
        root_dir=train_dir,
        frames_per_video=frames_per_video,
        transform=train_transform,
        logger=logger  # pass logger to dataset
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    return loader

def get_test_loader(batch_size=2, num_workers=0, frames_per_video=8):
    dataset = VideoDataset(
        root_dir=test_dir,
        frames_per_video=frames_per_video,
        transform=test_transform,
        logger=logger  # pass logger to dataset
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    return loader
