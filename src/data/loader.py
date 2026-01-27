#src/data/loader.py


from torch.utils.data import DataLoader
from src.data.datset import VideoDataset
from src.data.transform import train_transform, test_transform

# =========================
# paths to data directories
# =========================
train_dir = "datas/processed/train"
test_dir = "datas/processed/test"

# =========================
# DataLoader for training set
# =========================

def get_train_loader(batch_size=4, num_workers=2):
    train_dataset =VideoDataset(
        root_dir=train_dir,
        frames_per_video=16,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader

def get_test_loader(batch_size=4, num_workers=2):
    test_dataset=VideoDataset(
        root_dir=test_dir,
        frames_per_video=16,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader
    