from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset
from src.data.transform import train_transform, test_transform

train_dir = "datas/processed/train"
test_dir = "datas/processed/test"

def get_train_loader(batch_size=4, num_workers=2):
    try:
        train_dataset = VideoDataset(
            root_dir=train_dir,
            frames_per_video=16,
            transform=train_transform
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader
    except Exception as e:
        print(f"[ERROR] Failed to create train loader: {e}")
        return None

def get_test_loader(batch_size=4, num_workers=2):
    try:
        test_dataset = VideoDataset(
            root_dir=test_dir,
            frames_per_video=16,
            transform=test_transform
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return test_loader
    except Exception as e:
        print(f"[ERROR] Failed to create test loader: {e}")
        return None
