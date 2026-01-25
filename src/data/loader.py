from torch.utils.data import DataLoader
from src.data.dataset import train_dataset , test_dataset

train_dataset=DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=1
)

test_dataset=DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1
)