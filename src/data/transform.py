# src/data/transform.py

from torchvision import transforms

# Transform applied to each frame during training
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),  # <-- reduced size for faster training
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform applied to each frame during testing/validation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),  # <-- reduced size for faster testing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
