from torchvision import transforms

# =========================
# 1. TRANSFORMS FOR VIDEO FRAMES
# =========================

# Transform applied to EACH FRAME during training
train_transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert OpenCV frame (numpy) to PIL Image
    transforms.Resize((224, 224)),      # Resize frame
    transforms.RandomHorizontalFlip(),  # Augmentation: flip randomly
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2
    ),                                   # Random color changes
    transforms.ToTensor(),               # Convert PIL Image -> Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )                                    # Normalize like ImageNet
])

# Transform applied to EACH FRAME during testing/validation
test_transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert OpenCV frame (numpy) to PIL Image
    transforms.Resize((224, 224)),      # Resize frame
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
