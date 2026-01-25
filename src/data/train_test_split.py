import os 
import random
from pathlib import Path
import shutil

# def split_dataset(
#     soruce_dir,
#     target_dir,
#     train_ratio=0.8,
#     speed=42,
#     video_exts=('.mp4', '.avi', '.mov', '.mkv')     
    
# ):
#     random.seed(speed)
#     soruce_dir = Path(soruce_dir)
#     target_dir = Path(target_dir)
    
#     train_dir = target_dir / 'train'
#     test_dir = target_dir / 'test'
    
#     train_dir.mkdir(parents=True, exist_ok=True)
#     test_dir.mkdir(parents=True, exist_ok=True)
    
#     for class_dir in soruce_dir.iterdir():
#         if not class_dir.is_dir():
#             continue    
        
#         video=[]
#         for video in class_dir.iterdir():
#             if video.suffix.lower() in video_exts:
#                 video.append(video)
                
#         random.shuffle(video)
#         split_idx = int(len(video) * train_ratio)
#         train_videos = video[:split_idx]
#         test_videos = video[split_idx:]
        
        
#         (train_dir / class_dir.name).mkdir(exist_ok=True)
#         (test_dir / class_dir.name).mkdir(exist_ok=True)
        
        
        
#         for video in train_videos:
#             shutil.copy(video,train_dir//class_dir.name/video.name)
            
            
#         for video in test_videos:
#             shutil.copy(video,test_dir//class_dir.name/video.name)
            
            
#             print(f"Class {class_dir.name}: {len(train_videos)} train videos, {len(test_videos)} test videos.")
        
        
#         print("Dataset split completed.")
                
#     if __name__ == "__main__":
#         split_dataset(
#         source_dir="datas/raw",
#         target_dir="datas/processed",
#         train_ratio=0.8,
#         seed=42
#     )

import os
import random
import shutil

# =========================
# 1. PATHS
# =========================
raw_data_folder = "datas/raw"
processed_data_folder = "datas/processed"

train_folder = os.path.join(processed_data_folder, "train")
test_folder = os.path.join(processed_data_folder, "test")

# =========================
# 2. SETTINGS
# =========================
train_percentage = 0.8
random_seed = 42
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

random.seed(random_seed)

# =========================
# 3. CREATE TRAIN & TEST FOLDERS
# =========================
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# =========================
# 4. LOOP OVER EACH CLASS
# =========================
class_names = os.listdir(raw_data_folder)

for class_name in class_names:
    class_path = os.path.join(raw_data_folder, class_name)

    # Skip if it's not a folder
    if not os.path.isdir(class_path):
        continue

    # =========================
    # 5. COLLECT VIDEOS
    # =========================
    videos = []

    for file_name in os.listdir(class_path):
        if file_name.lower().endswith(video_extensions):
            videos.append(file_name)

    # =========================
    # 6. SHUFFLE & SPLIT
    # =========================
    random.shuffle(videos)

    total_videos = len(videos)
    train_count = int(total_videos * train_percentage)

    train_videos = videos[:train_count]
    test_videos = videos[train_count:]

    # =========================
    # 7. CREATE CLASS FOLDERS
    # =========================
    train_class_folder = os.path.join(train_folder, class_name)
    test_class_folder = os.path.join(test_folder, class_name)

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(test_class_folder, exist_ok=True)

    # =========================
    # 8. COPY TRAIN VIDEOS
    # =========================
    for video_name in train_videos:
        src = os.path.join(class_path, video_name)
        dst = os.path.join(train_class_folder, video_name)
        shutil.copy(src, dst)

    # =========================
    # 9. COPY TEST VIDEOS
    # =========================
    for video_name in test_videos:
        src = os.path.join(class_path, video_name)
        dst = os.path.join(test_class_folder, video_name)
        shutil.copy(src, dst)

    # =========================
    # 10. PRINT RESULT
    # =========================
    print(
        f"{class_name}: {len(train_videos)} train | {len(test_videos)} test"
    )

print("✅ Dataset split completed!")
