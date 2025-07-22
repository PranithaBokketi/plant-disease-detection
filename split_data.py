import os
import shutil
import random

def split_dataset(base_dir, train_dir, val_dir, split_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        
        # ✅ Skip if not a folder
        if not os.path.isdir(category_path):
            continue

        # ✅ Skip folders with no image files
        if not any(fname.lower().endswith(('.jpg', '.jpeg', '.png')) for fname in os.listdir(category_path)):
            print(f"Skipping non-image folder: {category_path}")
            continue

        images = [img for img in os.listdir(category_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        train_class_dir = os.path.join(train_dir, category)
        val_class_dir = os.path.join(val_dir, category)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_class_dir, img))

        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(val_class_dir, img))

    print("✅ Dataset split completed.")

# Set paths
base_dir = "C:/Users/pranitha/Desktop/ASSIGNMENTS/plant disease detection/PlantVillage"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

split_dataset(base_dir, train_dir, val_dir)

