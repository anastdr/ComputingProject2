import os
from pathlib import Path
import shutil

# Base path to processed_data
dataset_dir = Path(__file__).resolve().parent.parent.parent / 'processed_data_copy'

# Subdirectories that contain images
image_folders = ['train_images', 'val_images', 'test_images']

for folder_name in image_folders:
    folder_path = dataset_dir / folder_name
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        continue

    print(f"\n📁 Processing folder: {folder_path}")

    for image_file in os.listdir(folder_path):
        file_path = folder_path / image_file

        # Skip if it's a directory, hidden file, or unsupported format
        if file_path.is_dir() or image_file.startswith('.') or not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Check naming convention: CLASS__DISEASE__XYZ.jpg
        parts = image_file.split('__')
        if len(parts) < 2:
            print(f"⚠️ Skipping malformed file: {image_file}")
            continue

        class_label = parts[0]
        disease_label = parts[1].split('_')[0]  # Only take the first part before any _ if needed

        # Only move files with all-uppercase plant names
        if not class_label.isupper():
            print(f"⛔ Skipping lowercase class name: {image_file}")
            continue

        # Construct folder name and path
        class_disease = f"{class_label}__{disease_label}"
        target_folder = folder_path / class_disease  # stay inside the same subfolder (train/val/test)

        target_folder.mkdir(exist_ok=True)

        # Destination file path
        destination = target_folder / image_file

        try:
            shutil.move(str(file_path), str(destination))
            print(f"✅ Moved {image_file} → {target_folder.name}/")
        except Exception as e:
            print(f"❌ Failed to move {image_file}: {e}")

print("\n🎉 All done organizing images!")
