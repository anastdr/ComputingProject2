import os
import json
from pathlib import Path

# Base path to processed_data
dataset_dir = Path(__file__).resolve().parent.parent.parent / 'proces_data'

# Subdirectories that contain images
image_folders = ['train_images', 'val_images', 'test_images']

# Initialize dictionary to store separate label maps for each folder
label_maps = {
    'train_images': {},
    'val_images': {},
    'test_images': {}
}

# Initialize current label counts for each folder
current_labels = {
    'train_images': 0,
    'val_images': 0,
    'test_images': 0
}

# Iterate over both train_images, val_images, and test_images directories
for folder_name in image_folders:
    folder_path = dataset_dir / folder_name
    if not folder_path.exists():
        print(f" Folder not found: {folder_path}")
        continue

    print(f"\n Processing folder: {folder_path}")

    for class_disease_folder in os.listdir(folder_path):
        class_disease_folder_path = folder_path / class_disease_folder

        # Skip if it's a file, hidden file, or unsupported format
        if class_disease_folder_path.is_file() or class_disease_folder.startswith('.'):
            continue

        # Check naming convention: CLASS__DISEASE
        if '__' not in class_disease_folder:
            print(f"⚠️ Skipping malformed folder: {class_disease_folder}")
            continue

        # Extract class and disease labels
        class_label, disease_label = class_disease_folder.split('__', 1)
        combined_label = f"{class_label}__{disease_label}"

        # Check if the label map for the current folder already has the class label
        if combined_label not in label_maps[folder_name]:
            label_maps[folder_name][combined_label] = current_labels[folder_name]
            current_labels[folder_name] += 1

        print(f"{combined_label} added to {folder_name} label map with label {label_maps[folder_name][combined_label]}")

# Save the label maps for each folder inside processed_data
for folder_name in image_folders:
    label_map_path = dataset_dir / f'{folder_name}_label_map.json'
    with open(label_map_path, 'w') as f:
        json.dump(label_maps[folder_name], f, indent=4)
    print(f" {folder_name}_label_map.json saved!")

print("\n All done organizing images and label maps saved!")
