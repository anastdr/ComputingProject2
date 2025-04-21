import os
from sklearn.model_selection import train_test_split  # type: ignore
from pathlib import Path
from preprocess import (
    preprocess_image,
    grayscale_image,
    horizontal_flip_image,
    vertical_flip_image,
    rotate_image,
    color_jitter_image,
    random_crop_image,
    gaussian_blur_image,
    invert_image,
    zoom_image,
    heavy_color_jitter_image,
    perspective_transform_image,
    affine_transform_image
)

# Define your augmentation pipeline
AUGMENTATIONS = {
    "original": [],
    "grayscale": [grayscale_image],
    "hflip": [horizontal_flip_image],
    "vflip": [vertical_flip_image],
    "rotate": [rotate_image],
    "color": [color_jitter_image],
    "crop": [random_crop_image],
    "gaussian_blur": [gaussian_blur_image],
    "invert": [invert_image],
    "zoom": [zoom_image],
    "heavy_color_jitter": [heavy_color_jitter_image],
    "perspective_transform": [perspective_transform_image],
    "affine_transform": [affine_transform_image]
}

def split_and_preprocess_data(dataset_dir, processed_data_dir, test_size=0.1, val_size=0.2):
    os.makedirs(processed_data_dir, exist_ok=True)

    # Create main folders for each split
    for split in ['train_images', 'val_images', 'test_images']:
        os.makedirs(os.path.join(processed_data_dir, split), exist_ok=True)

    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Extract species and label
        if '__' in class_name:
            species, label = class_name.split('__')
        else:
            species, label = class_name, 'UNKNOWN'

        # Get all image files (ignoring hidden files)
        image_files = [f for f in os.listdir(class_path) if not f.startswith('.')]
        n = len(image_files)

        # Initialize empty splits
        train_files, val_files, test_files = [], [], []

        # Split images into train/val/test based on the count available
        if n >= 3:
            train_files, test_files = train_test_split(image_files, test_size=test_size)
            train_files, val_files = train_test_split(train_files, test_size=val_size)
        elif n == 2:
            # Use one image for train and the other for test
            train_files = [image_files[0]]
            test_files = [image_files[1]]
        elif n == 1:
            # Only one image: use it for training
            train_files = image_files.copy()
        else:
            print(f"⚠️ Skipping class '{class_name}' (no images)")
            continue

        # Organize files by split
        split_map = {
            'train_images': train_files,
            'val_images': val_files,
            'test_images': test_files
        }

        for split_name, files in split_map.items():
            # Create a subdirectory for the class within the split folder
            class_folder = os.path.join(processed_data_dir, split_name, class_name)
            os.makedirs(class_folder, exist_ok=True)

            for f in files:
                image_path = os.path.join(class_path, f)

                # Apply augmentations only for the respective split
                for aug_name, transform_list in AUGMENTATIONS.items():
                    new_filename = f"{species}__{label}_{aug_name}_{f}"
                    save_path = os.path.join(class_folder, new_filename)

                    # Preprocess and save the image with the given transformations
                    preprocess_image(image_path, save_path, transformations=transform_list)
                    print(f"Saved: {save_path}")

    print("✅ All images processed, augmented, and split into train/val/test.")

if __name__ == "__main__":
    # Adjust the following paths as needed
    dataset_dir = Path(__file__).resolve().parent.parent / 'dataset'
    processed_data_dir = Path(__file__).resolve().parent.parent.parent / 'proces_data'
    split_and_preprocess_data(str(dataset_dir), str(processed_data_dir))



if __name__ == "__main__":
    # Adjust the following paths as needed
    dataset_dir = Path(__file__).resolve().parent.parent / 'dataset'
    processed_data_dir = Path(__file__).resolve().parent.parent.parent / 'proces_data'
    split_and_preprocess_data(str(dataset_dir), str(processed_data_dir))
