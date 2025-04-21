import os
from pathlib import Path

def check_invalid_extensions(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        # Loop through all the files in the directory
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()  # Get file extension and convert to lowercase
                
            # If the file extension is invalid, print the file path
            if file_extension not in valid_extensions:
                print(f"Invalid file: {os.path.join(root, file)} (Extension: {file_extension})")

# Replace with your actual paths
check_invalid_extensions(Path(__file__).resolve().parent.parent/ 'proces_data'/'train_images')
check_invalid_extensions(Path(__file__).resolve().parent.parent/ 'proces_data'/'test_images')
check_invalid_extensions(Path(__file__).resolve().parent.parent / 'proces_data'/'val_images')