
from torchvision import transforms  # type: ignore
from PIL import Image  # type: ignore
import os 

# === Core Image Size Standard ===
STANDARD_SIZE = (256, 256)

# === Basic Transforms ===
def resize_image(image, size=STANDARD_SIZE):
    return transforms.Resize(size)(image)

def grayscale_image(image):
    return transforms.Grayscale(num_output_channels=3)(image)

def horizontal_flip_image(image):
    return transforms.RandomHorizontalFlip(p=1.0)(image)

def vertical_flip_image(image):
    return transforms.RandomVerticalFlip(p=1.0)(image)

def rotate_image(image, degrees=30):
    return transforms.RandomRotation(degrees)(image)

def color_jitter_image(image):
    # Mild jitter
    return transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)

def heavy_color_jitter_image(image):
    # Strong lighting adjustment
    return transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)(image)

def random_crop_image(image, size=STANDARD_SIZE):
    image = transforms.Resize((300, 300))(image)
    return transforms.RandomCrop(size)(image)

def gaussian_blur_image(image):
    return transforms.GaussianBlur(kernel_size=5)(image)

def invert_image(image):
    return transforms.RandomInvert(p=1.0)(image)

def zoom_image(image):
    return transforms.RandomResizedCrop(STANDARD_SIZE, scale=(0.8, 1.0))(image)

def perspective_transform_image(image):
    return transforms.RandomPerspective(distortion_scale=0.4, p=1.0)(image)

def affine_transform_image(image):
    return transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)(image)

# === Preprocessing Entrypoint ===
def preprocess_image(image_path, save_path, transformations=[]):
    image = Image.open(image_path).convert('RGB')
    image = resize_image(image)

    for transform in transformations:
        image = transform(image)

    image.save(save_path)
