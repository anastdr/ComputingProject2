from pathlib import Path
import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torchvision import datasets, transforms, models  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import json

# ==== Config ====
data_dir = Path(__file__).resolve().parent.parent / 'proces_data'  # Change to your processed data directory
batch_size = 32
epochs = 10
lr = 0.0004
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'plant_disease_model_test_2.pth'
class_index_path = 'class_index.json'

# ==== Data Transforms ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define valid extensions for images
valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

# === Load Data ===
def get_image_files(directory: Path):
    """
    Returns a list of valid image files in the given directory.
    Skips invalid files like `.DS_Store` and any hidden files.
    """
    image_files = []
    for file in directory.iterdir():
        if file.name.startswith('.'):  # Skip hidden files
            print(f"Skipping hidden file: {file}")
            continue
        
        if file.suffix.lower() in valid_extensions and file.name != '.DS_Store':
            image_files.append(file)
            print(f"Valid file: {file}")  # Print out valid image file paths
        elif file.name == '.DS_Store':
            print(f"Skipping invalid file: {file}")
    return image_files

# Directories for the dataset
train_dir = data_dir / 'train_images'
val_dir = data_dir / 'val_images'
test_dir = data_dir / 'test_images'

# Check folder structure for debugging
print("Training Directory Structure:")
for root, dirs, files in os.walk(train_dir):
    print(f"Root: {root}")
    for file in files:
        print(f"  - {file}")

print("\nValidation Directory Structure:")
for root, dirs, files in os.walk(val_dir):
    print(f"Root: {root}")
    for file in files:
        print(f"  - {file}")

print("\nTest Directory Structure:")
for root, dirs, files in os.walk(test_dir):
    print(f"Root: {root}")
    for file in files:
        print(f"  - {file}")

# Get image files
train_image_files = get_image_files(train_dir)
val_image_files = get_image_files(val_dir)
test_image_files = get_image_files(test_dir)

# Check if directories contain valid images
print(f"Found {len(train_image_files)} training images.")
print(f"Found {len(val_image_files)} validation images.")
print(f"Found {len(test_image_files)} test images.")

# Use ImageFolder to load the datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Save class index mapping
with open(class_index_path, 'w') as f:
    json.dump(train_dataset.class_to_idx, f)

# ==== Load Pretrained Model with Dropout ====
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the last layer to match the number of classes

# Add Dropout Layer (drop probability can be adjusted)
model.layer4[1].register_module('dropout', nn.Dropout(p=0.5))

model = model.to(device)

# ==== Loss and Optimizer with L2 Regularization (Weight Decay) ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization (weight decay)

# ==== Learning Rate Scheduler ====
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR by half every 5 epochs

# ==== Early Stopping Setup ====
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==== Training Loop ====
def train(model, train_loader, val_loader, epochs):
    early_stopping = EarlyStopping(patience=3, delta=0.01)  # Early stopping with patience of 3 epochs

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader)

        # Print stats for each epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        # Step the scheduler (this reduces the LR)
        scheduler.step()

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

# ==== Evaluation Function ====
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct / total, loss_total / len(loader)

# ==== Train ====
train(model, train_loader, val_loader, epochs)

# ==== Save Model ====
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

# ==== Test + Confusion Matrix ====
def test_and_confusion(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    print(f"🧪 Test Accuracy: {acc:.4f}")
    print("📊 Confusion Matrix:")
    print(cm)

    # Optional: plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ==== Final Test ====
test_and_confusion(model, test_loader)
