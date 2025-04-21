import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import models, datasets, transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import json
import os
from pathlib import Path
import seaborn as sns # type: ignore
from collections import Counter


# ==== Config ====
data_dir = Path(__file__).resolve().parent.parent / 'proces_data'
model_path = 'plant_disease_model.pth'
class_index_path = 'class_index.json'
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== Load Test Dataset ====
test_dir = os.path.join(data_dir, 'test_images')
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==== Load Class Index ====
with open(class_index_path) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ==== Load Model ====
model = models.resnet18(pretrained=False)
num_classes = len(class_to_idx)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ==== Run Inference + Collect Predictions ====
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ==== Confusion Matrix ====
cm = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)

print(f"🧪 Overall Test Accuracy: {acc:.4f}")

# ==== Per-class Accuracy ====
print("\n🎯 Per-Class Accuracy:")
for i, class_name in idx_to_class.items():
    class_total = (np.array(all_labels) == i).sum()
    class_correct = ((np.array(all_labels) == i) & (np.array(all_preds) == i)).sum()
    acc_class = class_correct / class_total if class_total > 0 else 0
    print(f"{class_name:30s}: {acc_class:.2%} ({class_correct}/{class_total})")


# ==== Plot Confusion Matrix ====
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=False) 