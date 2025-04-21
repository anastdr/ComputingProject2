import torch  # type: ignore
import torch.nn as nn  # type: ignore # ✅ Needed for defining the final layer
import json
from torchvision import models # type: ignore

def load_class_mapping(json_path: str) -> dict:
    """Load the class index mapping from JSON and reverse it."""
    with open(json_path, "r") as f:
        class_to_idx = json.load(f)
    return {v: k for k, v in class_to_idx.items()}  # idx -> class_name

def load_model(model_path, num_classes):
    """Rebuild and load a ResNet18 model with trained weights."""
    model = models.resnet18(weights=None)  # or pretrained=False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # redefine final layer
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def get_prediction_info(model_output: torch.Tensor, idx_to_class: dict) -> dict:
    """Convert model output to class name, and extract plant and disease with readable format."""
    _, predicted_idx = torch.max(model_output, 1)
    class_name = idx_to_class[predicted_idx.item()]

    # Split class_name into plant and disease
    if "__" in class_name:
        plant, disease = class_name.split("__")
    else:
        plant, disease = class_name, "UNKNOWN"
    
    # Replace underscores with spaces and capitalize each word
    plant = plant.replace("_", " ").title()
    disease = disease.replace("_", " ").title()

    return {
        "index": predicted_idx.item(),
        "class_name": class_name,
        "plant": plant,
        "disease": disease
    }
