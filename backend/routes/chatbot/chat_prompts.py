import json
import torch  # type: ignore

def load_class_mapping(json_path: str) -> dict:
    """Load class-to-index JSON and reverse it to index-to-class."""
    with open(json_path, "r") as f:
        class_to_idx = json.load(f)
    return {v: k for k, v in class_to_idx.items()}

def get_prediction_info(model_output: torch.Tensor, idx_to_class: dict) -> dict:
    """Convert model output to predicted class and split plant/disease."""
    _, predicted_idx = torch.max(model_output, 1)
    class_name = idx_to_class[predicted_idx.item()]

    if "__" in class_name:
        plant, disease = class_name.split("__")
    else:
        plant, disease = class_name, "UNKNOWN"

    return {
        "index": predicted_idx.item(),
        "class_name": class_name,
        "plant": plant.replace("_", " ").title(),
        "disease": disease.replace("_", " ").title()
    }

def generate_full_care_prompt(plant: str, disease: str) -> str:
    """Create prompt that requests plant/disease care suggestions from Gemini."""
    return (
        f"The user uploaded an image of a plant. Based on the prediction, "
        f"the plant is '{plant}' and the disease is '{disease}'. "
        f"Please generate a helpful response that includes:\n"
        f"- Confirmation of the plant and disease\n"
        f"- A brief description of the plant\n"
        f"- Information about the disease and its symptoms\n"
        f"- Step-by-step treatment or cure suggestions\n"
        f"- General care tips to keep this plant healthy\n"
        f"The tone should be friendly, supportive, and helpful."
    )
