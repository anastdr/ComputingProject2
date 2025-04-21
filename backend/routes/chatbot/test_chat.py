import os
import torch # type: ignore
from dotenv import load_dotenv # type: ignore
from chatbot import get_gemini_response  # Your Gemini API utility
from chat_prompts import load_class_mapping, get_prediction_info, generate_full_care_prompt
from torchvision import transforms, models # type: ignore
import torch.nn as nn # type: ignore
from PIL import Image # type: ignore
from pathlib import Path

# Load environment variables
load_dotenv()

def load_model(model_path, num_classes):
    """Rebuild and load a ResNet18 model with trained weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def run_test():
    # Load class mapping
    class_mapping_path = Path(__file__).resolve().parent.parent.parent.parent / 'training' / 'class_index.json'
    idx_to_class = load_class_mapping(class_mapping_path)

    # Load model
    model_path = Path(__file__).resolve().parent.parent.parent / 'routes' /'models'/ 'plant_disease_model_test.pth'
    num_classes = len(idx_to_class)
    model = load_model(model_path, num_classes)

    # Load and preprocess the image
    image_path = Path.home() / "Downloads" / "tomatoes.jpg"
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Batch size 1

    # Run on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    # Run inference
    with torch.no_grad():
        model_output = model(input_batch)

    # Get class and prediction info
    prediction_info = get_prediction_info(model_output, idx_to_class)

    print(f"\n🔎 Predicted Class Index: {prediction_info['index']}")
    print(f"🔎 Predicted Class Name: {prediction_info['class_name']}")
    print(f"🔎 Plant: {prediction_info['plant']}")
    print(f"🔎 Disease: {prediction_info['disease']}\n")
    plant = prediction_info['plant']
    disease = prediction_info['disease']

    # Build prompt and get response
    prompt = generate_full_care_prompt(plant, disease)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("API key not found. Make sure to set GEMINI_API_KEY in your .env file.")
        return

    response = get_gemini_response(prompt, api_key)
    print("🪴 PlantMama's response:\n")
    print(response)

if __name__ == "__main__":
    run_test()
