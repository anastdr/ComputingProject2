from fastapi import FastAPI, Body, UploadFile, File, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from PIL import Image  # type: ignore
import torch  # type: ignore
import os
from fastapi import Body # type: ignore
import torch.nn as nn # type: ignore
from dotenv import load_dotenv  # type: ignore
from io import BytesIO
from pathlib import Path
from torchvision import transforms  # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List
from fastapi.requests import Request # type: ignore
from fastapi.responses import JSONResponse # type: ignore

from routes.chatbot.chat_prompts import load_class_mapping, get_prediction_info, generate_full_care_prompt # type: ignore
from routes.chatbot.chatbot import get_gemini_response # type: ignore
from routes.chatbot.model_to_chatbot import load_model  # type: ignore # use correct path if load_model is defined here


# Initialize FastAPI app
app = FastAPI()

origins = [
   "http://localhost:5173",  
   "https://computing-project2-ten.vercel.app",
   "https://computing-project2-9ako.vercel.app",
  "https://computing-project2-9ako-git-main-anastdrs-projects.vercel.app"    # local dev if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Define paths for the model and class mapping
MODEL_PATH = Path(__file__).resolve().parent / "routes" / "models" / "plant_disease_model_test.pth"
CLASS_MAP_PATH = Path(__file__).resolve().parent / "routes" / "models" / "class_index.json"

# Load class mapping (index to class) and determine number of classes
idx_to_class = load_class_mapping(CLASS_MAP_PATH)
num_classes = len(idx_to_class)

# Load the pre-trained model with the correct architecture
model = load_model(MODEL_PATH, num_classes=num_classes)

# Define image transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict-chatbot")
async def predict_chatbot(image: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        prediction = get_prediction_info(output, idx_to_class)
        plant, disease = prediction["plant"], prediction["disease"]

        prompt = generate_full_care_prompt(plant, disease)

        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not set in environment")

        gemini_response = get_gemini_response(prompt, api_key)

        return {
            "plant": plant,
            "disease": disease,
            "chatbot_message": gemini_response,
            "index": prediction["index"],
            "class_name": prediction["class_name"]
        }

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")


@app.options("/chat")
async def chat_options():
    # Handle CORS preflight request
    return JSONResponse(status_code=200, content={})


@app.post("/chat")
async def chat_post(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="Missing 'message' in request body")

        if not api_key:
            return {"error": "API key not set"}

        prompt = f"The user asked: {message}\nPlease provide a helpful response about plants."
        response = get_gemini_response(prompt, api_key)
        return {"reply": response}

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
@app.get("/")
def root():
    return {"message": "PlantMama backend is alive"}
    



