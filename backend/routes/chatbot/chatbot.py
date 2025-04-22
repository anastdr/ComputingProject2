import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv # type: ignore
import os

def configure_gemini(api_key: str):
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)

def get_gemini_response(prompt: str, api_key: str) -> str:
    """Send prompt to Gemini and return just the text response."""
    configure_gemini(api_key)
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    try:
        return response.candidates[0].content.parts[0].text
    except (AttributeError, IndexError):
        return "❌ Sorry, I couldn't generate a response. Please try again!"


