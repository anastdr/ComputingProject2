import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv # type: ignore
import os

def configure_gemini(api_key: str):
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)

def get_gemini_response(prompt: str, api_key: str) -> str:
    """Send prompt to Gemini and get response text."""
    configure_gemini(api_key)
    
    # Initialize the model with the correct model name
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content using the prompt
    response = model.generate_content(prompt)
    
    # Debugging: Check the type of the response
    print(f"Response type: {type(response)}")
    
    # Ensure that the response is a string
    if isinstance(response, str):
        return response
    else:
        # If it's a non-string object, convert it to string
        return str(response)

