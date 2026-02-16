
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Listing available models and their methods...")

try:
    models = client.models.list()
    for m in models:
        print(f"Model: {m.name}")
    
    print("\nTesting Image Generation...")
    # Use standard imagen-3.0-generate-001
    response = client.models.generate_images(
        model='imagen-3.0-generate-001',
        prompt='A beautiful sunset over a robotic city',
    )
    if response.generated_images:
        print(f"Success! Generated {len(response.generated_images)} images.")
    else:
        print("Failed: No images returned.")
except Exception as e:
    print(f"Error: {e}")
