
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment.")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing available models...")

try:
    print("Models:")
    for m in client.models.list():
        # print details to stdout
        print(f"Name: {getattr(m, 'name', 'Unknown')}")
    
    print("\n-------------------")
    print("Test generation with gemini-2.0-flash:")
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents="Hello")
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n-------------------")
    print("Test generation with gemini-1.5-flash:")
    try:
        response = client.models.generate_content(model='gemini-1.5-flash', contents="Hello")
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

except Exception as e:
    print(f"General Error: {e}")
