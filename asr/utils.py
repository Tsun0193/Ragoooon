import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

def query(filename):
    assert os.path.exists(filename)
    with open(filename, "rb") as f:
        data = f.read()
    print("Sending request...")
    try:
        response = requests.post(API_URL, headers=headers, data=data)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    
    response = response.json()
    return response["text"].strip()

if __name__ == "__main__":
    pass
