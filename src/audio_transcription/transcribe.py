import requests
from typing import Any, Dict


def audio_to_text(filename: str, api_key: str) -> Dict[str, Any]:
    """
    Sends a POST request to the Hugging Face API with the audio file and returns the response as a JSON object.
    
    Args:
        filename (str): The path to the audio file to be sent to the API.
        api_key (str): The API key for authorization.
        
    Returns:
        Dict[str, Any]: The JSON response from the API.
    """
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

if __name__ == '__main__':
    # Example usage
    output = audio_to_text("sample1.flac", "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(output)
