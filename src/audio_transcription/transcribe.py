import requests
from typing import Any, Dict
import os
import dotenv
import aiohttp
import asyncio
from gradio_client import Client

dotenv.load_dotenv()

# def audio_to_text(filename: str, api_key: str) -> Dict[str, Any]:
#     """
#     Sends a POST request to the Hugging Face API with the audio file and returns the response as a JSON object.
    
#     Args:
#         filename (str): The path to the audio file to be sent to the API.
#         api_key (str): The API key for authorization.
        
#     Returns:
#         Dict[str, Any]: The JSON response from the API.
#     """
#     API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
#     headers = {"Authorization": f"Bearer {api_key}"}
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()

# async def audio_to_text(text: str):
#     API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
#     headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    
#     async with aiohttp.ClientSession() as session:
#         while True:
#             async with session.post(API_URL, headers=headers, data=text) as response:
#                 result = await response.json()
#                 if 'error' not in result or result.get('error') != 'Model masakhane/m2m100_418M_fon_fr_rel_news is currently loading':
#                     # break  # Exit loop if response is not in loading state or does not contain error
#                     await asyncio.sleep(5)  # Wait for 5 seconds before checking again

#         return result
    
    
import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
# headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# output = query("Alina_Eng.wav")

# if __name__ == '__main__':
#     # Example usage
#     HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#     output = audio_to_text("assets/Pitch-tonic-sample-voice.m4a")
#     print(output)

# asyncio.run(audio_to_text("Alina_Eng.wav"))


def audio_to_text(audio_file_location:str):

    client = Client("https://openai-whisper.hf.space/")
    result = client.predict(
                    audio_file_location,	# str (filepath or URL to file) in 'inputs' Audio component
                    "transcribe",	# str in 'Task' Radio component
                    api_name="/predict"
                )
    return result

if __name__ == '__main__':
    audio_file_location = "./assets/Pitch-tonic-sample-voice.m4a"
    sample = audio_to_text(audio_file_location)

    x=0