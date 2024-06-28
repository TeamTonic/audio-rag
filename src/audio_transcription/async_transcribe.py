import os
import aiohttp
from typing import Any, Dict
import asyncio

async def async_transcibe(filename: str) -> Dict[str, Any]:
    """
    Sends a POST request to the Hugging Face API with the audio file and returns the response as a JSON object.
    
    Args:
        filename (str): The path to the audio file to be sent to the API.
        
    Returns:
        Dict[str, Any]: The JSON response from the API.
    """
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    async with aiohttp.ClientSession() as session:
        with open(filename, "rb") as f:
            data = f.read()
        async with session.post(API_URL, headers=headers, data=data) as response:
            return await response.json()

async def main():
    data = await async_transcibe("/home/isayahc/projects/FanTonic/Alina_Eng.wav")
    x=0

if __name__ == '__main__':
    asyncio.run(main())