from google import genai
from google.genai import types
import os
from typing import cast
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class Image(BaseModel):
    image_type: str = Field(description="Type of the image, e.g., 'Picture', 'Drawing', 'Plot', etc.")
    image_name: str = Field(description="Name or title of the image.")
    image_description: str = Field(description="A brief description of the image content.")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

def caption_image(image_data: bytes):
    response = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=[
            types.Part.from_bytes(
            data=image_data,
            mime_type='image/jpeg',
            ),
            'Caption this image.'
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": Image,
        },
    )
    image: Image = cast(Image, response.parsed)
    return image

if __name__ == "__main__":

    with open('barplot_9_0.png', 'rb') as f:
        image_bytes = f.read()

    caption = caption_image(image_bytes)
    print("Image Type:", caption.image_type)
    print("Image Name:", caption.image_name)
    print("Image Description:", caption.image_description)
