from google import genai
from google.genai import types
from groq import Groq
import base64
import os
import os
import json
from typing import cast
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class Image(BaseModel):
    image_type: str = Field(description="Type of the image, e.g., 'Picture', 'Drawing', 'Plot', etc.")
    image_name: str = Field(description="Name or title of the image.")
    image_description: str = Field(description="A brief description of the image content.")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def caption_image(image_data: bytes):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[
                types.Part.from_bytes(
                data=image_data,
                mime_type='image/jpeg',
                ),
                'Caption this image'
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Image,
            },
        )
        image: Image = cast(Image, response.parsed)
    except Exception as e:
        print(f"🚨 Error captioning image.")
        return None

    return image

def caption_image_groq(image_data: bytes):
    base64_image = base64.b64encode(image_data).decode('utf-8')
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Caption this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "product_review",
                "schema": Image.model_json_schema()
            }

        }
    )
    result = chat_completion.choices[0].message.content
    image: Image = Image(**json.loads(result))
    return image

if __name__ == "__main__":
    with open("data/newplot.png", "rb") as image_file:
        a = caption_image_groq(image_file.read())
        print(a)
