import fitz
import io
import re
import os
import pymupdf
import markdown
from PIL import Image
from src.image_captioning import caption_image
from llama_index.core import Document
from bs4 import BeautifulSoup

def get_images_description(page) -> str:
    image_list = page.get_images(full=True)
    descriptions = []

    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        caption = caption_image(image_bytes)
        if not caption:
            continue
        desc_str = (
            f"Image: {caption.image_name} "
            f"Type: {caption.image_type} "
            f"Description: {caption.image_description}"
        )
        descriptions.append(desc_str)

    if not descriptions:
        return ""

    result = '\n'.join(descriptions)
    return result

def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s\.]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_markdown(md_text: str) -> str:
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=' ')
    return text.strip()

def get_document_from_pdf(path_to_pdf: str) -> Document:
    doc = pymupdf.open(path_to_pdf)
    text = ' '.join([page.get_text() + ' ' +  get_images_description(page) for page in doc])
    text = clean_text(text)
    return Document(
        text=text,
        metadata={
            "file_path": path_to_pdf,
            "file_name": os.path.basename(path_to_pdf)
        }
    )

def get_document_from_txt(path_to_txt: str) -> Document:
    with open(path_to_txt, "r", encoding="utf-8") as f:
        text = f.read()
    text = clean_text(text)
    return Document(
        text=text,
        metadata={
            "file_path": path_to_txt,
            "file_name": os.path.basename(path_to_txt)
        }
    )

def get_document_from_md(path_to_md: str) -> Document:
    with open(path_to_md, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_markdown = clean_markdown(text)
    cleaned_text = clean_text(cleaned_markdown)
    return Document(
        text=cleaned_text,
        metadata={
            "file_path": path_to_md,
            "file_name": os.path.basename(path_to_md)
        }
    )
