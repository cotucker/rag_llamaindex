import fitz
import io
from PIL import Image
from src.image_captioning import caption_image

file = "data/The Rust Programming Language.pdf"
pdf_file = fitz.open(file)

for page_index in range(len(pdf_file)):
    page = pdf_file.load_page(page_index)
    image_list = page.get_images(full=True)

    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_name = f"image{page_index+1}_{image_index}.{image_ext}"

def get_images_description(page) -> str:
    image_list = page.get_images(full=True)
    descriptions = []

    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        caption = caption_image(image_bytes)
        descriptions.append({
            "image_index": image_index,
            "image_name": f"image{page.number+1}_{image_index}.{base_image['ext']}",
            "image_title": caption.image_name,
            "image_type": caption.image_type,
            "image_description": caption.image_description
        })

    return str(descriptions)
