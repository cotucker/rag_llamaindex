import fitz
import io
from PIL import Image
from src.image_captioning import caption_image

file = "data/The Rust Programming Language.pdf"
pdf_file = fitz.open(file)

for page_index in range(len(pdf_file)):
    page = pdf_file.load_page(page_index)  # load the page
    image_list = page.get_images(full=True)  # get images on the page

    if image_list:
        print(f"[+] Found a total of {len(image_list)} images on page {page_index}")
    else:
        continue

    for image_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_name = f"image{page_index+1}_{image_index}.{image_ext}"
        print(f"[+] Image {image_name}")
        print(caption_image(image_bytes))
