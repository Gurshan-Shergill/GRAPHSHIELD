import fitz  # PyMuPDF
import os
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Normalizes images to grayscale and standard 256x256 dimensions."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L")
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    return img

def extract_figures_from_pdf(pdf_path: str, output_folder: str = "extracted_figures") -> list[str]:
    """Extracts embedded figures from PDF pages and normalizes them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    doc = fitz.open(pdf_path)
    extracted_paths = []
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            try:
                processed_img = preprocess_image(image_bytes)
                output_filename = f"page{page_index + 1}_img{img_index + 1}.png"
                output_path = os.path.join(output_folder, output_filename)
                processed_img.save(output_path, format="PNG")
                extracted_paths.append(output_path)
            except Exception:
                continue
                
    doc.close()
    return extracted_paths