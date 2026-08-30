import fitz  # PyMuPDF
from PIL import Image
import io

def extract_pdf_images(pdf_path):
    """
    Extracts all embedded visual images from a PDF file.
    Returns a list of PIL Image objects.
    """
    doc = fitz.open(pdf_path)
    extracted_images = []
    
    for page in doc:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Load into PIL Image for imagehash / OCR evaluation
            pil_img = Image.open(io.BytesIO(image_bytes))
            extracted_images.append(pil_img)
            
    return extracted_images

def extract_pdf_pages_as_images(pdf_path):
    """
    Fallback method: Renders entire PDF pages as images if no direct images are embedded.
    """
    doc = fitz.open(pdf_path)
    page_images = []
    
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(pil_img)
        
    return page_images