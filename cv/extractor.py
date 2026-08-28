import os
import fitz  # PyMuPDF

def extract_figures_from_pdf(pdf_path: str, output_folder: str = "extracted_figures") -> list[str]:
    """
    Extracts all images from a PDF file and saves them to a specified output folder.
    Returns a list of file paths for all extracted images.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    saved_image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Standardized filename structure: pageX_imgY.ext
            img_name = f"page{page_index + 1}_img{img_index}.{image_ext}"
            img_path = os.path.join(output_folder, img_name)
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            saved_image_paths.append(img_path)

    doc.close()
    return saved_image_paths

if __name__ == "__main__":
    print("PDF Extractor Engine initialized successfully.")