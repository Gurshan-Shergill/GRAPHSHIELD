import fitz  # PyMuPDF
from difflib import SequenceMatcher

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all raw text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + " "
    doc.close()
    return full_text.strip()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculates percentage match between two text documents."""
    if not text1 or not text2:
        return 0.0
    
    # Calculate similarity ratio using SequenceMatcher
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return round(ratio * 100.0, 2)