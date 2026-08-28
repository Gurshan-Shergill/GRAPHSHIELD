from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from cv.extractor import extract_figures_from_pdf
from core.hash_engine import compute_phash

app = FastAPI(
    title="GraphShield API",
    description="Backend service for PDF figure extraction and image plagiarism analysis."
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "GraphShield API is running"}

@app.post("/extract")
async def extract_pdf_figures(file: UploadFile = File(...)):
    """
    Upload a PDF file to extract all embedded figures and calculate their pHashes.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save the uploaded PDF temporarily
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_folder = f"extracted_{file.filename.replace('.pdf', '')}"
    
    try:
        # Extract figures using your CV module
        image_paths = extract_figures_from_pdf(temp_pdf_path, output_folder=output_folder)
        
        # Calculate pHash for each extracted figure
        figures_data = []
        for path in image_paths:
            phash_val = compute_phash(path)
            figures_data.append({
                "file_path": path,
                "phash": phash_val
            })
            
        return {
            "filename": file.filename,
            "total_figures_extracted": len(figures_data),
            "figures": figures_data
        }
        
    finally:
        # Clean up temporary PDF file after processing
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    