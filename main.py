from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from cv.extractor import extract_figures_from_pdf
from core.hash_engine import compute_phash, compute_dhash
from core.database import init_db, save_figure_hash, find_matches

app = FastAPI(
    title="GraphShield API",
    description="Advanced Computer Vision Backend for Graph and Image Plagiarism Detection."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def read_root():
    return {"status": "online", "message": "GraphShield Advanced Detection Engine Active"}

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """Receives PDF, extracts figures, computes hashes, and checks database for matches."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_folder = f"extracted_{file.filename.replace('.pdf', '')}"
    
    try:
        image_paths = extract_figures_from_pdf(temp_pdf_path, output_folder=output_folder)
        results = []
        
        for path in image_paths:
            phash_val = compute_phash(path)
            dhash_val = compute_dhash(path)
            
            # Check against previously stored papers in the database
            matches = find_matches(phash_val, dhash_val, threshold=12)
            
            # Index current figure into database
            save_figure_hash(file.filename, path, phash_val, dhash_val)
            
            results.append({
                "figure_path": path,
                "phash": phash_val,
                "dhash": dhash_val,
                "potential_matches": matches,
                "is_suspicious": len(matches) > 0
            })
            
        return {
            "filename": file.filename,
            "total_figures_analyzed": len(results),
            "figures": results
        }
        
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)