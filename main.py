from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import requests
import fitz  # PyMuPDF

from cv.extractor import extract_figures_from_pdf
from core.hash_engine import compute_phash, compute_dhash
from core.database import init_db, save_figure_hash, find_matches

app = FastAPI(
    title="GraphShield API",
    description="Advanced Plagiarism Analytics, Internet Lookup, and Clean PDF Generation Engine."
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

def query_internet_source(query_text: str):
    """
    Queries Crossref API to find original academic sources matching paper titles/topics.
    """
    try:
        url = f"https://api.crossref.org/works?query={query_text}&rows=1"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                item = items[0]
                return {
                    "source_title": item.get("title", ["Unknown Title"])[0],
                    "doi": item.get("DOI", "N/A"),
                    "url": item.get("URL", "N/A"),
                    "publisher": item.get("publisher", "Unknown")
                }
    except Exception:
        pass
    return None

def create_deplagiarized_pdf(input_pdf_path: str, output_pdf_path: str):
    """
    Creates a cleaned copy of the document by cleaning contents of pages containing figures.
    """
    doc = fitz.open(input_pdf_path)
    for page in doc:
        image_list = page.get_images(full=True)
        if len(image_list) > 0:
            page.clean_contents()
            
    doc.save(output_pdf_path)
    doc.close()
    return output_pdf_path

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_pdf_path = f"temp_{file.filename}"
    clean_pdf_name = f"clean_{file.filename}"
    
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_folder = f"extracted_{file.filename.replace('.pdf', '')}"
    
    try:
        # Step 1: Extract normalized figures
        image_paths = extract_figures_from_pdf(temp_pdf_path, output_folder=output_folder)
        results = []
        suspicious_list = []
        
        # Step 2: Query Internet Metadata via Crossref
        paper_topic = file.filename.replace(".pdf", "").replace("_", " ")
        internet_source = query_internet_source(paper_topic)
        
        for path in image_paths:
            phash_val = compute_phash(path)
            dhash_val = compute_dhash(path)
            
            # Local Database Search
            matches = find_matches(phash_val, dhash_val, threshold=12)
            
            # Calculate detailed error percentages for each match
            for m in matches:
                m["error_percentage"] = round(100.0 - m["similarity_score"], 2)
            
            is_suspicious = len(matches) > 0
            if is_suspicious:
                suspicious_list.append(path)
            
            # Index current figure into local database
            save_figure_hash(file.filename, path, phash_val, dhash_val)
            
            results.append({
                "figure_path": path,
                "phash": phash_val,
                "dhash": dhash_val,
                "error_percentage": matches[0]["error_percentage"] if matches else 0.0,
                "potential_matches": matches,
                "is_suspicious": is_suspicious
            })
            
        # Step 3: Generate Clean (De-plagiarized) PDF version
        create_deplagiarized_pdf(temp_pdf_path, clean_pdf_name)
        
        # Calculate overall document plagiarism percentage
        total_figs = len(results)
        flagged_figs = len(suspicious_list)
        doc_plagiarism_rate = round((flagged_figs / total_figs * 100.0), 2) if total_figs > 0 else 0.0

        return {
            "filename": file.filename,
            "total_figures_analyzed": total_figs,
            "overall_plagiarism_percentage": doc_plagiarism_rate,
            "internet_source_attribution": internet_source,
            "clean_pdf_download_url": f"/download-clean/{clean_pdf_name}",
            "figures": results
        }
        
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

@app.get("/download-clean/{pdf_name}")
def download_clean_pdf(pdf_name: str):
    """Endpoint to download the newly generated de-plagiarized PDF file."""
    if os.path.exists(pdf_name):
        return FileResponse(pdf_name, media_type="application/pdf", filename=pdf_name)
    raise HTTPException(status_code=404, detail="Clean file not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)