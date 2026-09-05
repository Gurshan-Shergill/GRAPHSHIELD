import fitz
import hashlib
from typing import List, Dict, Any
from difflib import SequenceMatcher
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


_model = None


def get_embedding_model():
    global _model
    if _model is None and EMBEDDINGS_AVAILABLE:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + " "
    doc.close()
    return full_text.strip()


def extract_text_blocks(pdf_path: str, min_length: int = 50) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    blocks = []
    for page_idx, page in enumerate(doc, 1):
        text = page.get_text()
        for block in text.split("\n\n"):
            block = block.strip()
            if len(block) >= min_length:
                blocks.append({
                    "text": block,
                    "page": page_idx,
                    "hash": hashlib.md5(block.encode()).hexdigest()[:12]
                })
    doc.close()
    return blocks


def calculate_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return round(ratio * 100.0, 2)


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    model = get_embedding_model()
    if model is None:
        return calculate_text_similarity(text1, text2)
    
    try:
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return round(max(0.0, cos_sim) * 100.0, 2)
    except Exception:
        return calculate_text_similarity(text1, text2)


def find_text_matches(
    query_blocks: List[Dict[str, Any]], 
    reference_blocks: List[Dict[str, Any]], 
    threshold: float = 75.0,
    use_semantic: bool = True
) -> List[Dict[str, Any]]:
    matches = []
    for q_block in query_blocks:
        best_match = None
        best_score = 0.0
        
        for ref_block in reference_blocks:
            if use_semantic:
                score = calculate_semantic_similarity(q_block["text"], ref_block["text"])
            else:
                score = calculate_text_similarity(q_block["text"], ref_block["text"])
            
            if score > best_score:
                best_score = score
                best_match = ref_block
        
        if best_match and best_score >= threshold:
            matches.append({
                "query_text": q_block["text"][:200] + "..." if len(q_block["text"]) > 200 else q_block["text"],
                "query_page": q_block["page"],
                "query_hash": q_block["hash"],
                "matched_text": best_match["text"][:200] + "..." if len(best_match["text"]) > 200 else best_match["text"],
                "matched_page": best_match["page"],
                "matched_hash": best_match["hash"],
                "similarity_score": best_score,
                "match_type": "semantic" if use_semantic else "lexical"
            })
    
    return matches