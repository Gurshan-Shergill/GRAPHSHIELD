import io
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector
import numpy as np

from app.core.config import get_settings
from app.core.security import get_current_user_org_id, verify_api_key_dependency
from app.core.database import get_session
from app.models import Figure, Organization, ShodhgangaThesis
from app.schemas.search import FigureSearchRequest, FigureSearchResponse, FigureMatchResult
from app.services.fingerprint.clip_embedder import get_clip_embedding
from app.services.vector_search.hybrid_search import HybridVectorSearch

settings = get_settings()
router = APIRouter(prefix='/search', tags=['Search'])
vector_search = HybridVectorSearch()


@router.post('/figures', response_model=FigureSearchResponse)
async def search_figures(
    file: UploadFile = File(..., description='Image file to search'),
    scopes: str = Form('[ my_org, all_opted_in, shodhganga]'),
    top_k: int = Form(10),
    threshold: float = Form(0.70),
    org_id: int = Depends(get_current_user_org_id),
    api_key: str = Depends(verify_api_key_dependency),
    db: AsyncSession = Depends(get_session),
):
    '''Search for similar figures by uploading an image'''
    import json
    scope_list = json.loads(scopes) if isinstance(scopes, str) else scopes

    # Read and validate image
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Empty file')

    # Generate CLIP embedding
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(content)).convert('RGB')
        query_embedding = get_clip_embedding(image)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid image: {str(e)}')

    # Search
    import time
    start = time.time()
    matches = await vector_search.search(
        db=db,
        query_embedding=query_embedding,
        org_id=org_id,
        scopes=scope_list,
        top_k=top_k,
        threshold=threshold,
    )
    search_time = (time.time() - start) * 1000

    return FigureSearchResponse(
        matches=matches,
        total_matches=len(matches),
        search_time_ms=search_time,
    )


@router.post('/figures/{figure_id}/similar', response_model=FigureSearchResponse)
async def find_similar_to_figure(
    figure_id: int,
    scopes: str = Form('[my_org, all_opted_in, shodhganga]'),
    top_k: int = Form(10),
    threshold: float = Form(0.70),
    org_id: int = Depends(get_current_user_org_id),
    api_key: str = Depends(verify_api_key_dependency),
    db: AsyncSession = Depends(get_session),
):
    '''Find figures similar to an already indexed figure'''
    import json
    scope_list = json.loads(scopes) if isinstance(scopes, str) else scopes

    result = await db.execute(select(Figure).where(Figure.id == figure_id, Figure.paper.has(org_id=org_id)))
    figure = result.scalar_one_or_none()
    if not figure or figure.clip_embedding is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Figure not found or not indexed')

    import time
    start = time.time()
    query_embedding = np.array(figure.clip_embedding, dtype=np.float32)
    matches = await vector_search.search(
        db=db,
        query_embedding=query_embedding,
        org_id=org_id,
        scopes=scope_list,
        top_k=top_k,
        threshold=threshold,
    )
    search_time = (time.time() - start) * 1000

    # Filter out self
    matches = [m for m in matches if m.figure_id != figure_id]

    return FigureSearchResponse(
        query_figure_id=figure_id,
        matches=matches,
        total_matches=len(matches),
        search_time_ms=search_time,
    )
