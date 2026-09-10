from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
import logging
import httpx
from bs4 import BeautifulSoup
from PIL import Image
import io

from app.models import ShodhgangaThesis
from app.services.fingerprint.clip_embedder import get_clip_embedding
from app.services.extraction.pdf_extractor import extract_figures_from_pdf
from app.core.config import get_settings
from app.core.storage import storage

settings = get_settings()
logger = logging.getLogger(__name__)

SHODHGANGA_BASE = 'https://shodhganga.inflibnet.ac.in'
SHODHGANGA_BROWSE = f'{SHODHGANGA_BASE}/handle/10603/'

async def sync_shodhganga(db: AsyncSession, limit: int = 100) -> int:
    '''Sync Shodhganga theses - fetch new theses and index their figures'''
    synced = 0

    try:
        # Get already indexed thesis IDs
        result = await db.execute(select(ShodhgangaThesis.thesis_id))
        indexed_ids = {row[0] for row in result}

        # Fetch thesis list (simplified - would need proper pagination)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f'{SHODHGANGA_BROWSE}?offset=0&limit={limit}')
            soup = BeautifulSoup(response.text, 'html.parser')

            # Parse thesis links (this is simplified - real implementation needs proper parsing)
            thesis_links = soup.find_all('a', href=True)
            for link in thesis_links[:limit]:
                href = link['href']
                if '/handle/10603/' in href and '?mode=full' in href:
                    thesis_id = href.split('/')[-2]
                    if thesis_id in indexed_ids:
                        continue

                    # Process thesis
                    await _process_thesis(db, client, thesis_id)
                    synced += 1

    except Exception as e:
        logger.error(f'Shodhganga sync failed: {e}')

    return synced


async def _process_thesis(db: AsyncSession, client: httpx.AsyncClient, thesis_id: str):
    '''Process a single thesis: download PDF, extract figures, generate embeddings'''
    try:
        # Get thesis metadata page
        meta_url = f'{SHODHGANGA_BASE}/handle/10603/{thesis_id}?mode=full'
        response = await client.get(meta_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract metadata (simplified)
        title = soup.find('td', string='Title:')
        title_text = title.find_next('td').get_text(strip=True) if title else 'Unknown'

        author = soup.find('td', string='Author:')
        author_text = author.find_next('td').get_text(strip=True) if author else 'Unknown'

        # Find PDF link
        pdf_link = None
        for a in soup.find_all('a', href=True):
            if a['href'].endswith('.pdf'):
                pdf_link = a['href']
                if not pdf_link.startswith('http'):
                    pdf_link = f'{SHODHGANGA_BASE}{pdf_link}'
                break

        if not pdf_link:
            return

        # Download PDF
        pdf_response = await client.get(pdf_link, timeout=60.0)
        pdf_bytes = pdf_response.content

        # Extract figures
        figures = extract_figures_from_pdf(pdf_bytes)

        # Generate embeddings
        embeddings = []
        for fig in figures:
            emb = get_clip_embedding(fig['image'])
            embeddings.append(emb.tolist())

        # Store in DB
        thesis = ShodhgangaThesis(
            thesis_id=thesis_id,
            title=title_text,
            author=author_text,
            year=2024,  # Parse from metadata
            university='Unknown',  # Parse from metadata
            figure_embeddings=embeddings,
            figure_count=len(figures),
            metadata={'pdf_url': pdf_link},
            indexed_at=datetime.utcnow(),
        )
        db.add(thesis)
        await db.commit()

        logger.info(f'Indexed Shodhganga thesis {thesis_id}: {len(figures)} figures')

    except Exception as e:
        logger.error(f'Failed to process thesis {thesis_id}: {e}')


async def index_org_repository(db: AsyncSession, org_id: int):
    '''Re-index all figures for an organization into vector search'''
    # This would be called by the vector search service to rebuild FAISS index
    from app.services.vector_search.hybrid_search import HybridVectorSearch
    search = HybridVectorSearch()
    search.mark_dirty()
    logger.info(f'Marked vector index dirty for org {org_id}')

from datetime import datetime
