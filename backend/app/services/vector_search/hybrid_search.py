import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
import faiss
import logging

from app.models import Figure, Organization, ShodhgangaThesis
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class HybridVectorSearch:
    '''Hybrid vector search: FAISS (fast) + pgvector (persistent, filtered)'''

    def __init__(self):
        self.faiss_index = None
        self.faiss_figure_ids: List[int] = []
        self.index_dirty = True
        self.embedding_dim = settings.embedding_dim

    async def _rebuild_faiss_index(self, db: AsyncSession, org_id: int, scopes: List[str]):
        '''Rebuild FAISS index from pgvector'''
        try:
            # Collect embeddings based on scopes
            embeddings = []
            figure_ids = []

            # My org figures
            if 'my_org' in scopes:
                result = await db.execute(
                    select(Figure.id, Figure.clip_embedding)
                    .join(Figure.paper)
                    .where(Figure.clip_embedding.is_not(None), Figure.paper.has(org_id=org_id))
                )
                for fig_id, emb in result:
                    embeddings.append(np.array(emb, dtype=np.float32))
                    figure_ids.append(fig_id)

            # All opted-in orgs
            if 'all_opted_in' in scopes:
                result = await db.execute(
                    select(Figure.id, Figure.clip_embedding)
                    .join(Figure.paper)
                    .join(Figure.paper.of_type(Organization))
                    .where(Figure.clip_embedding.is_not(None), Organization.repo_opt_in == True, Organization.id != org_id)
                )
                for fig_id, emb in result:
                    embeddings.append(np.array(emb, dtype=np.float32))
                    figure_ids.append(fig_id)

            # Shodhganga
            if 'shodhganga' in scopes:
                result = await db.execute(
                    select(ShodhgangaThesis.id, ShodhgangaThesis.figure_embeddings)
                    .where(ShodhgangaThesis.figure_embeddings.is_not(None))
                )
                for thesis_id, emb_list in result:
                    if emb_list:
                        for i, emb in enumerate(emb_list):
                            embeddings.append(np.array(emb, dtype=np.float32))
                            figure_ids.append(-(thesis_id * 1000 + i))  # Negative IDs for Shodhganga

            if embeddings:
                embeddings_array = np.vstack(embeddings).astype(np.float32)
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_array)

                # Create IVF-Flat index
                nlist = min(settings.faiss_nlist, len(embeddings) // 10 + 1)
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.faiss_index.train(embeddings_array)
                self.faiss_index.add(embeddings_array)
                self.faiss_figure_ids = figure_ids
                self.index_dirty = False
                logger.info(f'FAISS index rebuilt: {len(embeddings)} vectors, nlist={nlist}')
            else:
                self.faiss_index = None
                self.faiss_figure_ids = []
                self.index_dirty = False

        except Exception as e:
            logger.error(f'Failed to rebuild FAISS index: {e}')
            self.faiss_index = None
            self.faiss_figure_ids = []

    async def search(
        self,
        db: AsyncSession,
        query_embedding: np.ndarray,
        org_id: int,
        scopes: List[str],
        top_k: int = 50,
        threshold: float = 0.70,
    ) -> List[Dict[str, Any]]:
        '''Search for similar figures using hybrid approach'''
        # Rebuild index if dirty
        if self.index_dirty or self.faiss_index is None:
            await self._rebuild_faiss_index(db, org_id, scopes)

        if self.faiss_index is None or len(self.faiss_figure_ids) == 0:
            return []

        # FAISS search (fast, returns top candidates)
        query_vec = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        k = min(top_k * 3, len(self.faiss_figure_ids))  # Get more candidates for filtering
        scores, indices = self.faiss_index.search(query_vec, k)

        # Collect candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                fig_id = self.faiss_figure_ids[idx]
                candidates.append((fig_id, float(score)))

        # Filter and enrich with pgvector for exact scores and metadata
        results = []
        for fig_id, faiss_score in candidates[:top_k]:
            if fig_id > 0:
                # Local figure
                result = await db.execute(
                    select(Figure, Figure.paper.of_type(Organization).name.label('org_name'))
                    .join(Figure.paper)
                    .join(Figure.paper.of_type(Organization))
                    .where(Figure.id == fig_id)
                )
                row = result.first()
                if row:
                    fig, org_name = row
                    results.append({
                        'figure_id': fig.id,
                        'paper_id': fig.paper_id,
                        'paper_title': fig.paper.title,
                        'similarity_score': round(faiss_score * 100, 1),
                        'match_type': 'semantic',
                        'page_num': fig.page_num,
                        'minio_path': fig.minio_path,
                        'source': f'{org_name} (Figure {fig.id})',
                        'details': {'faiss_score': faiss_score},
                    })
            else:
                # Shodhganga figure
                thesis_id = abs(fig_id) // 1000
                fig_idx = abs(fig_id) % 1000
                result = await db.execute(select(ShodhgangaThesis).where(ShodhgangaThesis.id == thesis_id))
                thesis = result.scalar_one_or_none()
                if thesis:
                    results.append({
                        'figure_id': fig_id,
                        'paper_id': -thesis_id,
                        'paper_title': thesis.title,
                        'similarity_score': round(faiss_score * 100, 1),
                        'match_type': 'shodhganga',
                        'page_num': fig_idx + 1,
                        'minio_path': None,
                        'source': f'Shodhganga: {thesis.title} ({thesis.university}, {thesis.year})',
                        'details': {'faiss_score': faiss_score, 'thesis_id': thesis.thesis_id},
                    })

        # Sort by similarity
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

    def mark_dirty(self):
        '''Mark index as needing rebuild'''
        self.index_dirty = True
