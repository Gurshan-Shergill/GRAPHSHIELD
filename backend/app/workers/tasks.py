import json
import logging
from datetime import datetime
from celery import shared_task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.core.storage import storage
from app.models import Paper, PaperStatus, Job, JobStatus, JobStep, Figure, SimilarityMatch, MatchType, Report, ReportStatus, Organization, ShodhgangaThesis

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, name='app.workers.tasks.process_paper_job')
def process_paper_job(self, job_id: str, paper_id: int, pdf_bytes: bytes, options: dict):
    """Main async task to process a paper: extract, fingerprint, search, analyze, report"""
    import asyncio
    asyncio.run(_process_paper_job_async(job_id, paper_id, pdf_bytes, options))


async def _process_paper_job_async(job_id: str, paper_id: int, pdf_bytes: bytes, options: dict):
    async with async_session_factory() as db:
        try:
            # Update job status
            job = await db.get(Job, int(job_id))
            paper = await db.get(Paper, paper_id)
            if not job or not paper:
                logger.error(f"Job {job_id} or Paper {paper_id} not found")
                return

            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            paper.status = PaperStatus.PROCESSING
            await db.commit()

            # Step 1: Extract figures
            job.current_step = JobStep.EXTRACTION
            job.progress = 10
            await db.commit()

            from app.services.extraction.pdf_extractor import extract_figures_from_pdf, extract_pages_as_images
            figures = extract_figures_from_pdf(pdf_bytes)
            if not figures:
                figures = extract_pages_as_images(pdf_bytes)

            paper.total_figures = len(figures)
            await db.commit()

            # Step 2: Fingerprinting (hashes + CLIP embeddings)
            job.current_step = JobStep.FINGERPRINTING
            job.progress = 25
            await db.commit()

            from app.services.fingerprint.hasher import compute_perceptual_hashes
            from app.services.fingerprint.clip_embedder import get_clip_embedding

            figure_records = []
            for idx, fig_data in enumerate(figures):
                img = fig_data['image']
                hashes = compute_perceptual_hashes(img)
                clip_emb = get_clip_embedding(img).tolist()

                # Save figure image to MinIO
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                minio_path = f'orgs/{paper.org_id}/papers/{paper.id}/figures/fig_{idx+1}.png'
                storage.upload_bytes(minio_path, img_bytes.read(), 'image/png')

                figure = Figure(
                    paper_id=paper.id,
                    page_num=fig_data['page_num'],
                    bbox=fig_data['bbox'],
                    minio_path=minio_path,
                    phash=hashes['phash'],
                    dhash=hashes['dhash'],
                    clip_embedding=clip_emb,
                    figure_type=fig_data.get('caption', '')[:100],
                )
                db.add(figure)
                figure_records.append((figure, fig_data, hashes, clip_emb))

            await db.commit()

            # Step 3: Vector search
            job.current_step = JobStep.VECTOR_SEARCH
            job.progress = 40
            await db.commit()

            from app.services.vector_search.hybrid_search import HybridVectorSearch
            vector_search = HybridVectorSearch()

            all_matches = []
            for figure, fig_data, hashes, clip_emb in figure_records:
                matches = await vector_search.search(
                    db=db,
                    query_embedding=figure.clip_embedding,
                    org_id=paper.org_id,
                    scopes=['my_org', 'all_opted_in', 'shodhganga'],
                    top_k=10,
                    threshold=0.70,
                )

                # Save matches
                for match in matches:
                    match_record = SimilarityMatch(
                        query_figure_id=figure.id,
                        matched_figure_id=match['figure_id'] if match['figure_id'] > 0 else None,
                        similarity_score=match['similarity_score'],
                        match_type=MatchType(match['match_type']) if match['match_type'] in [m.value for m in MatchType] else MatchType.SEMANTIC,
                        details=match['details'],
                    )
                    db.add(match_record)
                    all_matches.append(match)

            await db.commit()

            # Step 4: Manipulation detection
            job.current_step = JobStep.MANIPULATION_DETECTION
            job.progress = 55
            await db.commit()

            from app.services.manipulation.detector import detect_manipulation, compute_manipulation_score
            all_images = [f['image'] for f in figures]
            manipulation_findings = []

            for idx, (figure, fig_data, hashes, clip_emb) in enumerate(figure_records):
                findings = detect_manipulation(fig_data['image'], all_images, idx)
                manipulation_score = compute_manipulation_score(findings)
                figure.manipulation_score = manipulation_score
                manipulation_findings.extend(findings)

                # Save internal duplicates
                for finding in findings:
                    if finding['type'] in ['duplicate_figure', 'highly_similar_figure', 'inverted_reuse', 'flipped_reuse', 'rotated_reuse']:
                        match_record = SimilarityMatch(
                            query_figure_id=figure.id,
                            matched_figure_id=finding['evidence'].get('matched_figure'),
                            similarity_score=100 - finding['evidence'].get('avg_distance', 0) * 100 / 64,
                            match_type=MatchType.INTERNAL_DUPLICATE,
                            details=finding,
                        )
                        db.add(match_record)

            await db.commit()

            # Step 5: Graph semantics
            job.current_step = JobStep.GRAPH_SEMANTICS
            job.progress = 65
            await db.commit()

            from app.services.graph_semantics.ocr_parser import extract_graph_data
            graph_findings = []
            for idx, (figure, fig_data, hashes, clip_emb) in enumerate(figure_records):
                graph_data = extract_graph_data(fig_data['image'])
                figure.ocr_data = graph_data
                if graph_data.get('chart_type') != 'unknown':
                    graph_findings.append({
                        'figure_index': idx,
                        'graph_type': graph_data['chart_type'],
                        'severity': 'low',
                        'description': f"Chart type detected: {graph_data['chart_type']}",
                        'evidence': graph_data,
                        'confidence': graph_data.get('confidence', 0),
                    })

            await db.commit()

            # Step 6: Validity check
            job.current_step = JobStep.VALIDITY_CHECK
            job.progress = 75
            await db.commit()

            from app.services.validity.checker import check_scientific_validity
            from app.services.extraction.pdf_extractor import extract_text_from_pdf
            full_text = extract_text_from_pdf(pdf_bytes)
            domain = 'materials_science'  # TODO: classify from text
            validity_findings = check_scientific_validity(domain, full_text)

            # Step 7: Ambiguity detection
            job.current_step = JobStep.AMBIGUITY_DETECTION
            job.progress = 85
            await db.commit()

            from app.services.ambiguity.detector import detect_pictorial_ambiguity
            captions = [f['caption'] for f in figures]
            ambiguity_findings = detect_pictorial_ambiguity(all_images, domain, full_text, captions)

            # Step 8: Generate reports
            job.current_step = JobStep.REPORT_GENERATION
            job.progress = 90
            await db.commit()

            from app.services.reports.generator import generate_ugc_report, generate_deplagiarized_pdf

            # Prepare report data
            paper_data = {
                'id': paper.id,
                'title': paper.title,
                'doi': paper.doi,
                'authors': paper.authors,
            }

            figures_data = []
            for figure, fig_data, hashes, clip_emb in figure_records:
                match = next((m for m in all_matches if m.get('figure_id') == figure.id), None)
                figures_data.append({
                    'index': figure.id,
                    'page': figure.page_num,
                    'similarity': match['similarity_score'] if match else 0,
                    'match_type': match['match_type'] if match else 'unique',
                    'source': match['source'] if match else 'Original',
                    'flags': match['details'].get('flags', []) if match else [],
                })

            findings = {
                'manipulation': manipulation_findings,
                'graph_semantics': graph_findings,
                'validity': validity_findings,
                'ambiguity': ambiguity_findings,
            }

            report_pdf = generate_ugc_report(paper_data, figures_data, all_matches, findings)
            deplagiarized_pdf = generate_deplagiarized_pdf(pdf_bytes, figures_data)

            # Upload reports to MinIO
            report_path = f'orgs/{paper.org_id}/papers/{paper.id}/report.pdf'
            deplag_path = f'orgs/{paper.org_id}/papers/{paper.id}/deplagiarized.pdf'
            storage.upload_bytes(report_path, report_pdf, 'application/pdf')
            storage.upload_bytes(deplag_path, deplagiarized_pdf, 'application/pdf')

            # Create report record
            report = Report(
                paper_id=paper.id,
                status=ReportStatus.COMPLETED,
                summary={
                    'total_figures': paper.total_figures,
                    'flagged_figures': sum(1 for f in figures_data if f['similarity'] >= 70),
                    'external_matches': sum(1 for f in figures_data if f['match_type'] != 'unique'),
                    'manipulation_findings': len(manipulation_findings),
                    'graph_semantics_findings': len(graph_findings),
                    'validity_issues': len(validity_findings),
                    'ambiguity_findings': len(ambiguity_findings),
                },
                report_minio_path=report_path,
                deplagiarized_minio_path=deplag_path,
                completed_at=datetime.utcnow(),
            )
            db.add(report)

            # Update paper
            paper.status = PaperStatus.COMPLETED
            paper.flagged_figures = sum(1 for f in figures_data if f['similarity'] >= 70)
            paper.overall_similarity = max((f['similarity'] for f in figures_data), default=0)
            paper.risk_level = 'CRITICAL' if paper.overall_similarity >= 50 else ('HIGH' if paper.overall_similarity >= 25 else ('MODERATE' if paper.overall_similarity >= 10 else 'LOW'))

            # Step 9: Index to repository if opted in
            job.current_step = JobStep.INDEXING
            job.progress = 95
            await db.commit()

            org = await db.get(Organization, paper.org_id)
            if org and org.repo_opt_in:
                from app.services.repository.shodhganga import index_org_repository
                await index_org_repository(db, paper.org_id)

            # Finalize
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.utcnow()
            job.result = {'report_path': report_path, 'deplagiarized_path': deplag_path}
            await db.commit()

            # Fire webhooks
            from app.workers.callbacks import fire_webhook
            await fire_webhook(db, paper.org_id, 'paper.processed', {'paper_id': paper.id, 'report_path': report_path})
            if paper.overall_similarity >= 70:
                await fire_webhook(db, paper.org_id, 'high_similarity.alert', {'paper_id': paper.id, 'similarity': paper.overall_similarity})

            logger.info(f"Paper {paper.id} processing completed successfully")

        except Exception as e:
            logger.exception(f"Paper processing failed for job {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            paper.status = PaperStatus.FAILED
            paper.processing_error = str(e)
            await db.commit()

            # Fire failure webhook
            from app.workers.callbacks import fire_webhook
            await fire_webhook(db, paper.org_id, 'paper.failed', {'paper_id': paper.id, 'error': str(e)})


@celery_app.task(name='app.workers.tasks.sync_shodhganga_task')
def sync_shodhganga_task():
    """Sync Shodhganga theses"""
    import asyncio
    asyncio.run(_sync_shodhganga_async())


async def _sync_shodhganga_async():
    async with async_session_factory() as db:
        from app.services.repository.shodhganga import sync_shodhganga
        count = await sync_shodhganga(db, limit=50)
        logger.info(f"Shodhganga sync completed: {count} theses indexed")


@celery_app.task(name='app.workers.tasks.cleanup_temp_files_task')
def cleanup_temp_files_task():
    """Clean up temporary files"""
    logger.info("Cleanup temp files task executed")


@celery_app.task(name='app.workers.tasks.reindex_embeddings_task')
def reindex_embeddings_task():
    """Reindex embeddings for vector search"""
    import asyncio
    asyncio.run(_reindex_embeddings_async())


async def _reindex_embeddings_async():
    async with async_session_factory() as db:
        from app.services.vector_search.hybrid_search import HybridVectorSearch
        search = HybridVectorSearch()
        search.mark_dirty()
        logger.info("Vector index marked dirty for rebuild")


@celery_app.task(name='app.workers.tasks.deliver_webhook_task')
def deliver_webhook_task(webhook_id: int, payload: dict):
    """Deliver webhook with retries"""
    import asyncio
    asyncio.run(_deliver_webhook_async(webhook_id, payload))


async def _deliver_webhook_async(webhook_id: int, payload: dict):
    import httpx
    from app.core.config import get_settings
    from app.models import WebhookEndpoint

    settings = get_settings()
    async with async_session_factory() as db:
        webhook = await db.get(WebhookEndpoint, webhook_id)
        if not webhook or not webhook.is_active:
            return

        for attempt in range(settings.webhook_max_retries):
            try:
                async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
                    response = await client.post(
                        webhook.url,
                        json=payload,
                        headers={'X-Webhook-Signature': webhook.secret}
                    )
                    response.raise_for_status()
                    webhook.last_success_at = datetime.utcnow()
                    webhook.failure_count = 0
                    await db.commit()
                    return
            except Exception as e:
                logger.warning(f"Webhook delivery attempt {attempt + 1} failed: {e}")
                webhook.failure_count += 1
                await db.commit()

        webhook.last_triggered_at = datetime.utcnow()
        await db.commit()