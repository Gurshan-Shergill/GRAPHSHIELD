"""Initial migration - create all tables

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Organizations
    op.create_table(
        'organizations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('repo_opt_in', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('shodhganga_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug'),
    )
    op.create_index('ix_organizations_slug_active', 'organizations', ['slug', 'is_active'])

    # Users
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('role', sa.Enum('org_admin', 'instructor', 'researcher', 'viewer', name='userrole'), nullable=False, server_default='researcher'),
        sa.Column('api_key_hash', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
    )
    op.create_index('ix_users_org_email', 'users', ['org_id', 'email'])

    # Papers
    op.create_table(
        'papers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('uploaded_by_id', sa.Integer(), nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('doi', sa.String(100), nullable=True),
        sa.Column('authors', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('source', sa.String(200), nullable=True),
        sa.Column('status', sa.Enum('uploaded', 'processing', 'completed', 'failed', name='paperstatus'), nullable=False, server_default='uploaded'),
        sa.Column('minio_path', sa.String(500), nullable=True),
        sa.Column('total_figures', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('flagged_figures', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('overall_similarity', sa.Float(), nullable=True),
        sa.Column('risk_level', sa.String(50), nullable=True),
        sa.Column('processing_error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['uploaded_by_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_papers_org_status', 'papers', ['org_id', 'status'])
    op.create_index('ix_papers_org_created', 'papers', ['org_id', 'created_at'])
    op.create_index('ix_papers_doi', 'papers', ['doi'])

    # Figures
    op.create_table(
        'figures',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('paper_id', sa.Integer(), nullable=False),
        sa.Column('page_num', sa.Integer(), nullable=False),
        sa.Column('bbox', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('minio_path', sa.String(500), nullable=True),
        sa.Column('phash', sa.String(64), nullable=False),
        sa.Column('dhash', sa.String(64), nullable=False),
        sa.Column('clip_embedding', Vector(512), nullable=True),
        sa.Column('manipulation_score', sa.Float(), nullable=True),
        sa.Column('ocr_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('figure_type', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_figures_paper_page', 'figures', ['paper_id', 'page_num'])
    op.create_index('ix_figures_phash_paper', 'figures', ['phash', 'paper_id'])
    op.create_index('ix_figures_clip_embedding_hnsw', 'figures', ['clip_embedding'], postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'clip_embedding': 'vector_cosine_ops'})

    # Similarity Matches
    op.create_table(
        'similarity_matches',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('query_figure_id', sa.Integer(), nullable=False),
        sa.Column('matched_figure_id', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('match_type', sa.Enum('exact', 'near_duplicate', 'semantic', 'manipulated', 'internal_duplicate', 'shodhganga', 'external', name='matchtype'), nullable=False, server_default='semantic'),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['query_figure_id'], ['figures.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['matched_figure_id'], ['figures.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_matches_query_score', 'similarity_matches', ['query_figure_id', 'similarity_score'])
    op.create_index('ix_matches_matched_score', 'similarity_matches', ['matched_figure_id', 'similarity_score'])

    # Reports
    op.create_table(
        'reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('paper_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('pending', 'generating', 'completed', 'failed', name='reportstatus'), nullable=False, server_default='pending'),
        sa.Column('summary', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('report_minio_path', sa.String(500), nullable=True),
        sa.Column('deplagiarized_minio_path', sa.String(500), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_reports_paper_status', 'reports', ['paper_id', 'status'])

    # Jobs
    op.create_table(
        'jobs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('paper_id', sa.Integer(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('queued', 'processing', 'completed', 'failed', 'cancelled', name='jobstatus'), nullable=False, server_default='queued'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_step', sa.Enum('extraction', 'fingerprinting', 'vector_search', 'manipulation_detection', 'graph_semantics', 'validity_check', 'ambiguity_detection', 'report_generation', 'indexing', name='jobstep'), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('celery_task_id', sa.String(100), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_jobs_paper_status', 'jobs', ['paper_id', 'status'])
    op.create_index('ix_jobs_org_status', 'jobs', ['org_id', 'status'])

    # API Keys
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('scopes', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('rate_limit', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('rate_limit_window', sa.Integer(), nullable=False, server_default='60'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_api_keys_org_active', 'api_keys', ['org_id', 'is_active'])

    # Shodhganga Theses
    op.create_table(
        'shodhganga_theses',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('thesis_id', sa.String(100), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('author', sa.String(255), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('university', sa.String(255), nullable=False),
        sa.Column('department', sa.String(255), nullable=True),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('pdf_url', sa.String(500), nullable=True),
        sa.Column('figure_embeddings', postgresql.ARRAY(Vector(512)), nullable=True),
        sa.Column('figure_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thesis_id'),
    )
    op.create_index('ix_shodhganga_university_year', 'shodhganga_theses', ['university', 'year'])
    op.create_index('ix_shodhganga_author', 'shodhganga_theses', ['author'])
    op.create_index('ix_shodhganga_embeddings_hnsw', 'shodhganga_theses', ['figure_embeddings'], postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'figure_embeddings': 'vector_cosine_ops'})

    # Webhook Endpoints
    op.create_table(
        'webhook_endpoints',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(500), nullable=False),
        sa.Column('events', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('secret', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('failure_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_webhooks_org_active', 'webhook_endpoints', ['org_id', 'is_active'])


def downgrade() -> None:
    op.drop_table('webhook_endpoints')
    op.drop_table('shodhganga_theses')
    op.drop_table('api_keys')
    op.drop_table('jobs')
    op.drop_table('reports')
    op.drop_table('similarity_matches')
    op.drop_table('figures')
    op.drop_table('papers')
    op.drop_table('users')
    op.drop_table('organizations')
    op.execute('DROP EXTENSION IF EXISTS vector')