"""Initial migration - create figure_hashes table

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-15 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'figure_hashes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('paper_name', sa.Text(), nullable=False),
        sa.Column('figure_path', sa.Text(), nullable=False),
        sa.Column('phash', sa.Text(), nullable=False),
        sa.Column('dhash', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_figure_hashes_phash', 'figure_hashes', ['phash'])
    op.create_index('idx_figure_hashes_paper', 'figure_hashes', ['paper_name'])


def downgrade() -> None:
    op.drop_index('idx_figure_hashes_paper', table_name='figure_hashes')
    op.drop_index('idx_figure_hashes_phash', table_name='figure_hashes')
    op.drop_table('figure_hashes')