import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import calculate_hamming_distance
from core.text_engine import calculate_text_similarity, calculate_semantic_similarity
from core.hash_engine import generate_image_hashes
from PIL import Image


def test_hamming_distance():
    assert calculate_hamming_distance("0000000000000000", "0000000000000000") == 0
    assert calculate_hamming_distance("0000000000000000", "ffffffffffffffff") == 64
    assert calculate_hamming_distance("abcdef1234567890", "abcdef1234567890") == 0


def test_text_similarity():
    text1 = "This is a test sentence for similarity checking."
    text2 = "This is a test sentence for similarity checking."
    assert calculate_text_similarity(text1, text2) == 100.0
    
    text3 = "Completely different text with no overlap."
    sim = calculate_text_similarity(text1, text3)
    assert 0 <= sim < 50


def test_semantic_similarity_fallback():
    text1 = "Machine learning models process data efficiently."
    text2 = "ML algorithms handle information effectively."
    sim = calculate_semantic_similarity(text1, text2)
    assert 0 <= sim <= 100


def test_image_hash_generation():
    img = Image.new('RGB', (100, 100), color='red')
    hashes = generate_image_hashes([img])
    assert len(hashes) == 1
    assert len(str(hashes[0])) == 16


class TestDatabase:
    @pytest.mark.asyncio
    async def test_init_db_sqlite(self):
        from core.database import init_db, close_pool, get_sqlite_conn
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            os.environ['USE_CLOUD_SQL'] = 'false'
            os.environ['SQLITE_PATH'] = db_path
            
            await init_db()
            
            conn = get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='figure_hashes'")
            assert cursor.fetchone() is not None
            conn.close()
        finally:
            await close_pool()
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])