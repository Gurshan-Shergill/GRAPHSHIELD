import imagehash
from PIL import Image
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_perceptual_hashes(image: Image.Image) -> Dict[str, str]:
    '''Compute multiple perceptual hashes for an image'''
    try:
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        hashes = {
            'phash': str(imagehash.phash(image)),
            'dhash': str(imagehash.dhash(image)),
            'whash': str(imagehash.whash(image)),
            'ahash': str(imagehash.average_hash(image)),
        }
        return hashes
    except Exception as e:
        logger.error(f'Failed to compute hashes: {e}')
        return {'phash': '', 'dhash': '', 'whash': '', 'ahash': ''}


def hamming_distance(hash1: str, hash2: str) -> int:
    '''Calculate Hamming distance between two hex hashes'''
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return 64  # Max distance


def similarity_from_hash(hash1: str, hash2: str, hash_size: int = 64) -> float:
    '''Convert Hamming distance to similarity percentage'''
    dist = hamming_distance(hash1, hash2)
    return max(0.0, (1.0 - dist / hash_size) * 100.0)
