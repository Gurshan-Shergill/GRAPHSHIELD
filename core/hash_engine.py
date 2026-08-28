from PIL import Image
import imagehash

def compute_phash(image_path: str) -> str:
    """Computes Perceptual Hash (pHash) for macro structural similarity."""
    img = Image.open(image_path)
    return str(imagehash.phash(img))

def compute_dhash(image_path: str) -> str:
    """Computes Difference Hash (dHash) for graph edge and gradient alignment."""
    img = Image.open(image_path)
    return str(imagehash.dhash(img))