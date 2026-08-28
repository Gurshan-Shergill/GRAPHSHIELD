from PIL import Image
import imagehash

def compute_phash(image_path: str) -> str:
    """Generates a 64-bit perceptual hash string for a given image."""
    img = Image.open(image_path)
    # Convert image to pHash
    hash_value = imagehash.phash(img)
    return str(hash_value)

def compare_images(image_path_1: str, image_path_2: str) -> dict:
    """
    Compares two images using pHash and returns the Hamming Distance
    and a normalized similarity percentage.
    """
    hash1 = imagehash.phash(Image.open(image_path_1))
    hash2 = imagehash.phash(Image.open(image_path_2))
    
    # Hamming distance: number of differing bits out of 64
    distance = hash1 - hash2
    
    # Convert distance to similarity score (0.0 to 100.0%)
    similarity = max(0.0, (1.0 - (distance / 64.0)) * 100.0)
    
    return {
        "hamming_distance": distance,
        "similarity_score": round(similarity, 2),
        "is_suspicious": distance <= 10  # Flag if distance is low (highly similar)
    }

if __name__ == "__main__":
    print("pHash Engine initialized successfully.")