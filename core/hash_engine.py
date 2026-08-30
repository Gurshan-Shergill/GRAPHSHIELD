import imagehash

def generate_image_hashes(pil_images):
    """
    Calculates perceptual hashes for a list of PIL images.
    """
    hashes = []
    for img in pil_images:
        try:
            # Perceptual hash for image structure/visual comparison
            h = imagehash.phash(img)
            hashes.append(h)
        except Exception as e:
            continue
    return hashes

def compare_image_hashes(hashes1, hashes2, max_hamming_distance=10):
    """
    Compares two lists of image hashes to detect duplicate or plagiarized images.
    """
    matched_count = 0
    for h1 in hashes1:
        for h2 in hashes2:
            # Difference in hamming distance: smaller difference means closer match
            if abs(h1 - h2) <= max_hamming_distance:
                matched_count += 1
                break
                
    return matched_count