from PIL import Image
from typing import List, Dict, Any
import logging

# Import existing forensics
from core.image_forensics import analyze_image_forensics, ForensicFinding

logger = logging.getLogger(__name__)


def detect_manipulation(
    image: Image.Image,
    all_images: List[Image.Image] = None,
    index: int = 0
) -> List[Dict[str, Any]]:
    '''
    Detect image manipulation using ELA, SIFT, inversion, flip, rotation detection.
    Returns list of findings as dicts.
    '''
    findings = analyze_image_forensics(image, all_images, index)

    # Convert to dict format
    return [
        {
            'type': f.type,
            'severity': f.severity,
            'description': f.description,
            'evidence': f.evidence,
            'confidence': f.confidence,
            'location': f.location,
        }
        for f in findings
    ]


def compute_manipulation_score(findings: List[Dict[str, Any]]) -> float:
    '''Compute overall manipulation score (0-100)'''
    if not findings:
        return 0.0

    severity_weights = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1}
    total_score = 0.0
    for f in findings:
        weight = severity_weights.get(f['severity'], 0.1)
        total_score += weight * f['confidence']

    # Normalize to 0-100
    return min(100.0, total_score * 20.0)
