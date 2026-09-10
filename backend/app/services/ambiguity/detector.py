from PIL import Image
from typing import List, Dict, Any
import logging

# Import existing ambiguity detector
from core.ambiguity_detector import detect_pictorial_ambiguity as core_detect_ambiguity, AmbiguityFinding

logger = logging.getLogger(__name__)


def detect_pictorial_ambiguity(
    images: List[Image.Image],
    domain: str,
    paper_text: str = '',
    captions: List[str] = None
) -> List[Dict[str, Any]]:
    '''Detect pictorial ambiguity and domain consistency issues'''
    findings = core_detect_ambiguity(images, domain, paper_text, captions)

    # Convert to dict format
    return [
        {
            'type': f.type,
            'severity': f.severity,
            'description': f.description,
            'evidence': f.evidence,
            'confidence': f.confidence,
            'figure_index': f.figure_index,
        }
        for f in findings
    ]
