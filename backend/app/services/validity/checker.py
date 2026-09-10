from typing import List, Dict, Any
import logging

# Import existing validity checker
from core.validity_checker import check_scientific_validity as core_check_validity, ValidityIssue

logger = logging.getLogger(__name__)


def check_scientific_validity(domain: str, text: str, extracted_data: Dict = None) -> List[Dict[str, Any]]:
    '''Check scientific validity of claims in text and extracted data'''
    issues = core_check_validity(domain, text, extracted_data)

    # Convert to dict format
    return [
        {
            'severity': issue.severity,
            'category': issue.category,
            'description': issue.description,
            'evidence': issue.evidence,
            'location': issue.location,
            'confidence': issue.confidence,
        }
        for issue in issues
    ]
