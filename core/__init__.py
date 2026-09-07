from .config import get_settings, Settings
from .database import init_db, save_figure_hash, find_matches, get_stats, close_pool
from .hash_engine import generate_image_hashes, compare_image_hashes
from .text_engine import extract_text_blocks, find_text_matches, calculate_semantic_similarity, calculate_text_similarity
from .domain_classifier import classify_domain, extract_key_terms, get_domain_context
from .validity_checker import check_scientific_validity, ValidityIssue, ScientificValidityChecker
from .image_forensics import analyze_image_forensics, ForensicFinding, ImageForensicsAnalyzer
from .graph_analyzer import analyze_graph_chart, GraphFinding, GraphChartAnalyzer
from .ambiguity_detector import detect_pictorial_ambiguity, AmbiguityFinding, PictorialAmbiguityDetector

__all__ = [
    "get_settings",
    "Settings",
    "init_db",
    "save_figure_hash",
    "find_matches",
    "get_stats",
    "close_pool",
    "generate_image_hashes",
    "compare_image_hashes",
    "extract_text_blocks",
    "find_text_matches",
    "calculate_semantic_similarity",
    "calculate_text_similarity",
    "classify_domain",
    "extract_key_terms",
    "get_domain_context",
    "check_scientific_validity",
    "ValidityIssue",
    "ScientificValidityChecker",
    "analyze_image_forensics",
    "ForensicFinding",
    "ImageForensicsAnalyzer",
    "analyze_graph_chart",
    "GraphFinding",
    "GraphChartAnalyzer",
    "detect_pictorial_ambiguity",
    "AmbiguityFinding",
    "PictorialAmbiguityDetector",
]