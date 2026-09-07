import re
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


_DOMAIN_KEYWORDS = {
    "materials_science": [
        "xrd", "x-ray diffraction", "crystallite", "lattice parameter", "scherrer",
        "band gap", "semiconductor", "nanoparticle", "thin film", "sputtering",
        "annealing", "morphology", "grain size", "phase analysis", "diffraction peak",
        "miller indices", "crystal structure", "unit cell", "doping", "ferroelectric",
        "piezoelectric", "dielectric", "perovskite", "oxide", "ceramic"
    ],
    "chemistry": [
        "synthesis", "reaction yield", "catalyst", "ftir", "nmr", "uv-vis",
        "chromatography", "mass spectrometry", "molar ratio", "stoichiometry",
        "reflux", "precipitation", "hydrothermal", "sol-gel", "characterization",
        "functional group", "bond vibration", "chemical shift", "peak assignment"
    ],
    "physics": [
        "quantum", "photoluminescence", "raman", "magnetization", "hysteresis",
        "ferromagnetic", "superconductivity", "critical temperature", "hall effect",
        "carrier concentration", "mobility", "band structure", "density of states",
        "phase transition", "order parameter", "symmetry breaking"
    ],
    "biology": [
        "cell culture", "western blot", "pcr", "elisa", "microscopy", "confocal",
        "flow cytometry", "gene expression", "protein", "antibody", "viability",
        "proliferation", "apoptosis", "differentiation", "immunohistochemistry",
        "rna-seq", "sequencing", "genome", "transcriptome", "proteome"
    ],
    "environmental": [
        "adsorption", "photocatalysis", "degradation", "water treatment", "heavy metal",
        "pollutant", "contaminant", "isotherm", "kinetics", "langmuir", "freundlich",
        "photodegradation", "wastewater", "effluent", "remediation"
    ],
    "electrical_engineering": [
        "transistor", "mosfet", "cmos", "finfet", "threshold voltage", "subthreshold",
        "mobility", "on/off ratio", "transfer curve", "output curve", "capacitance",
        "c-v", "i-v", "breakdown", "leakage current", "gate oxide", "high-k"
    ],
    "mechanical_engineering": [
        "tensile strength", "young modulus", "hardness", "fracture toughness",
        "fatigue", "creep", "wear", "compression", "stress-strain", "dislocation",
        "grain boundary", "phase transformation", "martensite", "austenite"
    ],
}

_DOMAIN_EMBEDDINGS = {}
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        _precompute_embeddings()
    return _MODEL


def _precompute_embeddings():
    global _DOMAIN_EMBEDDINGS
    model = get_model()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        text = " ".join(keywords)
        _DOMAIN_EMBEDDINGS[domain] = model.encode(text)


def classify_domain(text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    model = get_model()
    text_emb = model.encode(text)
    
    scores = {}
    for domain, domain_emb in _DOMAIN_EMBEDDINGS.items():
        cos_sim = np.dot(text_emb, domain_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(domain_emb))
        scores[domain] = float(max(0.0, cos_sim))
    
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_domains[:top_k]


def extract_key_terms(text: str, domain: str) -> List[str]:
    keywords = _DOMAIN_KEYWORDS.get(domain, [])
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            found.append(kw)
    return found


def get_domain_context(domain: str) -> Dict:
    contexts = {
        "materials_science": {
            "typical_characterizations": ["XRD", "SEM", "TEM", "FTIR", "UV-Vis", "PL", "Raman", "XPS", "VSM", "EIS"],
            "impossible_patterns": {
                "xrd": ["negative_intensity", "peak_at_zero", "identical_peaks_different_samples", "impossible_d_spacing"],
                "sem": ["identical_morphology_different_conditions", "impossible_scale_bar"],
                "graphs": ["perfect_linear_fit_r2_1", "identical_curves_different_params", "negative_concentration"]
            },
            "physical_constraints": {
                "band_gap_range": (0.1, 6.0),
                "crystallite_size_nm": (1, 500),
                "lattice_constant_angstrom": (2, 10),
                "annealing_temp_c": (100, 1500)
            }
        },
        "chemistry": {
            "typical_characterizations": ["FTIR", "NMR", "UV-Vis", "MS", "TGA", "DSC", "GC-MS", "HPLC"],
            "impossible_patterns": {
                "ftir": ["negative_absorbance", "peaks_beyond_range", "identical_spectra_different_compounds"],
                "nmr": ["impossible_chemical_shift", "negative_integration"],
                "graphs": ["yield_over_100", "negative_concentration", "impossible_kinetics"]
            },
            "physical_constraints": {
                "reaction_yield_percent": (0, 100),
                "concentration_molar": (1e-6, 20),
                "temperature_c": (-80, 300)
            }
        },
        "biology": {
            "typical_characterizations": ["Western Blot", "PCR", "ELISA", "Microscopy", "Flow Cytometry", "Sequencing"],
            "impossible_patterns": {
                "western": ["identical_bands_different_samples", "impossible_molecular_weight", "loading_control_mismatch"],
                "pcr": ["amplification_without_template", "impossible_ct_values"],
                "graphs": ["cell_viability_over_100", "negative_proliferation", "impossible_doubling_time"]
            },
            "physical_constraints": {
                "cell_viability_percent": (0, 100),
                "ct_value": (10, 40),
                "molecular_weight_kda": (5, 500)
            }
        },
    }
    return contexts.get(domain, {})