import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidityIssue:
    severity: str
    category: str
    description: str
    evidence: str
    location: str
    confidence: float


class ScientificValidityChecker:
    def __init__(self, domain: str):
        self.domain = domain
        self.context = self._get_domain_context()
    
    def _get_domain_context(self) -> Dict:
        contexts = {
            "materials_science": {
                "xrd_constraints": {
                    "two_theta_range": (5, 90),
                    "intensity_non_negative": True,
                    "d_spacing_range": (0.5, 10),
                    "fwhm_positive": True,
                    "crystallite_size_nm": (1, 500)
                },
                "band_gap_range_ev": (0.1, 6.0),
                "annealing_temp_range_c": (100, 1500),
                "impossible_patterns": [
                    "identical_xrd_peaks_different_compositions",
                    "negative_diffraction_intensity",
                    "peak_at_zero_degrees",
                    "impossible_miller_indices",
                    "lattice_parameter_out_of_range"
                ]
            },
            "chemistry": {
                "yield_range": (0, 100),
                "concentration_range_m": (1e-9, 50),
                "temperature_range_c": (-196, 500),
                "ph_range": (0, 14),
                "impossible_patterns": [
                    "yield_over_100_percent",
                    "negative_absorbance",
                    "negative_concentration",
                    "impossible_nmr_shifts",
                    "mass_balance_violation"
                ]
            },
            "biology": {
                "viability_range": (0, 100),
                "ct_value_range": (10, 45),
                "mw_range_kda": (3, 500),
                "impossible_patterns": [
                    "viability_over_100",
                    "negative_ct_value",
                    "identical_western_bands",
                    "loading_control_inconsistency",
                    "impossible_doubling_time"
                ]
            },
            "physics": {
                "temperature_range_k": (0, 2000),
                "magnetic_field_range_t": (0, 50),
                "impossible_patterns": [
                    "negative_resistance",
                    "superconductivity_above_tc",
                    "violation_thermodynamics"
                ]
            }
        }
        return contexts.get(self.domain, {})
    
    def check_text_claims(self, text: str) -> List[ValidityIssue]:
        issues = []
        
        numbers_with_units = re.findall(
            r'(\d+(?:\.\d+)?)\s*([a-zA-Z/%°µμ]+)', text
        )
        
        for value_str, unit in numbers_with_units:
            try:
                value = float(value_str)
                issues.extend(self._check_value_constraints(value, unit, text))
            except ValueError:
                pass
        
        issues.extend(self._check_impossible_claims(text))
        
        return issues
    
    def _check_value_constraints(self, value: float, unit: str, context: str) -> List[ValidityIssue]:
        issues = []
        unit_lower = unit.lower()
        
        if self.domain == "materials_science":
            if "ev" in unit_lower or "band gap" in context.lower():
                if not (0.1 <= value <= 6.0):
                    issues.append(ValidityIssue(
                        severity="high",
                        category="physical_impossibility",
                        description=f"Band gap {value} {unit} outside physically possible range for semiconductors/insulators",
                        evidence=f"Found: {value} {unit} in text",
                        location="text",
                        confidence=0.85
                    ))
            elif "nm" in unit_lower and ("crystallite" in context.lower() or "grain" in context.lower() or "particle" in context.lower()):
                if not (1 <= value <= 500):
                    issues.append(ValidityIssue(
                        severity="medium",
                        category="unlikely_value",
                        description=f"Crystallite/particle size {value} nm is unusual",
                        evidence=f"Found: {value} {unit} in text",
                        location="text",
                        confidence=0.6
                    ))
            elif "°c" in unit_lower or "c" == unit_lower:
                if "anneal" in context.lower() or "sinter" in context.lower() or "calcine" in context.lower():
                    if not (100 <= value <= 1500):
                        issues.append(ValidityIssue(
                            severity="medium",
                            category="unlikely_value",
                            description=f"Annealing temperature {value}°C outside typical range",
                            evidence=f"Found: {value} {unit} in text",
                            location="text",
                            confidence=0.65
                        ))
        
        elif self.domain == "chemistry":
            if "%" in unit and "yield" in context.lower():
                if not (0 <= value <= 100):
                    issues.append(ValidityIssue(
                        severity="critical",
                        category="physical_impossibility",
                        description=f"Reaction yield {value}% exceeds 100%",
                        evidence=f"Found: {value}% yield in text",
                        location="text",
                        confidence=0.95
                    ))
            elif "m" == unit_lower or "mol/l" in unit_lower or "molar" in context.lower():
                if not (1e-9 <= value <= 50):
                    issues.append(ValidityIssue(
                        severity="medium",
                        category="unlikely_value",
                        description=f"Concentration {value} M is unusual",
                        evidence=f"Found: {value} {unit} in text",
                        location="text",
                        confidence=0.55
                    ))
        
        elif self.domain == "biology":
            if "%" in unit and ("viability" in context.lower() or "proliferation" in context.lower()):
                if not (0 <= value <= 100):
                    issues.append(ValidityIssue(
                        severity="critical",
                        category="physical_impossibility",
                        description=f"Cell viability/proliferation {value}% exceeds 100%",
                        evidence=f"Found: {value}% in text",
                        location="text",
                        confidence=0.95
                    ))
        
        return issues
    
    def _check_impossible_claims(self, text: str) -> List[ValidityIssue]:
        issues = []
        text_lower = text.lower()
        
        impossible_phrases = {
            "materials_science": [
                (r"crystallite size.*?0\s*nm", "Crystallite size cannot be zero"),
                (r"negative.*intensity", "Negative diffraction intensity is impossible"),
                (r"peak at 0°", "XRD peak at 0° is impossible"),
                (r"lattice parameter.*?100\s*[Åa]", "Lattice parameter >100Å is impossible for typical crystals"),
                (r"band gap.*?10\s*ev", "Band gap >10eV is extremely rare"),
            ],
            "chemistry": [
                (r"yield.*?1\d{2,}\s*%", "Yield over 100% is impossible"),
                (r"negative.*absorbance", "Negative absorbance is impossible"),
                (r"negative.*concentration", "Negative concentration is impossible"),
            ],
            "biology": [
                (r"viability.*?1\d{2,}\s*%", "Cell viability over 100% is impossible"),
                (r"ct value.*?[0-9]\s*$", "Ct value <10 is extremely unlikely"),
            ]
        }
        
        patterns = impossible_phrases.get(self.domain, [])
        for pattern, desc in patterns:
            if re.search(pattern, text_lower):
                issues.append(ValidityIssue(
                    severity="critical",
                    category="physical_impossibility",
                    description=desc,
                    evidence=f"Pattern matched in text",
                    location="text",
                    confidence=0.9
                ))
        
        return issues
    
    def check_xrd_data(self, peaks: List[Dict]) -> List[ValidityIssue]:
        issues = []
        if self.domain != "materials_science":
            return issues
        
        constraints = self.context.get("xrd_constraints", {})
        two_theta_range = constraints.get("two_theta_range", (5, 90))
        
        for i, peak in enumerate(peaks):
            two_theta = peak.get("two_theta", 0)
            intensity = peak.get("intensity", 0)
            fwhm = peak.get("fwhm", 0)
            d_spacing = peak.get("d_spacing", 0)
            
            if not (two_theta_range[0] <= two_theta <= two_theta_range[1]):
                issues.append(ValidityIssue(
                    severity="high",
                    category="xrd_anomaly",
                    description=f"Peak {i+1}: 2θ = {two_theta}° outside valid range {two_theta_range}",
                    evidence=f"Peak data: 2θ={two_theta}, I={intensity}",
                    location=f"peak_{i+1}",
                    confidence=0.9
                ))
            
            if intensity < 0 and constraints.get("intensity_non_negative", True):
                issues.append(ValidityIssue(
                    severity="critical",
                    category="xrd_anomaly",
                    description=f"Peak {i+1}: Negative intensity {intensity} is physically impossible",
                    evidence=f"Peak data: 2θ={two_theta}, I={intensity}",
                    location=f"peak_{i+1}",
                    confidence=0.99
                ))
            
            if fwhm <= 0 and constraints.get("fwhm_positive", True):
                issues.append(ValidityIssue(
                    severity="high",
                    category="xrd_anomaly",
                    description=f"Peak {i+1}: FWHM = {fwhm} must be positive",
                    evidence=f"Peak data: FWHM={fwhm}",
                    location=f"peak_{i+1}",
                    confidence=0.85
                ))
        
        return issues
    
    def check_graph_data(self, graph_type: str, data_points: List[Tuple[float, float]], 
                         metadata: Dict = None) -> List[ValidityIssue]:
        issues = []
        
        if len(data_points) < 2:
            return issues
        
        x_vals = np.array([p[0] for p in data_points])
        y_vals = np.array([p[1] for p in data_points])
        
        if np.any(y_vals < 0) and graph_type in ["absorbance", "intensity", "concentration", "viability", "yield"]:
            issues.append(ValidityIssue(
                severity="critical",
                category="graph_anomaly",
                description=f"{graph_type.capitalize()} graph contains negative values which are physically impossible",
                evidence=f"Min y-value: {np.min(y_vals):.4f}",
                location="graph_data",
                confidence=0.95
            ))
        
        if graph_type == "xrd":
            issues.extend(self._check_xrd_patterns(x_vals, y_vals))
        
        elif graph_type in ["iv", "cv", "transfer", "output"]:
            issues.extend(self._check_electrical_characteristics(x_vals, y_vals, graph_type))
        
        elif graph_type in ["kinetics", "isotherm", "degradation"]:
            issues.extend(self._check_kinetic_models(x_vals, y_vals, graph_type))
        
        return issues
    
    def _check_xrd_patterns(self, two_theta: np.ndarray, intensity: np.ndarray) -> List[ValidityIssue]:
        issues = []
        
        if np.all(intensity == intensity[0]):
            issues.append(ValidityIssue(
                severity="high",
                category="xrd_anomaly",
                description="XRD pattern shows constant intensity - likely fabricated or corrupted data",
                evidence="All intensity values identical",
                location="xrd_pattern",
                confidence=0.8
            ))
        
        peaks_idx = self._find_peaks(intensity)
        if len(peaks_idx) > 1:
            peak_positions = two_theta[peaks_idx]
            diffs = np.diff(np.sort(peak_positions))
            if np.std(diffs) < 0.01 and len(diffs) > 2:
                issues.append(ValidityIssue(
                    severity="high",
                    category="xrd_anomaly",
                    description="XRD peaks show unnaturally uniform spacing - possible synthetic data",
                    evidence=f"Peak spacing std: {np.std(diffs):.4f}°",
                    location="xrd_pattern",
                    confidence=0.7
                ))
        
        return issues
    
    def _check_electrical_characteristics(self, x: np.ndarray, y: np.ndarray, gtype: str) -> List[ValidityIssue]:
        issues = []
        
        if gtype == "iv":
            if np.all(y >= 0) and np.all(x >= 0):
                pass
            elif np.any(y < 0) and np.any(y > 0):
                pass
            else:
                issues.append(ValidityIssue(
                    severity="medium",
                    category="electrical_anomaly",
                    description="I-V curve shows unusual quadrant behavior",
                    evidence="Current values span unexpected range",
                    location="iv_curve",
                    confidence=0.6
                ))
        
        return issues
    
    def _check_kinetic_models(self, x: np.ndarray, y: np.ndarray, gtype: str) -> List[ValidityIssue]:
        issues = []
        
        if gtype == "degradation" and np.any(y > 100):
            issues.append(ValidityIssue(
                severity="high",
                category="kinetic_anomaly",
                description="Degradation efficiency exceeds 100%",
                evidence=f"Max value: {np.max(y):.2f}%",
                location="degradation_curve",
                confidence=0.9
            ))
        
        return issues
    
    def _find_peaks(self, data: np.ndarray, distance: int = 5) -> np.ndarray:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, distance=distance)
        return peaks


def check_scientific_validity(domain: str, text: str, 
                               extracted_data: Dict = None) -> List[ValidityIssue]:
    checker = ScientificValidityChecker(domain)
    issues = []
    
    issues.extend(checker.check_text_claims(text))
    
    if extracted_data:
        if "xrd_peaks" in extracted_data:
            issues.extend(checker.check_xrd_data(extracted_data["xrd_peaks"]))
        
        if "graphs" in extracted_data:
            for gname, gdata in extracted_data["graphs"].items():
                issues.extend(checker.check_graph_data(
                    gname, gdata.get("points", []), gdata.get("metadata", {})
                ))
    
    return issues