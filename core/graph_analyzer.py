import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
import re


@dataclass
class GraphFinding:
    graph_type: str
    severity: str
    description: str
    evidence: Dict
    confidence: float
    bbox: Tuple[int, int, int, int]


class GraphChartAnalyzer:
    def __init__(self, domain: str = "materials_science"):
        self.domain = domain
    
    def analyze(self, image: Image.Image) -> List[GraphFinding]:
        findings = []
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        chart_type = self._detect_chart_type(cv_image)
        
        if chart_type == "xrd":
            findings.extend(self._analyze_xrd_pattern(cv_image))
        elif chart_type in ["sem", "tem", "microscopy"]:
            findings.extend(self._analyze_microscopy_image(cv_image))
        elif chart_type in ["uvvis", "absorption", "pl", "photoluminescence"]:
            findings.extend(self._analyze_spectroscopy(cv_image, chart_type))
        elif chart_type in ["iv", "cv", "transfer", "output", "electrical"]:
            findings.extend(self._analyze_electrical_characteristics(cv_image))
        elif chart_type in ["kinetics", "isotherm", "degradation", "tga", "dsc"]:
            findings.extend(self._analyze_kinetic_thermal(cv_image, chart_type))
        else:
            findings.extend(self._analyze_generic_chart(cv_image))
        
        findings.extend(self._check_repeating_patterns(cv_image))
        findings.extend(self._check_impossible_values(cv_image, chart_type))
        
        return findings
    
    def _detect_chart_type(self, image: np.ndarray) -> str:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        text = self._extract_text_from_image(gray)
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["xrd", "x-ray diffraction", "2θ", "two theta", "diffraction", "intensity (a.u.)", "intensity (cps)"]):
            return "xrd"
        elif any(kw in text_lower for kw in ["sem", "tem", "magnification", "scale bar", "µm", "um", "nm"]):
            return "sem"
        elif any(kw in text_lower for kw in ["uv-vis", "uv-vis", "absorption", "absorbance", "wavelength (nm)", "λ (nm)"]):
            return "uvvis"
        elif any(kw in text_lower for kw in ["photoluminescence", "pl spectra", "emission", "excitation"]):
            return "pl"
        elif any(kw in text_lower for kw in ["i-v", "iv curve", "current (a)", "voltage (v)", "j-v", "jv curve"]):
            return "iv"
        elif any(kw in text_lower for kw in ["c-v", "cv curve", "capacitance", "frequency", "1/c²"]):
            return "cv"
        elif any(kw in text_lower for kw in ["transfer", "output curve", "ids", "vgs", "vds", "mobility"]):
            return "transfer"
        elif any(kw in text_lower for kw in ["kinetics", "pseudo-first", "pseudo-second", "langmuir", "freundlich"]):
            return "kinetics"
        elif any(kw in text_lower for kw in ["tga", "thermogravimetric", "weight loss", "dtg"]):
            return "tga"
        elif any(kw in text_lower for kw in ["dsc", "differential scanning", "heat flow", "exotherm", "endotherm"]):
            return "dsc"
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            horizontal = sum(1 for l in lines if abs(l[0][1] - l[0][3]) < 5)
            vertical = sum(1 for l in lines if abs(l[0][0] - l[0][2]) < 5)
            
            if horizontal > 5 and vertical > 5:
                return "xy_plot"
        
        return "unknown"
    
    def _extract_text_from_image(self, gray: np.ndarray) -> str:
        try:
            import pytesseract
            return pytesseract.image_to_string(gray)
        except:
            return ""
    
    def _extract_curve_data(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        points = largest_contour.reshape(-1, 2)
        
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        
        sorted_indices = np.argsort(x_vals)
        x_vals = x_vals[sorted_indices]
        y_vals = y_vals[sorted_indices]
        
        unique_x, indices = np.unique(x_vals, return_index=True)
        y_vals = y_vals[indices]
        
        y_vals = np.max(y_vals) - y_vals
        
        return unique_x, y_vals
    
    def _analyze_xrd_pattern(self, image: np.ndarray) -> List[GraphFinding]:
        findings = []
        
        curve_data = self._extract_curve_data(image)
        if curve_data is None:
            return findings
        
        x_vals, y_vals = curve_data
        
        if len(x_vals) < 10:
            return findings
        
        y_smooth = savgol_filter(y_vals, min(51, len(y_vals)//2*2+1), 3)
        
        peaks, properties = find_peaks(y_smooth, distance=10, prominence=np.max(y_smooth)*0.05)
        
        if len(peaks) == 0:
            findings.append(GraphFinding(
                graph_type="xrd",
                severity="high",
                description="XRD pattern shows no detectable diffraction peaks - possibly amorphous or corrupted data",
                evidence={"peak_count": 0, "max_intensity": float(np.max(y_vals))},
                confidence=0.7,
                bbox=(0, 0, image.shape[1], image.shape[0])
            ))
            return findings
        
        peak_positions = x_vals[peaks]
        peak_intensities = y_smooth[peaks]
        
        if len(peaks) > 1:
            sorted_peaks = np.sort(peak_positions)
            spacings = np.diff(sorted_peaks)
            
            if len(spacings) > 2 and np.std(spacings) < 0.02 * np.mean(spacings):
                findings.append(GraphFinding(
                    graph_type="xrd",
                    severity="high",
                    description="XRD peaks show suspiciously uniform spacing - possible synthetic/fabricated pattern",
                    evidence={
                        "peak_count": len(peaks),
                        "mean_spacing": float(np.mean(spacings)),
                        "spacing_std": float(np.std(spacings)),
                        "uniformity_ratio": float(np.std(spacings) / (np.mean(spacings) + 1e-6))
                    },
                    confidence=0.75,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        negative_intensity = np.sum(y_vals < -0.01 * np.max(y_vals))
        if negative_intensity > len(y_vals) * 0.05:
            findings.append(GraphFinding(
                graph_type="xrd",
                severity="critical",
                description=f"XRD pattern contains {negative_intensity} negative intensity points - physically impossible",
                evidence={"negative_point_count": int(negative_intensity), "total_points": len(y_vals)},
                confidence=0.95,
                bbox=(0, 0, image.shape[1], image.shape[0])
            ))
        
        if len(peaks) >= 2:
            ratios = peak_intensities / np.max(peak_intensities)
            if np.all(ratios > 0.95):
                findings.append(GraphFinding(
                    graph_type="xrd",
                    severity="high",
                    description="All XRD peaks have nearly identical intensity - unlikely for real crystalline materials",
                    evidence={"peak_intensity_ratios": ratios.tolist()},
                    confidence=0.7,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        return findings
    
    def _analyze_microscopy_image(self, image: np.ndarray) -> List[GraphFinding]:
        findings = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        
        h, w = magnitude.shape
        center = (h // 2, w // 2)
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        rings = cv2.HoughCircles(
            (magnitude / magnitude.max() * 255).astype(np.uint8),
            cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=min(h, w)//3
        )
        
        if rings is not None and len(rings[0]) > 3:
            findings.append(GraphFinding(
                graph_type="sem",
                severity="medium",
                description="FFT shows multiple diffraction rings - possible polycrystalline or contaminated sample",
                evidence={"ring_count": len(rings[0])},
                confidence=0.6,
                bbox=(0, 0, image.shape[1], image.shape[0])
            ))
        
        return findings
    
    def _analyze_spectroscopy(self, image: np.ndarray, spec_type: str) -> List[GraphFinding]:
        findings = []
        
        curve_data = self._extract_curve_data(image)
        if curve_data is None:
            return findings
        
        x_vals, y_vals = curve_data
        
        if spec_type in ["uvvis", "absorption"]:
            if np.any(y_vals < -0.05 * np.max(y_vals)):
                findings.append(GraphFinding(
                    graph_type=spec_type,
                    severity="critical",
                    description="Absorbance spectrum contains negative values - physically impossible",
                    evidence={"min_value": float(np.min(y_vals)), "negative_count": int(np.sum(y_vals < 0))},
                    confidence=0.95,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
            
            if np.max(y_vals) > 5:
                findings.append(GraphFinding(
                    graph_type=spec_type,
                    severity="medium",
                    description="Absorbance values exceed 5 AU - likely saturated detector or incorrect baseline",
                    evidence={"max_absorbance": float(np.max(y_vals))},
                    confidence=0.7,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        elif spec_type == "pl":
            peaks, _ = find_peaks(y_vals, distance=20, prominence=np.max(y_vals)*0.1)
            if len(peaks) == 0:
                findings.append(GraphFinding(
                    graph_type=spec_type,
                    severity="medium",
                    description="PL spectrum shows no emission peaks - possible quenching or measurement error",
                    evidence={"peak_count": 0},
                    confidence=0.6,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        return findings
    
    def _analyze_electrical_characteristics(self, image: np.ndarray) -> List[GraphFinding]:
        findings = []
        
        curve_data = self._extract_curve_data(image)
        if curve_data is None:
            return findings
        
        x_vals, y_vals = curve_data
        
        if len(x_vals) < 5:
            return findings
        
        if np.all(y_vals >= 0) and np.all(x_vals >= 0):
            pass
        elif np.any(y_vals < 0) and np.any(y_vals > 0):
            zero_crossings = np.where(np.diff(np.sign(y_vals)))[0]
            if len(zero_crossings) > 2:
                findings.append(GraphFinding(
                    graph_type="electrical",
                    severity="medium",
                    description="I-V curve shows multiple zero crossings - unusual for simple devices",
                    evidence={"zero_crossings": len(zero_crossings)},
                    confidence=0.6,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        return findings
    
    def _analyze_kinetic_thermal(self, image: np.ndarray, chart_type: str) -> List[GraphFinding]:
        findings = []
        
        curve_data = self._extract_curve_data(image)
        if curve_data is None:
            return findings
        
        x_vals, y_vals = curve_data
        
        if chart_type in ["kinetics", "degradation"] and np.any(y_vals > 105):
            findings.append(GraphFinding(
                graph_type=chart_type,
                severity="critical",
                description="Degradation/removal efficiency exceeds 100% - impossible",
                evidence={"max_efficiency": float(np.max(y_vals))},
                confidence=0.99,
                bbox=(0, 0, image.shape[1], image.shape[0])
            ))
        
        if chart_type == "tga":
            if np.any(np.diff(y_vals) > 0):
                findings.append(GraphFinding(
                    graph_type=chart_type,
                    severity="high",
                    description="TGA shows weight gain during heating - possible oxidation or buoyancy effect not accounted for",
                    evidence={"weight_gain_points": int(np.sum(np.diff(y_vals) > 0))},
                    confidence=0.7,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        return findings
    
    def _analyze_generic_chart(self, image: np.ndarray) -> List[GraphFinding]:
        findings = []
        
        curve_data = self._extract_curve_data(image)
        if curve_data is None:
            return findings
        
        x_vals, y_vals = curve_data
        
        if len(x_vals) < 5:
            return findings
        
        if len(x_vals) > 10:
            correlation_matrix = np.corrcoef([x_vals, y_vals])[0, 1]
            if abs(correlation_matrix) > 0.999:
                findings.append(GraphFinding(
                    graph_type="generic",
                    severity="high",
                    description="Data points show perfect linear correlation (|r| > 0.999) - likely fabricated or over-fitted",
                    evidence={"correlation": float(correlation_matrix)},
                    confidence=0.8,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        return findings
    
    def _check_repeating_patterns(self, image: np.ndarray) -> List[GraphFinding]:
        findings = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        if h < 100 or w < 100:
            return findings
        
        template_size = min(64, h // 4, w // 4)
        if template_size < 16:
            return findings
        
        step = template_size // 2
        matches = []
        
        for y in range(0, h - template_size, step):
            for x in range(0, w - template_size, step):
                template = gray[y:y+template_size, x:x+template_size]
                
                if np.std(template) < 5:
                    continue
                
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.95 and (max_loc[0] != x or max_loc[1] != y):
                    dist = np.sqrt((max_loc[0] - x)**2 + (max_loc[1] - y)**2)
                    if dist > template_size:
                        matches.append((max_loc, dist, max_val))
        
        if len(matches) > 3:
            findings.append(GraphFinding(
                graph_type="pattern",
                severity="high",
                description=f"Found {len(matches)} repeating pattern instances - possible copy-paste manipulation",
                evidence={"match_count": len(matches), "avg_distance": float(np.mean([m[1] for m in matches]))},
                confidence=0.75,
                bbox=(0, 0, w, h)
            ))
        
        return findings
    
    def _check_impossible_values(self, image: np.ndarray, chart_type: str) -> List[GraphFinding]:
        findings = []
        
        curve_data = self._extract_curve_data(image)
        if curve_data is None:
            return findings
        
        x_vals, y_vals = curve_data
        
        if chart_type == "xrd":
            if np.any(y_vals < 0):
                findings.append(GraphFinding(
                    graph_type="xrd",
                    severity="critical",
                    description="XRD intensity cannot be negative",
                    evidence={"negative_count": int(np.sum(y_vals < 0))},
                    confidence=0.99,
                    bbox=(0, 0, image.shape[1], image.shape[0])
                ))
        
        return findings


def analyze_graph_chart(image: Image.Image, domain: str = "materials_science") -> List[GraphFinding]:
    analyzer = GraphChartAnalyzer(domain)
    return analyzer.analyze(image)