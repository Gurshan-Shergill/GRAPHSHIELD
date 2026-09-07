import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


@dataclass
class AmbiguityFinding:
    type: str
    severity: str
    description: str
    evidence: Dict
    confidence: float
    figure_index: int


class PictorialAmbiguityDetector:
    def __init__(self, domain: str, paper_text: str = ""):
        self.domain = domain
        self.paper_text = paper_text
        self.domain_context = self._get_domain_context()
        self._init_models()
    
    def _init_models(self):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.clip_available = False
        if OPEN_CLIP_AVAILABLE:
            try:
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai', device='cpu'
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.clip_available = True
            except Exception:
                self.clip_available = False
    
    def _get_domain_context(self) -> Dict:
        contexts = {
            "materials_science": {
                "expected_figures": ["XRD patterns", "SEM/TEM micrographs", "UV-Vis spectra", "PL spectra", 
                                   "Raman spectra", "XPS spectra", "VSM hysteresis loops", "EIS Nyquist plots",
                                   "band structure diagrams", "crystal structure models"],
                "unexpected_figures": ["western blots", "PCR gels", "cell culture images", "histology slides",
                                     "animal models", "clinical trial data"],
                "visual_features": {
                    "xrd": ["peaks", "2theta axis", "intensity", "miller indices"],
                    "sem": ["scale bar", "magnification", "morphology", "particles", "grains"],
                    "tem": ["lattice fringes", "d-spacing", "selected area diffraction", "scale bar"],
                    "uvvis": ["wavelength nm", "absorbance", "band gap", "tauc plot"],
                    "raman": ["raman shift cm-1", "intensity", "phonon modes"],
                }
            },
            "chemistry": {
                "expected_figures": ["FTIR spectra", "NMR spectra", "UV-Vis spectra", "MS spectra",
                                   "TGA/DSC curves", "chromatograms", "reaction schemes"],
                "unexpected_figures": ["XRD patterns", "SEM images", "hysteresis loops", "cell images"],
                "visual_features": {
                    "ftir": ["wavenumber cm-1", "transmittance", "functional groups"],
                    "nmr": ["chemical shift ppm", "integration", "multiplicity"],
                    "chromatogram": ["retention time", "peak area", "baseline"],
                }
            },
            "biology": {
                "expected_figures": ["western blots", "PCR gels", "microscopy images", "flow cytometry plots",
                                   "cell viability assays", "protein structures", "sequence alignments"],
                "unexpected_figures": ["XRD patterns", "hysteresis loops", "band structures", "EIS plots"],
                "visual_features": {
                    "western": ["molecular weight kda", "bands", "loading control", "antibody"],
                    "pcr": ["cycles", "fluorescence", "ct value", "amplification curve"],
                    "microscopy": ["scale bar", "cell morphology", "staining", "magnification"],
                }
            }
        }
        return contexts.get(self.domain, {})
    
    def analyze_figures(self, images: List[Image.Image], figure_captions: List[str] = None) -> List[AmbiguityFinding]:
        findings = []
        
        for idx, image in enumerate(images):
            caption = figure_captions[idx] if figure_captions and idx < len(figure_captions) else ""
            
            findings.extend(self._check_domain_consistency(image, caption, idx))
            findings.extend(self._check_caption_image_match(image, caption, idx))
            findings.extend(self._check_scale_bar_presence(image, caption, idx))
            findings.extend(self._check_image_quality(image, idx))
            findings.extend(self._check_unexpected_content(image, caption, idx))
        
        findings.extend(self._check_cross_figure_consistency(images, figure_captions))
        
        return findings
    
    def _check_domain_consistency(self, image: Image.Image, caption: str, idx: int) -> List[AmbiguityFinding]:
        findings = []
        
        if not self.clip_available:
            return findings
        
        expected = self.domain_context.get("expected_figures", [])
        unexpected = self.domain_context.get("unexpected_figures", [])
        
        all_categories = expected + unexpected
        if not all_categories:
            return findings
        
        image_input = self.clip_preprocess(image).unsqueeze(0)
        text_inputs = self.tokenizer(all_categories)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()
        
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        predicted_category = all_categories[max_idx]
        
        is_expected = predicted_category in expected
        
        if not is_expected and max_prob > 0.3:
            findings.append(AmbiguityFinding(
                type="domain_mismatch",
                severity="high",
                description=f"Figure {idx+1} appears to be '{predicted_category}' (confidence: {max_prob:.2f}) which is unexpected for {self.domain} research",
                evidence={
                    "predicted_category": predicted_category,
                    "confidence": float(max_prob),
                    "domain": self.domain,
                    "expected_categories": expected[:5]
                },
                confidence=max_prob,
                figure_index=idx
            ))
        
        if is_expected and max_prob < 0.2:
            findings.append(AmbiguityFinding(
                type="low_domain_confidence",
                severity="medium",
                description=f"Figure {idx+1} doesn't clearly match expected {self.domain} figure types (max confidence: {max_prob:.2f})",
                evidence={
                    "predicted_category": predicted_category,
                    "confidence": float(max_prob),
                    "top_3": [(all_categories[i], float(probs[i])) for i in np.argsort(probs)[-3:][::-1]]
                },
                confidence=1 - max_prob,
                figure_index=idx
            ))
        
        return findings
    
    def _check_caption_image_match(self, image: Image.Image, caption: str, idx: int) -> List[AmbiguityFinding]:
        findings = []
        
        if not caption or len(caption) < 10:
            return findings
        
        if not self.clip_available:
            return findings
        
        caption_emb = self.text_model.encode(caption)
        
        image_input = self.clip_preprocess(image).unsqueeze(0)
        text_input = self.tokenizer([caption])
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        
        if similarity < 0.15:
            findings.append(AmbiguityFinding(
                type="caption_mismatch",
                severity="high",
                description=f"Figure {idx+1} caption doesn't match image content (CLIP similarity: {similarity:.3f})",
                evidence={
                    "similarity": float(similarity),
                    "caption": caption[:200],
                    "threshold": 0.15
                },
                confidence=1 - similarity,
                figure_index=idx
            ))
        
        return findings
    
    def _check_scale_bar_presence(self, image: Image.Image, caption: str, idx: int) -> List[AmbiguityFinding]:
        findings = []
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        has_scale_bar_text = any(kw in caption.lower() for kw in ["scale bar", "scale:", "µm", "um", "nm", "magnification", "×", "x "])
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
        
        has_visual_scale_bar = False
        if lines is not None:
            horizontal_lines = [l for l in lines if abs(l[0][1] - l[0][3]) < 3]
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                length = abs(x2 - x1)
                if 30 < length < gray.shape[1] * 0.5:
                    roi = gray[max(0, y1-10):y1+10, x1:x2]
                    if roi.size > 0:
                        text_check = self._has_text_nearby(cv_image, x1, y1, length)
                        if text_check:
                            has_visual_scale_bar = True
                            break
        
        if has_scale_bar_text and not has_visual_scale_bar:
            findings.append(AmbiguityFinding(
                type="missing_scale_bar",
                severity="medium",
                description=f"Figure {idx+1} caption mentions scale bar but no visual scale bar detected",
                evidence={"caption_mentions_scale": True, "visual_scale_detected": False},
                confidence=0.7,
                figure_index=idx
            ))
        elif not has_scale_bar_text and has_visual_scale_bar:
            findings.append(AmbiguityFinding(
                type="unlabeled_scale_bar",
                severity="low",
                description=f"Figure {idx+1} has visual scale bar but caption doesn't mention it",
                evidence={"caption_mentions_scale": False, "visual_scale_detected": True},
                confidence=0.5,
                figure_index=idx
            ))
        
        return findings
    
    def _has_text_nearby(self, image: np.ndarray, x: int, y: int, length: int) -> bool:
        roi_y1 = max(0, y - 30)
        roi_y2 = min(image.shape[0], y + 30)
        roi_x1 = max(0, x - 10)
        roi_x2 = min(image.shape[1], x + length + 10)
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return False
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 500:
                x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                if h_c > 8 and w_c > 5:
                    return True
        return False
    
    def _check_image_quality(self, image: Image.Image, idx: int) -> List[AmbiguityFinding]:
        findings = []
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            findings.append(AmbiguityFinding(
                type="blurry_image",
                severity="medium",
                description=f"Figure {idx+1} appears blurry (Laplacian variance: {laplacian_var:.1f}) - possible low quality or intentional obfuscation",
                evidence={"laplacian_variance": float(laplacian_var), "threshold": 50},
                confidence=0.7,
                figure_index=idx
            ))
        
        mean_brightness = np.mean(gray)
        if mean_brightness < 20 or mean_brightness > 235:
            findings.append(AmbiguityFinding(
                type="exposure_issue",
                severity="low",
                description=f"Figure {idx+1} has extreme brightness (mean: {mean_brightness:.1f}) - possible over/under exposure",
                evidence={"mean_brightness": float(mean_brightness)},
                confidence=0.6,
                figure_index=idx
            ))
        
        return findings
    
    def _check_unexpected_content(self, image: Image.Image, caption: str, idx: int) -> List[AmbiguityFinding]:
        findings = []
        
        unexpected = self.domain_context.get("unexpected_figures", [])
        if not unexpected:
            return findings
        
        if not self.clip_available:
            return findings
        
        image_input = self.clip_preprocess(image).unsqueeze(0)
        text_inputs = self.tokenizer(unexpected)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()
        
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        
        if max_prob > 0.4:
            findings.append(AmbiguityFinding(
                type="unexpected_content",
                severity="high",
                description=f"Figure {idx+1} contains unexpected content for {self.domain}: '{unexpected[max_idx]}' (confidence: {max_prob:.2f})",
                evidence={
                    "detected_content": unexpected[max_idx],
                    "confidence": float(max_prob),
                    "domain": self.domain
                },
                confidence=max_prob,
                figure_index=idx
            ))
        
        return findings
    
    def _check_cross_figure_consistency(self, images: List[Image.Image], 
                                        captions: List[str] = None) -> List[AmbiguityFinding]:
        findings = []
        
        if len(images) < 2:
            return findings
        
        if not self.clip_available:
            return findings
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1_input = self.clip_preprocess(images[i]).unsqueeze(0)
                img2_input = self.clip_preprocess(images[j]).unsqueeze(0)
                
                with torch.no_grad():
                    feat1 = self.clip_model.encode_image(img1_input)
                    feat2 = self.clip_model.encode_image(img2_input)
                    
                    feat1 /= feat1.norm(dim=-1, keepdim=True)
                    feat2 /= feat2.norm(dim=-1, keepdim=True)
                    
                    similarity = (feat1 @ feat2.T).item()
                
                if similarity > 0.9:
                    cap1 = captions[i] if captions and i < len(captions) else f"Figure {i+1}"
                    cap2 = captions[j] if captions and j < len(captions) else f"Figure {j+1}"
                    
                    findings.append(AmbiguityFinding(
                        type="cross_figure_duplicate",
                        severity="critical",
                        description=f"Figures {i+1} and {j+1} are nearly identical (similarity: {similarity:.3f}) but have different captions",
                        evidence={
                            "figure_1": cap1[:100],
                            "figure_2": cap2[:100],
                            "similarity": float(similarity)
                        },
                        confidence=similarity,
                        figure_index=i
                    ))
        
        return findings


def detect_pictorial_ambiguity(images: List[Image.Image], domain: str, 
                               paper_text: str = "", captions: List[str] = None) -> List[AmbiguityFinding]:
    detector = PictorialAmbiguityDetector(domain, paper_text)
    return detector.analyze_figures(images, captions)