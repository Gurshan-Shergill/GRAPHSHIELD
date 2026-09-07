import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
from skimage.feature import match_template
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
import imagehash


@dataclass
class ForensicFinding:
    type: str
    severity: str
    description: str
    evidence: Dict
    confidence: float
    location: Tuple[int, int, int, int]


class ImageForensicsAnalyzer:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=2000)
    
    def analyze(self, image: Image.Image, all_images: List[Image.Image] = None, 
                index: int = 0) -> List[ForensicFinding]:
        findings = []
        cv_image = self._pil_to_cv(image)
        
        findings.extend(self._error_level_analysis(image))
        findings.extend(self._noise_analysis(cv_image))
        findings.extend(self._copy_move_detection(cv_image))
        findings.extend(self._inversion_detection(cv_image))
        findings.extend(self._color_consistency_check(cv_image))
        findings.extend(self._jpeg_artifact_analysis(image))
        findings.extend(self._metadata_check(image))
        
        if all_images:
            findings.extend(self._cross_image_comparison(image, all_images, index))
        
        return findings
    
    def _pil_to_cv(self, pil_img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _cv_to_pil(self, cv_img: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    def _error_level_analysis(self, image: Image.Image, quality: int = 90) -> List[ForensicFinding]:
        findings = []
        try:
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            recompressed = Image.open(buffer)
            
            diff = ImageChops.difference(image.convert('RGB'), recompressed.convert('RGB'))
            diff_array = np.array(diff)
            
            mean_diff = np.mean(diff_array)
            std_diff = np.std(diff_array)
            
            if std_diff > 15:
                findings.append(ForensicFinding(
                    type="ela_anomaly",
                    severity="high",
                    description="Error Level Analysis shows inconsistent compression artifacts - possible local manipulation",
                    evidence={
                        "mean_difference": float(mean_diff),
                        "std_difference": float(std_diff),
                        "threshold": 15
                    },
                    confidence=min(0.9, std_diff / 50),
                    location=(0, 0, image.width, image.height)
                ))
            
            ela_enhanced = ImageEnhance.Brightness(diff).enhance(20)
            ela_array = np.array(ela_enhanced.convert('L'))
            
            suspicious_regions = self._find_suspicious_regions(ela_array)
            for region in suspicious_regions:
                findings.append(ForensicFinding(
                    type="ela_local_manipulation",
                    severity="high",
                    description=f"ELA reveals suspicious region at {region}",
                    evidence={"region": region, "ela_intensity": float(np.mean(ela_array[region[1]:region[3], region[0]:region[2]]))},
                    confidence=0.75,
                    location=region
                ))
                
        except Exception as e:
            pass
        
        return findings
    
    def _find_suspicious_regions(self, ela_array: np.ndarray, block_size: int = 32) -> List[Tuple[int, int, int, int]]:
        regions = []
        h, w = ela_array.shape
        
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = ela_array[y:y+block_size, x:x+block_size]
                if np.std(block) > 30 and np.mean(block) > 40:
                    regions.append((x, y, x + block_size, y + block_size))
        
        return regions[:10]
    
    def _noise_analysis(self, image: np.ndarray) -> List[ForensicFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        noise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_residual = cv2.absdiff(gray, noise)
        
        noise_mean = np.mean(noise_residual)
        noise_std = np.std(noise_residual)
        
        h, w = gray.shape
        grid_size = 64
        noise_map = np.zeros((h // grid_size, w // grid_size))
        
        for i in range(0, h - grid_size, grid_size):
            for j in range(0, w - grid_size, grid_size):
                block = noise_residual[i:i+grid_size, j:j+grid_size]
                noise_map[i//grid_size, j//grid_size] = np.std(block)
        
        if noise_map.size > 0:
            noise_map_norm = (noise_map - np.mean(noise_map)) / (np.std(noise_map) + 1e-6)
            outliers = np.where(np.abs(noise_map_norm) > 2.5)
            
            if len(outliers[0]) > 0:
                findings.append(ForensicFinding(
                    type="noise_inconsistency",
                    severity="medium",
                    description=f"Found {len(outliers[0])} regions with inconsistent noise patterns - possible splicing",
                    evidence={
                        "outlier_count": int(len(outliers[0])),
                        "mean_noise": float(noise_mean),
                        "std_noise": float(noise_std)
                    },
                    confidence=0.7,
                    location=(0, 0, w, h)
                ))
        
        return findings
    
    def _copy_move_detection(self, image: np.ndarray) -> List[ForensicFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is not None and len(keypoints) > 10:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            suspicious_matches = []
            for match in matches:
                if match.distance < 0.1:
                    pt1 = keypoints[match.queryIdx].pt
                    pt2 = keypoints[match.trainIdx].pt
                    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    
                    if dist > 50:
                        suspicious_matches.append((pt1, pt2, dist))
            
            if len(suspicious_matches) > 5:
                findings.append(ForensicFinding(
                    type="copy_move_forgery",
                    severity="critical",
                    description=f"Detected {len(suspicious_matches)} matching keypoint pairs at different locations - strong evidence of copy-move forgery",
                    evidence={
                        "match_count": len(suspicious_matches),
                        "example_distance": suspicious_matches[0][2] if suspicious_matches else 0
                    },
                    confidence=0.85,
                    location=(int(suspicious_matches[0][0][0]), int(suspicious_matches[0][0][1]),
                             int(suspicious_matches[0][1][0]), int(suspicious_matches[0][1][1]))
                ))
        
        return findings
    
    def _inversion_detection(self, image: np.ndarray) -> List[ForensicFinding]:
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        inverted = 255 - gray
        
        corr = cv2.matchTemplate(gray, inverted, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)
        
        if max_val > 0.7:
            h, w = gray.shape
            findings.append(ForensicFinding(
                type="inversion_detected",
                severity="high",
                description="Image contains regions that match their inverted version - possible negative/inverted reuse",
                evidence={
                    "correlation": float(max_val),
                    "location": max_loc
                },
                confidence=max_val,
                location=(max_loc[0], max_loc[1], max_loc[0] + w//4, max_loc[1] + h//4)
            ))
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_inv = cv2.calcHist([inverted], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist, hist_inv, cv2.HISTCMP_CORREL)
        
        if hist_corr > 0.85:
            findings.append(ForensicFinding(
                type="global_inversion_similarity",
                severity="medium",
                description="Global histogram matches inverted version - image may be inverted or contain inverted regions",
                evidence={"histogram_correlation": float(hist_corr)},
                confidence=hist_corr,
                location=(0, 0, gray.shape[1], gray.shape[0])
            ))
        
        return findings
    
    def _color_consistency_check(self, image: np.ndarray) -> List[ForensicFinding]:
        findings = []
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        h, w = l_channel.shape
        grid = 8
        means = []
        
        for i in range(grid):
            for j in range(grid):
                y1, y2 = i * h // grid, (i + 1) * h // grid
                x1, x2 = j * w // grid, (j + 1) * w // grid
                block = l_channel[y1:y2, x1:x2]
                means.append(np.mean(block))
        
        means = np.array(means)
        if np.std(means) > 25:
            findings.append(ForensicFinding(
                type="illumination_inconsistency",
                severity="medium",
                description="Significant illumination variation across image regions - possible composite",
                evidence={"illumination_std": float(np.std(means)), "block_means": means.tolist()},
                confidence=0.65,
                location=(0, 0, w, h)
            ))
        
        return findings
    
    def _jpeg_artifact_analysis(self, image: Image.Image) -> List[ForensicFinding]:
        findings = []
        
        try:
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                if exif:
                    software = exif.get(305, "")
                    if software and any(s in software.lower() for s in ['photoshop', 'gimp', 'editor', 'paint']):
                        findings.append(ForensicFinding(
                            type="editing_software_detected",
                            severity="low",
                            description=f"Image edited with {software}",
                            evidence={"software": software},
                            confidence=0.9,
                            location=(0, 0, image.width, image.height)
                        ))
        except Exception:
            pass
        
        return findings
    
    def _metadata_check(self, image: Image.Image) -> List[ForensicFinding]:
        findings = []
        return findings
    
    def _cross_image_comparison(self, image: Image.Image, 
                                  all_images: List[Image.Image], index: int) -> List[ForensicFinding]:
        findings = []
        
        img_hash = imagehash.phash(image)
        img_dhash = imagehash.dhash(image)
        
        for i, other_img in enumerate(all_images):
            if i == index:
                continue
            
            other_hash = imagehash.phash(other_img)
            other_dhash = imagehash.dhash(other_img)
            
            p_dist = img_hash - other_hash
            d_dist = img_dhash - other_dhash
            avg_dist = (p_dist + d_dist) / 2
            
            if avg_dist <= 5:
                findings.append(ForensicFinding(
                    type="duplicate_figure",
                    severity="critical",
                    description=f"Figure {index+1} is nearly identical to Figure {i+1} (pHash dist: {p_dist}, dHash dist: {d_dist}) - likely duplicated",
                    evidence={
                        "matched_figure": i + 1,
                        "phash_distance": int(p_dist),
                        "dhash_distance": int(d_dist),
                        "avg_distance": float(avg_dist)
                    },
                    confidence=0.95,
                    location=(0, 0, image.width, image.height)
                ))
            elif avg_dist <= 12:
                cv_img1 = self._pil_to_cv(image)
                cv_img2 = self._pil_to_cv(other_img)
                
                gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                
                gray1_resized = cv2.resize(gray1, (256, 256))
                gray2_resized = cv2.resize(gray2, (256, 256))
                
                ssim_score, _ = ssim(gray1_resized, gray2_resized, full=True)
                
                if ssim_score > 0.85:
                    findings.append(ForensicFinding(
                        type="highly_similar_figure",
                        severity="high",
                        description=f"Figure {index+1} highly similar to Figure {i+1} (SSIM: {ssim_score:.3f}) - possible reuse with minor modifications",
                        evidence={
                            "matched_figure": i + 1,
                            "ssim_score": float(ssim_score),
                            "phash_distance": int(p_dist),
                            "dhash_distance": int(d_dist)
                        },
                        confidence=ssim_score,
                        location=(0, 0, image.width, image.height)
                    ))
                
                inverted2 = 255 - gray2_resized
                ssim_inv, _ = ssim(gray1_resized, inverted2, full=True)
                
                if ssim_inv > 0.8:
                    findings.append(ForensicFinding(
                        type="inverted_reuse",
                        severity="critical",
                        description=f"Figure {index+1} matches inverted version of Figure {i+1} (SSIM: {ssim_inv:.3f}) - inverted reuse detected",
                        evidence={
                            "matched_figure": i + 1,
                            "ssim_inverted": float(ssim_inv),
                            "phash_distance": int(p_dist)
                        },
                        confidence=ssim_inv,
                        location=(0, 0, image.width, image.height)
                    ))
                
                flipped_h = cv2.flip(gray2_resized, 1)
                ssim_fliph, _ = ssim(gray1_resized, flipped_h, full=True)
                
                flipped_v = cv2.flip(gray2_resized, 0)
                ssim_flipv, _ = ssim(gray1_resized, flipped_v, full=True)
                
                if ssim_fliph > 0.85 or ssim_flipv > 0.85:
                    flip_type = "horizontal" if ssim_fliph > ssim_flipv else "vertical"
                    findings.append(ForensicFinding(
                        type="flipped_reuse",
                        severity="high",
                        description=f"Figure {index+1} matches {flip_type} flipped version of Figure {i+1} - flipped reuse detected",
                        evidence={
                            "matched_figure": i + 1,
                            "flip_type": flip_type,
                            "ssim_score": float(max(ssim_fliph, ssim_flipv))
                        },
                        confidence=max(ssim_fliph, ssim_flipv),
                        location=(0, 0, image.width, image.height)
                    ))
                
                rotated_90 = cv2.rotate(gray2_resized, cv2.ROTATE_90_CLOCKWISE)
                rotated_180 = cv2.rotate(gray2_resized, cv2.ROTATE_180)
                rotated_270 = cv2.rotate(gray2_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                for rot_name, rot_img in [("90°", rotated_90), ("180°", rotated_180), ("270°", rotated_270)]:
                    ssim_rot, _ = ssim(gray1_resized, rot_img, full=True)
                    if ssim_rot > 0.85:
                        findings.append(ForensicFinding(
                            type="rotated_reuse",
                            severity="high",
                            description=f"Figure {index+1} matches {rot_name} rotated version of Figure {i+1} - rotated reuse detected",
                            evidence={
                                "matched_figure": i + 1,
                                "rotation": rot_name,
                                "ssim_score": float(ssim_rot)
                            },
                            confidence=ssim_rot,
                            location=(0, 0, image.width, image.height)
                        ))
        
        return findings


def analyze_image_forensics(image: Image.Image, all_images: List[Image.Image] = None, 
                            index: int = 0) -> List[ForensicFinding]:
    analyzer = ImageForensicsAnalyzer()
    return analyzer.analyze(image, all_images, index)