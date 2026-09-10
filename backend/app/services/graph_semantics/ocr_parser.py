from PIL import Image
from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

_easyocr_reader = None


def get_easyocr_reader(languages: str = 'en,hi'):
    '''Get EasyOCR reader (singleton)'''
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            lang_list = [l.strip() for l in languages.split(',')]
            _easyocr_reader = easyocr.Reader(lang_list, gpu=False)
            logger.info(f'EasyOCR initialized with languages: {lang_list}')
        except Exception as e:
            logger.warning(f'EasyOCR not available: {e}')
            _easyocr_reader = False
    return _easyocr_reader


def extract_graph_data(image: Image.Image, languages: str = 'en,hi') -> Dict[str, Any]:
    '''
    Extract structured data from graph/chart image using OCR.
    Returns: chart_type, axes, series, legend, confidence
    '''
    reader = get_easyocr_reader(languages)
    if not reader:
        return _fallback_tesseract(image)

    try:
        # Convert to numpy for EasyOCR
        img_array = np.array(image)

        # Run OCR
        results = reader.readtext(img_array)

        # Parse OCR results
        text_blocks = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and text.strip():
                text_blocks.append({
                    'text': text.strip(),
                    'confidence': conf,
                    'bbox': bbox,
                })

        # Classify chart type and extract structured data
        chart_data = _parse_chart_data(text_blocks, image.size)

        return chart_data

    except Exception as e:
        logger.error(f'EasyOCR failed: {e}')
        return _fallback_tesseract(image)


def _fallback_tesseract(image: Image.Image) -> Dict[str, Any]:
    '''Fallback to Tesseract OCR'''
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        return _parse_chart_data_from_text(text, image.size)
    except Exception:
        return {'chart_type': 'unknown', 'axes': {}, 'series': [], 'legend': [], 'confidence': 0.0}


def _parse_chart_data(text_blocks: List[Dict], image_size: tuple) -> Dict[str, Any]:
    '''Parse OCR text blocks into structured chart data'''
    all_text = ' '.join([b['text'] for b in text_blocks])

    # Detect chart type from keywords
    chart_type = _detect_chart_type(all_text)

    # Extract axes info
    axes = _extract_axes(text_blocks)

    # Extract data series
    series = _extract_series(text_blocks)

    # Extract legend
    legend = _extract_legend(text_blocks)

    return {
        'chart_type': chart_type,
        'axes': axes,
        'series': series,
        'legend': legend,
        'raw_text': all_text,
        'confidence': np.mean([b['confidence'] for b in text_blocks]) if text_blocks else 0.0,
    }


def _parse_chart_data_from_text(text: str, image_size: tuple) -> Dict[str, Any]:
    '''Parse chart data from plain text'''
    return _parse_chart_data(
        [{'text': line, 'confidence': 0.5, 'bbox': []} for line in text.split('\n') if line.strip()],
        image_size
    )


def _detect_chart_type(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['xrd', '2?', 'diffraction', 'intensity']):
        return 'xrd'
    elif any(kw in text_lower for kw in ['sem', 'tem', 'magnification', 'µm', 'scale']):
        return 'microscopy'
    elif any(kw in text_lower for kw in ['uv', 'vis', 'absorption', 'wavelength', 'absorbance']):
        return 'uvvis'
    elif any(kw in text_lower for kw in ['raman', 'cm-1', 'shift']):
        return 'raman'
    elif any(kw in text_lower for kw in ['i-v', 'iv', 'current', 'voltage']):
        return 'iv'
    elif any(kw in text_lower for kw in ['c-v', 'capacitance']):
        return 'cv'
    elif any(kw in text_lower for kw in ['tga', 'weight loss', 'dtg']):
        return 'tga'
    elif any(kw in text_lower for kw in ['dsc', 'heat flow', 'exotherm']):
        return 'dsc'
    return 'unknown'


def _extract_axes(text_blocks: List[Dict]) -> Dict[str, Any]:
    '''Extract axis labels and units from text blocks'''
    axes = {'x': {}, 'y': {}}
    for block in text_blocks:
        text = block['text'].lower()
        # Heuristic: text near bottom/horizontal = x-axis, left/vertical = y-axis
        # This is simplified - real implementation would use bbox positions
        if any(kw in text for kw in ['x-axis', 'x axis', 'time', 'wavelength', '2?', 'voltage']):
            axes['x']['label'] = block['text']
        elif any(kw in text for kw in ['y-axis', 'y axis', 'intensity', 'absorbance', 'current']):
            axes['y']['label'] = block['text']
    return axes


def _extract_series(text_blocks: List[Dict]) -> List[Dict]:
    '''Extract data series from text blocks'''
    series = []
    for block in text_blocks:
        # Look for numeric patterns that could be data points
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', block['text'])
        if len(numbers) >= 3:
            series.append({
                'label': block['text'][:50],
                'values': [float(n) for n in numbers[:20]],
            })
    return series[:5]


def _extract_legend(text_blocks: List[Dict]) -> List[str]:
    '''Extract legend entries'''
    legend = []
    for block in text_blocks:
        text = block['text'].lower()
        if any(kw in text for kw in ['legend', 'series', 'control', 'treatment', 'sample']):
            legend.append(block['text'])
    return legend[:10]


def compare_graph_semantics(data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
    '''Compare two graph semantic data structures'''
    result = {
        'same_chart_type': data1.get('chart_type') == data2.get('chart_type'),
        'axes_match': _compare_axes(data1.get('axes', {}), data2.get('axes', {})),
        'series_similarity': _compare_series(data1.get('series', []), data2.get('series', [])),
        'legend_overlap': _compare_legend(data1.get('legend', []), data2.get('legend', [])),
    }
    result['overall_match'] = (
        (1.0 if result['same_chart_type'] else 0.0) * 0.2 +
        result['axes_match'] * 0.3 +
        result['series_similarity'] * 0.3 +
        result['legend_overlap'] * 0.2
    )
    return result


def _compare_axes(axes1: Dict, axes2: Dict) -> float:
    if not axes1 or not axes2:
        return 0.0
    matches = 0
    total = 0
    for axis in ['x', 'y']:
        if axis in axes1 and axis in axes2:
            total += 1
            label1 = axes1[axis].get('label', '').lower()
            label2 = axes2[axis].get('label', '').lower()
            if label1 and label2:
                # Simple string similarity
                from difflib import SequenceMatcher
                sim = SequenceMatcher(None, label1, label2).ratio()
                if sim > 0.7:
                    matches += 1
    return matches / total if total > 0 else 0.0


def _compare_series(series1: List, series2: List) -> float:
    if not series1 or not series2:
        return 0.0
    # Compare first series values
    s1 = series1[0].get('values', [])
    s2 = series2[0].get('values', [])
    if not s1 or not s2:
        return 0.0
    # Normalize and compare
    import numpy as np
    arr1 = np.array(s1[:len(s2)]) if len(s1) >= len(s2) else np.array(s1)
    arr2 = np.array(s2[:len(s1)]) if len(s2) >= len(s1) else np.array(s2)
    if len(arr1) != len(arr2):
        return 0.0
    # Correlation
    corr = np.corrcoef(arr1, arr2)[0, 1] if len(arr1) > 1 else 0.0
    return max(0.0, corr)


def _compare_legend(legend1: List, legend2: List) -> float:
    if not legend1 or not legend2:
        return 0.0
    set1 = set(l.lower() for l in legend1)
    set2 = set(l.lower() for l in legend2)
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0
