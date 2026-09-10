import io
import fitz  # PyMuPDF
from PIL import Image
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_figures_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    '''
    Extract embedded figures/images from PDF.
    Returns list of dicts with: image (PIL Image), page_num, bbox, caption
    '''
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    figures = []

    for page_idx, page in enumerate(doc):
        # Get embedded images
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image['image']
                pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # Get image bbox on page
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else None

                # Try to find caption near image
                caption = _find_caption_near_image(page, bbox) if bbox else ''

                figures.append({
                    'image': pil_img,
                    'page_num': page_idx + 1,
                    'bbox': {'x0': bbox.x0, 'y0': bbox.y0, 'x1': bbox.x1, 'y1': bbox.y1} if bbox else {},
                    'caption': caption,
                })
            except Exception as e:
                logger.warning(f'Failed to extract image xref {xref}: {e}')
                continue

    doc.close()
    return figures


def extract_pages_as_images(pdf_bytes: bytes, dpi: int = 150) -> List[Dict[str, Any]]:
    '''Fallback: render entire pages as images'''
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    pages = []

    for page_idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        pil_img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

        # Extract text for caption detection
        text = page.get_text()
        caption = _extract_figure_captions(text)

        pages.append({
            'image': pil_img,
            'page_num': page_idx + 1,
            'bbox': {'x0': 0, 'y0': 0, 'x1': pix.width, 'y1': pix.height},
            'caption': caption,
        })

    doc.close()
    return pages


def _find_caption_near_image(page, bbox, max_distance: float = 100) -> str:
    '''Find figure caption text near an image bbox'''
    # Search below the image
    search_rect = fitz.Rect(bbox.x0, bbox.y1, bbox.x1, bbox.y1 + max_distance)
    text = page.get_text('text', clip=search_rect)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ['figure', 'fig.', 'fig ', 'image', 'chart', 'graph']):
            return line[:200]
    return ''


def _extract_figure_captions(text: str) -> str:
    '''Extract figure captions from page text'''
    captions = []
    for line in text.split('\n'):
        line = line.strip()
        lower = line.lower()
        if any(kw in lower for kw in ['figure', 'fig.', 'fig ']) and len(line) > 5:
            captions.append(line[:200])
    return ' | '.join(captions[:3])
