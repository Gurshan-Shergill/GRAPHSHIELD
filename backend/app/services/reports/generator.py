import os
import io
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML, CSS
from PIL import Image
import fitz

from app.core.config import get_settings
from app.core.storage import storage

settings = get_settings()

# Jinja2 environment
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=select_autoescape(['html', 'xml']),
)


def generate_ugc_report(
    paper_data: Dict[str, Any],
    figures_data: List[Dict[str, Any]],
    matches_data: List[Dict[str, Any]],
    findings: Dict[str, List[Dict[str, Any]]],
) -> bytes:
    '''Generate UGC-compliant HTML report and convert to PDF'''

    # Prepare template data
    template_data = {
        'paper': paper_data,
        'figures': figures_data,
        'matches': matches_data,
        'findings': findings,
        'summary': _compute_summary(paper_data, figures_data, matches_data, findings),
        'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        'version': settings.app_version,
    }

    # Render HTML
    template = env.get_template('ugc_report.html')
    html_content = template.render(**template_data)

    # Convert to PDF with WeasyPrint
    pdf_bytes = HTML(string=html_content).write_pdf(
        stylesheets=[CSS(string=_get_report_css())]
    )

    return pdf_bytes


def generate_deplagiarized_pdf(
    original_pdf_bytes: bytes,
    flagged_figures: List[Dict[str, Any]],
    threshold: float = 60.0
) -> bytes:
    '''Generate deplagiarized PDF with flagged figures removed/redacted'''
    doc = fitz.open(stream=original_pdf_bytes, filetype='pdf')
    new_doc = fitz.open()

    flagged_pages = set()
    for fig in flagged_figures:
        if fig.get('similarity', 0) >= threshold:
            page_idx = fig.get('page', 1) - 1
            if 0 <= page_idx < len(doc):
                flagged_pages.add(page_idx)

    for page_idx in range(len(doc)):
        if page_idx not in flagged_pages:
            new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        else:
            page = doc[page_idx]
            new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.show_pdf_page(page.rect, doc, page_idx)

            # Redact flagged figures
            flagged_figs = [f for f in flagged_figures if f.get('page', 1) - 1 == page_idx and f.get('similarity', 0) >= threshold]
            for fig in flagged_figs:
                try:
                    images = page.get_images(full=True)
                    for img in images:
                        xref = img[0]
                        img_rects = page.get_image_rects(xref)
                        for rect in img_rects:
                            new_page.add_redact_annot(rect, fill=(1, 1, 1))
                except Exception:
                    pass
            new_page.apply_redactions()

    output_bytes = new_doc.tobytes()
    new_doc.close()
    doc.close()
    return output_bytes


def _compute_summary(paper_data, figures_data, matches_data, findings) -> Dict[str, Any]:
    total_figures = len(figures_data)
    flagged_figures = sum(1 for f in figures_data if f.get('similarity', 0) >= 70)
    external_matches = sum(1 for f in figures_data if f.get('is_external_match', False))

    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for category_findings in findings.values():
        for finding in category_findings:
            severity_counts[finding.get('severity', 'low')] = severity_counts.get(finding.get('severity', 'low'), 0) + 1

    image_risk = 0.0 if total_figures == 0 else round((external_matches / total_figures) * 100, 1)
    text_risk = findings.get('text', {}).get('avg_similarity', 0.0)
    overall_risk = max(image_risk, text_risk)

    if overall_risk >= 50:
        risk_level = 'CRITICAL'
    elif overall_risk >= 25:
        risk_level = 'HIGH'
    elif overall_risk >= 10:
        risk_level = 'MODERATE'
    else:
        risk_level = 'LOW'

    return {
        'overall_similarity': overall_risk,
        'risk_level': risk_level,
        'total_figures': total_figures,
        'flagged_figures': flagged_figures,
        'external_matches': external_matches,
        'unique_figures': total_figures - flagged_figures,
        'severity_counts': severity_counts,
        'image_risk_score': image_risk,
        'text_risk_score': text_risk,
    }


def _get_report_css() -> str:
    return '''
    @page { size: A4; margin: 2.5cm; }
    body { font-family:  DejaVu Sans, Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.5; color: #1e293b; }
    .header { text-align: center; margin-bottom: 30px; border-bottom: 3px solid #2563eb; padding-bottom: 15px; }
    .title { font-size: 24pt; font-weight: bold; color: #0f172a; margin: 0; }
    .subtitle { font-size: 11pt; color: #64748b; margin: 5px 0 0; }
    .risk-card { display: table; width: 100%; border-collapse: collapse; margin: 20px 0; }
    .risk-cell { display: table-cell; width: 25%; padding: 20px; text-align: center; border: 1px solid #e2e8f0; vertical-align: middle; }
    .risk-critical { background: #fef2f2; border-color: #dc2626; }
    .risk-high { background: #fffbeb; border-color: #d97706; }
    .risk-moderate { background: #ecfeff; border-color: #06b6d4; }
    .risk-low { background: #f0fdf4; border-color: #16a34a; }
    .risk-value { font-size: 28pt; font-weight: bold; }
    .risk-label { font-size: 9pt; color: #94a3b8; text-transform: uppercase; }
    .section { margin: 30px 0; }
    .section-title { font-size: 13pt; font-weight: bold; color: #0f172a; border-bottom: 2px solid #2563eb; padding-bottom: 5px; margin-bottom: 15px; }
    .figure-table { width: 100%; border-collapse: collapse; font-size: 8pt; }
    .figure-table th { background: #0f172a; color: white; padding: 8px; text-align: left; }
    .figure-table td { padding: 6px 8px; border: 1px solid #e2e8f0; vertical-align: top; }
    .figure-table tr:nth-child(even) { background: #f8fafc; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 7pt; font-weight: bold; text-transform: uppercase; }
    .badge-critical { background: #fee2e2; color: #dc2626; }
    .badge-high { background: #fef3c7; color: #d97706; }
    .badge-moderate { background: #cffafe; color: #0891b2; }
    .badge-low { background: #dcfce7; color: #16a34a; }
    .badge-unique { background: #f1f5f9; color: #64748b; }
    .side-by-side { display: flex; gap: 10px; margin: 10px 0; }
    .side-by-side img { max-width: 48%; height: auto; border: 1px solid #e2e8f0; }
    .methodology { font-size: 8pt; color: #64748b; margin-top: 30px; }
    .methodology h4 { font-size: 9pt; color: #0f172a; margin: 15px 0 5px; }
    .footer { text-align: center; font-size: 7pt; color: #94a3b8; margin-top: 40px; padding-top: 15px; border-top: 1px solid #e2e8f0; }
    '''
