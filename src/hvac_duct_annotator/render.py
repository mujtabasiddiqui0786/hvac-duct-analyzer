from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np


@dataclass(slots=True)
class RenderedPage:
    doc: fitz.Document
    page: fitz.Page
    page_index: int
    dpi: int
    rotation: int
    pix: fitz.Pixmap
    image_bgr: np.ndarray
    drawing_bbox_px: tuple[int, int, int, int]
    px_per_pdf_point: float


def _auto_drawing_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    x0 = int(0.03 * w)
    y0 = int(0.05 * h)
    x1 = int(0.90 * w)
    y1 = int(0.70 * h)
    return x0, y0, x1, y1


def load_pdf_page(input_pdf: Path, page_index: int = 0, dpi: int = 300) -> RenderedPage:
    doc = fitz.open(str(input_pdf))
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = image[:, :, :3]
    image_bgr = image[:, :, ::-1].copy()
    drawing_bbox_px = _auto_drawing_bbox(image_bgr)
    return RenderedPage(
        doc=doc,
        page=page,
        page_index=page_index,
        dpi=dpi,
        rotation=page.rotation,
        pix=pix,
        image_bgr=image_bgr,
        drawing_bbox_px=drawing_bbox_px,
        px_per_pdf_point=zoom,
    )


def pdf_to_px(pt: tuple[float, float], px_per_pdf_point: float) -> tuple[float, float]:
    return pt[0] * px_per_pdf_point, pt[1] * px_per_pdf_point


def px_to_pdf(pt: tuple[float, float], px_per_pdf_point: float) -> tuple[float, float]:
    return pt[0] / px_per_pdf_point, pt[1] / px_per_pdf_point
