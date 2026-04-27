from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from hvac_duct_annotator.models import BBox, DimensionLabel, DuctDimensions, DuctKind, DuctSegment

RECT_RE = re.compile(r"^\s*(\d{1,2})\s*[\"”]?\s*[xX×]\s*(\d{1,2})\s*[\"”]?\s*$")
ROUND_RE = re.compile(r"^\s*(\d{1,2})\s*[\"”]?\s*[øØ⌀⏀]\s*$|^\s*(\d{1,2})\s*\"?\s*DIA\.?\s*$", re.IGNORECASE)
ALLOWED_SIZE = set(range(4, 61))
CONFUSIONS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "Z": "2", "B": "8"})
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRToken:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


class OCRReader:
    def __init__(self) -> None:
        self._reader = None
        self._fallback = None
        self._disabled = False

    def _ensure_reader(self) -> None:
        if self._disabled:
            return
        if self._reader is None:
            try:
                import easyocr

                self._reader = easyocr.Reader(["en"], gpu=False)
            except ModuleNotFoundError as exc:
                # Some Python builds miss stdlib extension modules like _lzma.
                # Try pytesseract fallback before disabling OCR.
                if "_lzma" in str(exc):
                    try:
                        import pytesseract  # type: ignore

                        self._fallback = pytesseract
                        logger.warning("EasyOCR unavailable (%s), using pytesseract fallback.", exc)
                        return
                    except Exception:
                        pass
                self._disabled = True
                logger.warning("OCR disabled due to missing module dependency: %s", exc)
            except Exception as exc:
                self._disabled = True
                logger.warning("OCR disabled due to initialization error: %s", exc)

    def read(self, image: np.ndarray) -> list[OCRToken]:
        self._ensure_reader()
        if self._reader is None and self._fallback is None:
            return []
        if self._reader is None and self._fallback is not None:
            text = self._fallback.image_to_data(image, output_type=self._fallback.Output.DICT)
            out: list[OCRToken] = []
            n = len(text["text"])
            for i in range(n):
                token = str(text["text"][i]).strip()
                if not token:
                    continue
                conf = float(text["conf"][i]) / 100.0 if str(text["conf"][i]).strip() not in {"-1", ""} else 0.0
                x, y, w, h = text["left"][i], text["top"][i], text["width"][i], text["height"][i]
                out.append(OCRToken(text=token, confidence=max(0.0, min(1.0, conf)), bbox=(x, y, x + w, y + h)))
            return out
        out: list[OCRToken] = []
        for item in self._reader.readtext(image):
            poly, txt, conf = item
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            out.append(OCRToken(text=str(txt), confidence=float(conf), bbox=(min(xs), min(ys), max(xs), max(ys))))
        return out


def normalize_dimension_text(raw: str) -> str:
    txt = raw.strip().translate(CONFUSIONS)
    txt = txt.replace(" ", "")
    txt = txt.replace("X", "x")
    for ch in ('"', "\u201c", "\u201d", "'", "×"):
        txt = txt.replace(ch, "")
    return txt


def parse_dimension(text: str) -> DuctDimensions | None:
    m = RECT_RE.match(text)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
        if w in ALLOWED_SIZE and h in ALLOWED_SIZE:
            return DuctDimensions(shape="rect", width_in=w, height_in=h)
    m = ROUND_RE.match(text)
    if m:
        d = int(next(g for g in m.groups() if g))
        if d in ALLOWED_SIZE:
            return DuctDimensions(shape="round", diameter_in=d)
    return None


def detect_dimension_labels(image_bgr: np.ndarray, tokens: Iterable[OCRToken]) -> list[DimensionLabel]:
    labels: list[DimensionLabel] = []
    for tok in tokens:
        normalized = normalize_dimension_text(tok.text)
        if parse_dimension(normalized) is None:
            continue
        labels.append(
            DimensionLabel(
                text=tok.text,
                normalized=normalized,
                confidence=tok.confidence,
                bbox=BBox(x0=tok.bbox[0], y0=tok.bbox[1], x1=tok.bbox[2], y1=tok.bbox[3]),
            )
        )
    return labels


def run_ocr_near_segments(
    image_bgr: np.ndarray,
    segments: list[DuctSegment],
    px_per_pdf_point: float,
    tile_size: int = 320,
    tile_overlap: int = 48,
    ocr_scale: float = 2.0,
) -> list[OCRToken]:
    """
    Scan the drawing region in tiles instead of once per segment.
    Each tile is upscaled by *ocr_scale* before OCR so that small duct-dimension
    labels (≈10-14 px tall at 120 DPI) fill enough pixels for EasyOCR to read.

    Speed vs accuracy trade-offs:
      tile_size=320, ocr_scale=2.0  →  effective 640-px tiles  ≈ 20-30 tiles total
      vs. the original per-segment approach (≈1372 OCR crops).
    """
    if not segments:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    reader = OCRReader()
    h, w = gray.shape[:2]
    z = px_per_pdf_point

    # Union bbox of all segments with margin.
    margin = int(tile_size * 0.4)
    all_x0 = max(0, int(min(s.bbox.x0 for s in segments) * z) - margin)
    all_y0 = max(0, int(min(s.bbox.y0 for s in segments) * z) - margin)
    all_x1 = min(w, int(max(s.bbox.x1 for s in segments) * z) + margin)
    all_y1 = min(h, int(max(s.bbox.y1 for s in segments) * z) + margin)

    step = tile_size - tile_overlap
    scale = float(ocr_scale)
    raw_tokens: list[OCRToken] = []

    ty = all_y0
    while ty < all_y1:
        tx = all_x0
        while tx < all_x1:
            tx1 = min(tx + tile_size, all_x1)
            ty1 = min(ty + tile_size, all_y1)
            roi = gray[ty:ty1, tx:tx1]
            if roi.size == 0:
                tx += step
                continue
            # Upscale for better OCR accuracy on small labels.
            roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for tok in reader.read(roi_up):
                # Map upscaled coordinates back to original pixel space.
                bx0 = tok.bbox[0] / scale + tx
                by0 = tok.bbox[1] / scale + ty
                bx1 = tok.bbox[2] / scale + tx
                by1 = tok.bbox[3] / scale + ty
                raw_tokens.append(
                    OCRToken(text=tok.text, confidence=tok.confidence, bbox=(bx0, by0, bx1, by1))
                )
            tx += step
        ty += step

    # Deduplicate: same text at nearly the same position (within 10 px).
    seen: list[OCRToken] = []
    for tok in raw_tokens:
        cx = (tok.bbox[0] + tok.bbox[2]) / 2
        cy = (tok.bbox[1] + tok.bbox[3]) / 2
        dup = False
        for prev in seen:
            if prev.text != tok.text:
                continue
            pcx = (prev.bbox[0] + prev.bbox[2]) / 2
            pcy = (prev.bbox[1] + prev.bbox[3]) / 2
            if abs(cx - pcx) < 10 and abs(cy - pcy) < 10:
                dup = True
                break
        if not dup:
            seen.append(tok)

    return seen


def assign_dimensions_from_labels(
    segments: list[DuctSegment],
    labels: list[DimensionLabel],
    px_per_pdf_point: float,
    max_dist_px: float = 420.0,
) -> None:
    z = px_per_pdf_point
    max_d2 = max_dist_px * max_dist_px
    for seg in segments:
        best = None
        best_d = 1e18
        cx = ((seg.bbox.x0 + seg.bbox.x1) / 2) * z
        cy = ((seg.bbox.y0 + seg.bbox.y1) / 2) * z
        for lb in labels:
            lx = (lb.bbox.x0 + lb.bbox.x1) / 2
            ly = (lb.bbox.y0 + lb.bbox.y1) / 2
            d = (lx - cx) ** 2 + (ly - cy) ** 2
            if d < best_d:
                best_d = d
                best = lb
        if best is None or best_d > max_d2:
            continue
        dims = parse_dimension(best.normalized)
        if dims is None:
            continue
        if seg.kind == DuctKind.ROUND and dims.shape != "round":
            continue
        seg.label = best
        seg.dimensions = dims
        seg.confidence = max(seg.confidence, min(1.0, best.confidence))
