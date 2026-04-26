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
) -> list[OCRToken]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    reader = OCRReader()
    tokens: list[OCRToken] = []
    h, w = gray.shape[:2]
    z = px_per_pdf_point
    for seg in segments:
        cx = int(((seg.bbox.x0 + seg.bbox.x1) / 2) * z)
        cy = int(((seg.bbox.y0 + seg.bbox.y1) / 2) * z)
        rad = max(
            45,
            int(max((seg.bbox.x1 - seg.bbox.x0) * z, (seg.bbox.y1 - seg.bbox.y0) * z) * 0.8),
        )
        x0 = max(0, cx - rad)
        y0 = max(0, cy - rad)
        x1 = min(w, cx + rad)
        y1 = min(h, cy + rad)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        for tok in reader.read(roi):
            tokens.append(
                OCRToken(
                    text=tok.text,
                    confidence=tok.confidence,
                    bbox=(tok.bbox[0] + x0, tok.bbox[1] + y0, tok.bbox[2] + x0, tok.bbox[3] + y0),
                )
            )
    return tokens


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
