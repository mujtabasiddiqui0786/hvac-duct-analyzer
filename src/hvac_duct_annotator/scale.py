from __future__ import annotations

import re

import cv2

from hvac_duct_annotator.models import ScaleInfo
from hvac_duct_annotator.ocr import OCRReader

SCALE_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\"?\s*=\s*(\d+)'\s*-\s*(\d+)\"?")


def _parse_scale_text(text: str) -> tuple[float, str] | None:
    m = SCALE_RE.search(text)
    if not m:
        return None
    num = int(m.group(1))
    den = int(m.group(2))
    feet = int(m.group(3))
    inches = int(m.group(4))
    drawn_in = num / den
    real_in = feet * 12 + inches
    if drawn_in <= 0 or real_in <= 0:
        return None
    return real_in / drawn_in, f'{num}/{den}"={feet}\'-{inches}"'


def detect_scale_info(image_bgr, dpi: int, default_expression: str = '1/4"=1\'-0"') -> ScaleInfo:
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = int(0.45 * w), int(0.68 * h), int(0.92 * w), int(0.98 * h)
    roi = image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    reader = OCRReader()
    text = " ".join(t.text for t in reader.read(gray))
    parsed = _parse_scale_text(text)
    if parsed is None:
        # fallback: 1/4" = 1'-0"
        real_per_drawn_in = 48.0
        source = "default"
        expression = default_expression
    else:
        real_per_drawn_in, expression = parsed
        source = "ocr"
    px_per_in = dpi
    inches_per_pixel = real_per_drawn_in / px_per_in
    return ScaleInfo(source=source, expression=expression, inches_per_pixel=inches_per_pixel)
