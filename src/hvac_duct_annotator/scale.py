from __future__ import annotations

import re
import statistics

import cv2

from hvac_duct_annotator.models import DuctSegment
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


def _calibrate_from_segments(
    segments: list[DuctSegment],
    px_per_pdf_point: float,
) -> float | None:
    """
    Regress inches_per_pixel from labelled segments.  Returns None when the
    sample set is too homogeneous (e.g. dimension propagation floods all
    segments with a single label), because that biases the regression badly.
    """
    samples: list[tuple[float, float]] = []  # (expected_in, px_width)
    z = px_per_pdf_point
    for seg in segments:
        if seg.dimensions is None or not seg.pixel_width:
            continue
        if seg.dimensions.shape == "round" and seg.dimensions.diameter_in:
            expected_in = float(seg.dimensions.diameter_in)
        elif seg.dimensions.width_in:
            expected_in = float(seg.dimensions.width_in)
        else:
            continue
        px_width = float(seg.pixel_width) * z
        if px_width <= 0:
            continue
        samples.append((expected_in, px_width))
    if len(samples) < 4:
        return None
    # Require at least two distinct physical sizes to trust the calibration.
    # When dimension propagation gives all segments the same size, the
    # regression is unreliable — fall back to the OCR/default scale instead.
    distinct_sizes = {round(e, 1) for e, _ in samples}
    if len(distinct_sizes) < 2:
        return None
    return statistics.median([e / p for e, p in samples])


def detect_scale_info(
    image_bgr,
    dpi: int,
    default_expression: str = '1/4"=1\'-0"',
    segments: list[DuctSegment] | None = None,
    px_per_pdf_point: float = 1.0,
) -> ScaleInfo:
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
    info = ScaleInfo(source=source, expression=expression, inches_per_pixel=inches_per_pixel)
    if segments:
        calibrated = _calibrate_from_segments(segments, px_per_pdf_point)
        if calibrated is not None:
            info.calibrated_inches_per_pixel = calibrated
            # Only override the OCR/default scale when calibration is within 3×;
            # otherwise the calibration is likely corrupted by bad dimension data.
            ratio = calibrated / inches_per_pixel
            if 0.33 <= ratio <= 3.0:
                info.inches_per_pixel = calibrated
                info.source = "calibrated"
    return info
