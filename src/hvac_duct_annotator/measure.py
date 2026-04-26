from __future__ import annotations

import math

from hvac_duct_annotator.models import DuctSegment, LengthInfo, ScaleInfo


def _polyline_len(points) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += math.hypot(b.x - a.x, b.y - a.y)
    return total


def measure_segments(
    segments: list[DuctSegment],
    scale: ScaleInfo,
    px_per_pdf_point: float,
) -> None:
    z = px_per_pdf_point
    for seg in segments:
        pdf_len = _polyline_len(seg.centerline)
        if pdf_len <= 0:
            pdf_len = seg.pixel_length
        px_len = pdf_len * z
        real_in = px_len * scale.inches_per_pixel
        feet = int(real_in // 12)
        inches = int(round(real_in - feet * 12))
        if inches == 12:
            feet += 1
            inches = 0
        seg.length = LengthInfo(feet=feet, inches=inches, total_inches=real_in)
