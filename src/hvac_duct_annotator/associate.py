from __future__ import annotations

import math

from hvac_duct_annotator.models import DimensionLabel, DuctSegment
from hvac_duct_annotator.ocr import parse_dimension


def _segment_orientation(seg: DuctSegment) -> float:
    if len(seg.centerline) < 2:
        return 0.0
    p1 = seg.centerline[0]
    p2 = seg.centerline[-1]
    return math.atan2(p2.y - p1.y, p2.x - p1.x)


def _label_orientation(label: DimensionLabel) -> float:
    dx = label.bbox.x1 - label.bbox.x0
    dy = label.bbox.y1 - label.bbox.y0
    return math.atan2(dy, dx)


def associate_labels(
    segments: list[DuctSegment],
    labels: list[DimensionLabel],
    px_per_pdf_point: float,
    max_dist_px: float = 260.0,
) -> None:
    z = px_per_pdf_point
    for seg in segments:
        sx = ((seg.bbox.x0 + seg.bbox.x1) / 2) * z
        sy = ((seg.bbox.y0 + seg.bbox.y1) / 2) * z
        sang = _segment_orientation(seg)
        best_score = -1e18
        best_label = None
        for lb in labels:
            lx = (lb.bbox.x0 + lb.bbox.x1) / 2
            ly = (lb.bbox.y0 + lb.bbox.y1) / 2
            dist = math.hypot(lx - sx, ly - sy)
            if dist > max_dist_px:
                continue
            lang = _label_orientation(lb)
            d_ang = abs((sang - lang + math.pi) % math.pi)
            d_ang = min(d_ang, math.pi - d_ang)
            orientation_bonus = 1.0 - (d_ang / (math.pi / 2))
            score = (max_dist_px - dist) + 50.0 * orientation_bonus + 80.0 * lb.confidence
            if score > best_score:
                best_score = score
                best_label = lb
        if best_label is None:
            continue
        dims = parse_dimension(best_label.normalized)
        if dims is None:
            continue
        seg.label = best_label
        seg.dimensions = dims
        seg.confidence = max(seg.confidence, min(1.0, 0.6 * seg.confidence + 0.4 * best_label.confidence))
