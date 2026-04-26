from __future__ import annotations

from hvac_duct_annotator.models import DuctSegment


def _expected_width_from_dims(seg: DuctSegment) -> float | None:
    if seg.dimensions is None:
        return None
    if seg.dimensions.shape == "round":
        return float(seg.dimensions.diameter_in or 0.0)
    return float(seg.dimensions.width_in or 0.0)


def _consistency_score(seg: DuctSegment, inches_per_pixel: float) -> float:
    expected_in = _expected_width_from_dims(seg)
    if expected_in is None or expected_in <= 0 or not seg.pixel_width:
        return 0.0
    measured_in = float(seg.pixel_width) * inches_per_pixel
    ratio_err = abs(measured_in - expected_in) / max(expected_in, 1e-6)
    return max(0.0, 1.0 - min(1.0, ratio_err))


def apply_validation(segments: list[DuctSegment], inches_per_pixel: float, threshold: float = 0.75) -> None:
    for seg in segments:
        seg.consistency_score = _consistency_score(seg, inches_per_pixel)
        # geom_score can be set earlier by detect.py; fallback to confidence.
        seg.geom_score = max(seg.geom_score, min(1.0, seg.confidence))
        seg.ocr_score = seg.label.confidence if seg.label is not None else 0.0
        seg.confidence = (
            0.45 * seg.geom_score
            + 0.35 * seg.ocr_score
            + 0.20 * seg.consistency_score
        )
        if seg.confidence < threshold:
            seg.review_required = True
            if seg.review_reason is None:
                seg.review_reason = "low confidence composite score"
