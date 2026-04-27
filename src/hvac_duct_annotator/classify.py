from __future__ import annotations

import math

import cv2
import numpy as np

from hvac_duct_annotator.models import DuctClass, DuctKind, DuctSegment
from hvac_duct_annotator.symbols import SymbolAnchor


def enrich_round_ducts_with_hough(
    image_bgr: np.ndarray,
    segments: list[DuctSegment],
    px_per_pdf_point: float,
) -> None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=24,
        param1=120,
        param2=22,
        minRadius=6,
        maxRadius=80,
    )
    if circles is None:
        return
    detected = np.round(circles[0, :]).astype(int)
    z = px_per_pdf_point
    for seg in segments:
        if seg.kind != DuctKind.ROUND:
            continue
        cx = ((seg.bbox.x0 + seg.bbox.x1) / 2) * z
        cy = ((seg.bbox.y0 + seg.bbox.y1) / 2) * z
        hit = False
        for x, y, r in detected:
            if math.hypot(cx - x, cy - y) < max(10, r * 0.8):
                seg.pixel_width = max(seg.pixel_width or 0.0, float(2 * r))
                hit = True
                break
        if hit:
            seg.confidence = max(seg.confidence, 0.82)


def classify_segments_by_diffuser_adjacency(segments: list[DuctSegment], diffuser_tags) -> None:
    for seg in segments:
        sx = (seg.bbox.x0 + seg.bbox.x1) / 2
        sy = (seg.bbox.y0 + seg.bbox.y1) / 2
        best = None
        best_d = 1e18
        for dtype, _, box in diffuser_tags:
            dx = (box.x0 + box.x1) / 2
            dy = (box.y0 + box.y1) / 2
            d = (sx - dx) ** 2 + (sy - dy) ** 2
            if d < best_d:
                best_d = d
                best = dtype
        if best is None:
            continue
        if best in {"A", "B"}:
            seg.classification = DuctClass.SUPPLY
        elif best == "C":
            seg.classification = DuctClass.RETURN
        elif best == "F":
            seg.classification = DuctClass.EXHAUST


def classify_with_symbol_anchors(segments: list[DuctSegment], anchors: list[SymbolAnchor]) -> None:
    """
    Upgrade path: use generic symbol anchors (A/B/C/F + equipment) with
    nearest-neighbor routing heuristic.
    """
    typed = [a for a in anchors if a.kind in {"A", "B", "C", "F"}]
    if not typed:
        return
    for seg in segments:
        sx = (seg.bbox.x0 + seg.bbox.x1) / 2
        sy = (seg.bbox.y0 + seg.bbox.y1) / 2
        best = None
        best_d = 1e18
        for anchor in typed:
            ax = (anchor.bbox.x0 + anchor.bbox.x1) / 2
            ay = (anchor.bbox.y0 + anchor.bbox.y1) / 2
            d = (sx - ax) ** 2 + (sy - ay) ** 2
            if d < best_d:
                best_d = d
                best = anchor.kind
        if best in {"A", "B"}:
            seg.classification = DuctClass.SUPPLY
        elif best == "C":
            seg.classification = DuctClass.RETURN
        elif best == "F":
            seg.classification = DuctClass.EXHAUST
