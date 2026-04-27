from __future__ import annotations

import math

import cv2
import numpy as np

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


def _line_crossings_penalty(
    sx: float,
    sy: float,
    lx: float,
    ly: float,
    segments: list[DuctSegment],
    current_id: str,
    px_per_pdf_point: float,
) -> float:
    # Penalize label-to-segment lines that cut through many other segment boxes.
    z = px_per_pdf_point
    penalty = 0.0
    for other in segments:
        if other.id == current_id:
            continue
        ox0 = other.bbox.x0 * z
        oy0 = other.bbox.y0 * z
        ox1 = other.bbox.x1 * z
        oy1 = other.bbox.y1 * z
        # Midpoint test as cheap crossing proxy.
        mx = (sx + lx) * 0.5
        my = (sy + ly) * 0.5
        if ox0 <= mx <= ox1 and oy0 <= my <= oy1:
            penalty += 30.0
    return penalty


def _leader_line_bonus(edge_image: np.ndarray | None, sx: float, sy: float, lx: float, ly: float) -> float:
    if edge_image is None:
        return 0.0
    h, w = edge_image.shape[:2]
    x0, y0 = int(max(0, min(w - 1, sx))), int(max(0, min(h - 1, sy)))
    x1, y1 = int(max(0, min(w - 1, lx))), int(max(0, min(h - 1, ly)))
    # Raster sample line and reward if pixels exist (leader-like connection).
    canvas = np.zeros_like(edge_image)
    cv2.line(canvas, (x0, y0), (x1, y1), 255, 1)
    overlap = cv2.bitwise_and(canvas, edge_image)
    hits = int(np.count_nonzero(overlap))
    return min(25.0, hits * 0.4)


def _hough_line_candidates(edge_image: np.ndarray | None) -> list[tuple[int, int, int, int]]:
    if edge_image is None:
        return []
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=14,
        maxLineGap=5,
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in ln[0]) for ln in lines]


def _point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2)
    t = c1 / c2
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


def _leader_trace_bonus(
    hough_lines: list[tuple[int, int, int, int]],
    sx: float,
    sy: float,
    lx: float,
    ly: float,
) -> float:
    # True geometric test: reward when an actual detected line segment likely bridges
    # label vicinity and segment vicinity with matching direction.
    if not hough_lines:
        return 0.0
    tx = sx - lx
    ty = sy - ly
    tlen = math.hypot(tx, ty)
    if tlen < 1e-6:
        return 0.0
    tx /= tlen
    ty /= tlen
    best = 0.0
    for x1, y1, x2, y2 in hough_lines:
        lvx = x2 - x1
        lvy = y2 - y1
        llen = math.hypot(lvx, lvy)
        if llen < 1e-6:
            continue
        lvx /= llen
        lvy /= llen
        align = abs(lvx * tx + lvy * ty)
        if align < 0.92:
            continue
        d_label = _point_to_segment_distance(lx, ly, x1, y1, x2, y2)
        d_seg = _point_to_segment_distance(sx, sy, x1, y1, x2, y2)
        # Segment should be close to both endpoints of the connection path.
        if d_label > 18 or d_seg > 22:
            continue
        score = 20.0 * align + max(0.0, 10.0 - 0.2 * (d_label + d_seg))
        if score > best:
            best = score
    return min(35.0, best)


def associate_labels(
    segments: list[DuctSegment],
    labels: list[DimensionLabel],
    px_per_pdf_point: float,
    image_bgr: np.ndarray | None = None,
    max_dist_px: float = 260.0,
) -> None:
    z = px_per_pdf_point
    edge_image = None
    hough_lines: list[tuple[int, int, int, int]] = []
    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.Canny(gray, 70, 170)
        hough_lines = _hough_line_candidates(edge_image)
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
            crossing_penalty = _line_crossings_penalty(sx, sy, lx, ly, segments, seg.id, z)
            leader_bonus = _leader_line_bonus(edge_image, sx, sy, lx, ly)
            trace_bonus = _leader_trace_bonus(hough_lines, sx, sy, lx, ly)
            score = (
                (max_dist_px - dist)
                + 50.0 * orientation_bonus
                + 80.0 * lb.confidence
                + leader_bonus
                + trace_bonus
                - crossing_penalty
            )
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
