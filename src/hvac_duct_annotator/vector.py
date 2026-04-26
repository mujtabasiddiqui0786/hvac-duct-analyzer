from __future__ import annotations

import math
from dataclasses import dataclass

import fitz
import networkx as nx
import numpy as np

from hvac_duct_annotator.models import BBox, DuctKind, DuctSegment, Point


@dataclass(slots=True)
class RawLine:
    p1: tuple[float, float]
    p2: tuple[float, float]
    width: float

    @property
    def angle(self) -> float:
        return math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])


def _inside_bbox(pt: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def extract_raw_geometry(
    page: fitz.Page,
    px_per_pdf_point: float,
    drawing_bbox_px: tuple[int, int, int, int],
) -> tuple[list[RawLine], list[tuple[float, float, float]]]:
    lines: list[RawLine] = []
    circles: list[tuple[float, float, float]] = []
    x0, y0, x1, y1 = drawing_bbox_px
    bbox_pdf = (x0 / px_per_pdf_point, y0 / px_per_pdf_point, x1 / px_per_pdf_point, y1 / px_per_pdf_point)
    for item in page.get_drawings():
        stroke = item.get("width", 0.5)
        if stroke is None:
            stroke = 0.5
        for seg in item.get("items", []):
            tag = seg[0]
            if tag == "l":
                p1 = (seg[1].x, seg[1].y)
                p2 = (seg[2].x, seg[2].y)
                if _inside_bbox(p1, bbox_pdf) and _inside_bbox(p2, bbox_pdf):
                    ln = RawLine(p1=p1, p2=p2, width=float(stroke))
                    if ln.length >= 8:
                        lines.append(ln)
            elif tag == "c":
                pts = [seg[1], seg[2], seg[3], seg[4]]
                cx = sum(p.x for p in pts) / 4.0
                cy = sum(p.y for p in pts) / 4.0
                r = max(math.hypot(p.x - cx, p.y - cy) for p in pts)
                if _inside_bbox((cx, cy), bbox_pdf):
                    circles.append((cx, cy, r))
    return lines, circles


def _line_projection_overlap(a: RawLine, b: RawLine) -> float:
    va = np.array([a.p2[0] - a.p1[0], a.p2[1] - a.p1[1]], dtype=float)
    norm = np.linalg.norm(va)
    if norm == 0:
        return 0.0
    va = va / norm
    a_proj = sorted([np.dot(np.array(a.p1), va), np.dot(np.array(a.p2), va)])
    b_proj = sorted([np.dot(np.array(b.p1), va), np.dot(np.array(b.p2), va)])
    lo = max(a_proj[0], b_proj[0])
    hi = min(a_proj[1], b_proj[1])
    ov = max(0.0, hi - lo)
    return ov / max(1e-6, min(a.length, b.length))


def _line_distance(a: RawLine, b: RawLine) -> float:
    ax, ay = a.p1
    bx, by = b.p1
    vx, vy = a.p2[0] - a.p1[0], a.p2[1] - a.p1[1]
    den = math.hypot(vx, vy)
    if den == 0:
        return 99999.0
    return abs((bx - ax) * vy - (by - ay) * vx) / den


def build_centerline_segments(
    lines: list[RawLine],
    circles: list[tuple[float, float, float]],
) -> tuple[list[DuctSegment], nx.Graph]:
    segments: list[DuctSegment] = []
    used: set[int] = set()
    seg_i = 1
    graph = nx.Graph()

    for i, a in enumerate(lines):
        if i in used:
            continue
        best_j = None
        best_score = -1.0
        for j, b in enumerate(lines):
            if i == j or j in used:
                continue
            d_ang = abs((a.angle - b.angle + math.pi) % math.pi)
            d_ang = min(d_ang, math.pi - d_ang)
            if d_ang > math.radians(3):
                continue
            gap = _line_distance(a, b)
            if not (3 <= gap <= 90):
                continue
            ov = _line_projection_overlap(a, b)
            if ov < 0.55:
                continue
            score = ov - 0.002 * gap
            if score > best_score:
                best_j = j
                best_score = score
        if best_j is None:
            continue
        b = lines[best_j]
        used.add(i)
        used.add(best_j)
        p1 = ((a.p1[0] + b.p1[0]) / 2.0, (a.p1[1] + b.p1[1]) / 2.0)
        p2 = ((a.p2[0] + b.p2[0]) / 2.0, (a.p2[1] + b.p2[1]) / 2.0)
        pixel_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        x0 = min(a.p1[0], a.p2[0], b.p1[0], b.p2[0])
        y0 = min(a.p1[1], a.p2[1], b.p1[1], b.p2[1])
        x1 = max(a.p1[0], a.p2[0], b.p1[0], b.p2[0])
        y1 = max(a.p1[1], a.p2[1], b.p1[1], b.p2[1])
        seg = DuctSegment(
            id=f"D-{seg_i:04d}",
            kind=DuctKind.RECT,
            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
            centerline=[Point(x=p1[0], y=p1[1]), Point(x=p2[0], y=p2[1])],
            pixel_width=_line_distance(a, b),
            pixel_length=pixel_len,
        )
        segments.append(seg)
        graph.add_node((round(p1[0], 1), round(p1[1], 1)))
        graph.add_node((round(p2[0], 1), round(p2[1], 1)))
        graph.add_edge(
            (round(p1[0], 1), round(p1[1], 1)),
            (round(p2[0], 1), round(p2[1], 1)),
            segment_id=seg.id,
        )
        seg_i += 1

    for cx, cy, r in circles:
        seg = DuctSegment(
            id=f"D-{seg_i:04d}",
            kind=DuctKind.ROUND,
            bbox=BBox(x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r),
            centerline=[Point(x=cx, y=cy)],
            pixel_width=2 * r,
            pixel_length=2 * math.pi * r,
        )
        segments.append(seg)
        seg_i += 1

    return segments, graph
