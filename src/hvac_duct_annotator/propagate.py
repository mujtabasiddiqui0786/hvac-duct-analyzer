from __future__ import annotations

import math

import networkx as nx

from hvac_duct_annotator.models import DuctDimensions, DuctSegment


def _segment_angle(seg: DuctSegment) -> float:
    if len(seg.centerline) < 2:
        return 0.0
    a, b = seg.centerline[0], seg.centerline[-1]
    return math.atan2(b.y - a.y, b.x - a.x)


def _segment_mid(seg: DuctSegment) -> tuple[float, float]:
    return ((seg.bbox.x0 + seg.bbox.x1) / 2.0, (seg.bbox.y0 + seg.bbox.y1) / 2.0)


def _can_share_dimension(a: DuctSegment, b: DuctSegment) -> bool:
    ma = _segment_mid(a)
    mb = _segment_mid(b)
    dist = math.hypot(mb[0] - ma[0], mb[1] - ma[1])
    da = _segment_angle(a)
    db = _segment_angle(b)
    d_ang = abs((da - db + math.pi) % math.pi)
    d_ang = min(d_ang, math.pi - d_ang)
    return dist < 80 and d_ang < math.radians(12)


def _graph_neighbor_segments(
    graph: nx.Graph,
    segment_id: str,
) -> set[str]:
    out: set[str] = set()
    for n1, n2, data in graph.edges(data=True):
        if data.get("segment_id") != segment_id:
            continue
        for u, v, d2 in graph.edges(n1, data=True):
            sid = d2.get("segment_id")
            if sid and sid != segment_id:
                out.add(sid)
        for u, v, d2 in graph.edges(n2, data=True):
            sid = d2.get("segment_id")
            if sid and sid != segment_id:
                out.add(sid)
    return out


def propagate_dimensions(segments: list[DuctSegment], graph: nx.Graph | None = None) -> None:
    """
    Collinear/local continuity propagation.
    Repeats until no new dimensions are inferred.
    """
    changed = True
    while changed:
        changed = False
        for target in segments:
            if target.dimensions is not None:
                continue
            best: tuple[float, DuctDimensions] | None = None
            tm = _segment_mid(target)
            graph_neighbors = None
            if graph is not None:
                graph_neighbors = _graph_neighbor_segments(graph, target.id)
            for src in segments:
                if src.dimensions is None or src.id == target.id:
                    continue
                if graph_neighbors is not None and src.id not in graph_neighbors:
                    continue
                if not _can_share_dimension(src, target):
                    continue
                sm = _segment_mid(src)
                d = math.hypot(tm[0] - sm[0], tm[1] - sm[1])
                if best is None or d < best[0]:
                    best = (d, src.dimensions)
            if best is None:
                continue
            target.dimensions = best[1]
            target.review_required = True
            target.review_reason = "dimension propagated from nearby collinear segment"
            changed = True
