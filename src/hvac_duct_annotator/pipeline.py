from __future__ import annotations

from pathlib import Path
import math
from collections import Counter
import cv2

from hvac_duct_annotator.annotate import annotate_pdf, write_csv_report
from hvac_duct_annotator.associate import associate_labels
from hvac_duct_annotator.classify import (
    classify_segments_by_diffuser_adjacency,
    classify_with_symbol_anchors,
    enrich_round_ducts_with_hough,
)
from hvac_duct_annotator.detect import filter_duct_candidates
from hvac_duct_annotator.ingest import extract_diffuser_tags, extract_native_text
from hvac_duct_annotator.models import AnnotationReport
from hvac_duct_annotator.ocr import assign_dimensions_from_labels, detect_dimension_labels, run_ocr_near_segments
from hvac_duct_annotator.measure import measure_segments
from hvac_duct_annotator.propagate import propagate_dimensions
from hvac_duct_annotator.render import load_pdf_page
from hvac_duct_annotator.scale import detect_scale_info
from hvac_duct_annotator.symbols import detect_symbol_anchors_from_text
from hvac_duct_annotator.validate import apply_validation
from hvac_duct_annotator.vector import build_centerline_segments, extract_raw_geometry


def _dining_hough_trunks(image_bgr, px_per_pdf_point, native_text):
    """
    Fallback detector for long horizontal dining trunks that may be missed by
    strict vector pairing.
    """
    from hvac_duct_annotator.models import BBox, DuctKind, DuctSegment, Point

    h, w = image_bgr.shape[:2]
    # Use a fixed right-top drawing ROI (native text boxes are unreliable on rotated plots).
    x0 = int(0.43 * w)
    y0 = int(0.12 * h)
    x1 = int(0.86 * w)
    y1 = int(0.46 * h)

    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=42, minLineLength=120, maxLineGap=10)
    if lines is None:
        return []

    horiz = []
    for ln in lines[:, 0]:
        xa, ya, xb, yb = [int(v) for v in ln]
        dx = abs(xb - xa)
        dy = abs(yb - ya)
        if dx < 120 or dy > 8:
            continue
        # to image coords
        xa += x0
        xb += x0
        ya += y0
        yb += y0
        if xa > xb:
            xa, xb = xb, xa
            ya, yb = yb, ya
        horiz.append((xa, ya, xb, yb))

    if not horiz:
        return []

    # Pair nearby parallel lines into trunks.
    trunks = []
    used = set()
    sid = 1
    for i, a in enumerate(horiz):
        if i in used:
            continue
        ax1, ay1, ax2, ay2 = a
        best = None
        best_gap = 1e9
        for j, b in enumerate(horiz):
            if i == j or j in used:
                continue
            bx1, by1, bx2, by2 = b
            overlap = min(ax2, bx2) - max(ax1, bx1)
            if overlap < 120:
                continue
            gap = abs(((ay1 + ay2) / 2) - ((by1 + by2) / 2))
            if 4 <= gap <= 42 and gap < best_gap:
                best = j
                best_gap = gap
        if best is None:
            continue
        bx1, by1, bx2, by2 = horiz[best]
        used.add(i)
        used.add(best)
        cx1 = (ax1 + bx1) / 2
        cy1 = (ay1 + by1) / 2
        cx2 = (ax2 + bx2) / 2
        cy2 = (ay2 + by2) / 2
        z = px_per_pdf_point
        seg = DuctSegment(
            id=f"HX-{sid:03d}",
            kind=DuctKind.RECT,
            bbox=BBox(
                x0=min(ax1, ax2, bx1, bx2) / z,
                y0=min(ay1, ay2, by1, by2) / z,
                x1=max(ax1, ax2, bx1, bx2) / z,
                y1=max(ay1, ay2, by1, by2) / z,
            ),
            centerline=[Point(x=cx1 / z, y=cy1 / z), Point(x=cx2 / z, y=cy2 / z)],
            pixel_width=float(best_gap),
            pixel_length=math.hypot(cx2 - cx1, cy2 - cy1),
            confidence=0.42,
            geom_score=0.42,
        )
        trunks.append(seg)
        sid += 1
    return trunks


def _segment_angle(seg):
    if len(seg.centerline) < 2:
        return 0.0
    a, b = seg.centerline[0], seg.centerline[-1]
    return math.atan2(b.y - a.y, b.x - a.x)


def _segment_endpoints(seg):
    if len(seg.centerline) < 2:
        c = seg.centerline[0]
        return (c.x, c.y), (c.x, c.y)
    a, b = seg.centerline[0], seg.centerline[-1]
    return (a.x, a.y), (b.x, b.y)


def _endpoint_min_dist(a, b):
    a1, a2 = _segment_endpoints(a)
    b1, b2 = _segment_endpoints(b)
    d = [
        math.hypot(a1[0] - b1[0], a1[1] - b1[1]),
        math.hypot(a1[0] - b2[0], a1[1] - b2[1]),
        math.hypot(a2[0] - b1[0], a2[1] - b1[1]),
        math.hypot(a2[0] - b2[0], a2[1] - b2[1]),
    ]
    return min(d)


def _merge_collinear_segments(segments):
    """
    Connect fragmented centerlines into longer trunk runs before visual gating.
    """
    candidates = [s for s in segments if s.kind.value == "rect" and len(s.centerline) >= 2]
    if not candidates:
        return segments

    n = len(candidates)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ai = candidates[i]
        ang_i = _segment_angle(ai)
        a1, a2 = _segment_endpoints(ai)
        ah = abs(a2[0] - a1[0]) >= abs(a2[1] - a1[1])
        for j in range(i + 1, n):
            bj = candidates[j]
            ang_j = _segment_angle(bj)
            b1, b2 = _segment_endpoints(bj)
            bh = abs(b2[0] - b1[0]) >= abs(b2[1] - b1[1])
            if ah != bh:
                continue
            d_ang = abs((ang_i - ang_j + math.pi) % math.pi)
            d_ang = min(d_ang, math.pi - d_ang)
            if d_ang > math.radians(5):
                continue
            if _endpoint_min_dist(ai, bj) > 12:
                continue
            # Guard against accidental bridging across separate trunks.
            amx, amy = (ai.bbox.x0 + ai.bbox.x1) / 2, (ai.bbox.y0 + ai.bbox.y1) / 2
            bmx, bmy = (bj.bbox.x0 + bj.bbox.x1) / 2, (bj.bbox.y0 + bj.bbox.y1) / 2
            if ah:
                if abs(amy - bmy) > 10:
                    continue
            else:
                if abs(amx - bmx) > 10:
                    continue
            union(i, j)

    clusters = {}
    for idx in range(n):
        r = find(idx)
        clusters.setdefault(r, []).append(candidates[idx])

    merged = []
    used_ids = set()
    for group in clusters.values():
        if len(group) == 1:
            merged.append(group[0])
            used_ids.add(group[0].id)
            continue
        pts = []
        for s in group:
            p1, p2 = _segment_endpoints(s)
            pts.extend([p1, p2])
            used_ids.add(s.id)
        # conservative merge axis: preserve horizontal/vertical trunk shape
        a0 = _segment_angle(group[0])
        horizontal = abs(math.cos(a0)) >= abs(math.sin(a0))
        if horizontal:
            ys = [(p[1]) for p in pts]
            y = sum(ys) / len(ys)
            left = min(pts, key=lambda p: p[0])
            right = max(pts, key=lambda p: p[0])
            best = ((left[0], y), (right[0], y), abs(right[0] - left[0]))
        else:
            xs = [(p[0]) for p in pts]
            x = sum(xs) / len(xs)
            top = min(pts, key=lambda p: p[1])
            bottom = max(pts, key=lambda p: p[1])
            best = ((x, top[1]), (x, bottom[1]), abs(bottom[1] - top[1]))
        x0 = min(s.bbox.x0 for s in group)
        y0 = min(s.bbox.y0 for s in group)
        x1 = max(s.bbox.x1 for s in group)
        y1 = max(s.bbox.y1 for s in group)
        base = max(group, key=lambda s: s.confidence).model_copy(deep=True)
        from hvac_duct_annotator.models import BBox, Point

        base.bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
        base.centerline = [Point(x=best[0][0], y=best[0][1]), Point(x=best[1][0], y=best[1][1])]
        base.pixel_length = best[2]
        base.pixel_width = max((s.pixel_width or 0.0) for s in group)
        base.confidence = max(s.confidence for s in group)
        # Majority dimension vote when available.
        dims = [s.dimensions for s in group if s.dimensions is not None]
        if dims:
            key = Counter(
                (
                    d.shape,
                    int(d.width_in or 0),
                    int(d.height_in or 0),
                    int(d.diameter_in or 0),
                )
                for d in dims
            ).most_common(1)[0][0]
            from hvac_duct_annotator.models import DuctDimensions

            if key[0] == "round":
                base.dimensions = DuctDimensions(shape="round", diameter_in=key[3])
            else:
                base.dimensions = DuctDimensions(shape="rect", width_in=key[1], height_in=key[2])
        merged.append(base)

    # Keep untouched non-rect or point-only segments.
    for s in segments:
        if s.id not in used_ids and (s.kind.value != "rect" or len(s.centerline) < 2):
            merged.append(s)
    return merged


def _dining_trunk_candidates(raw_segments, native_text):
    """
    Rescue pass: keep long horizontal right-side trunks even when hatch score is weak.
    """
    zones = _room_zones(native_text)
    # Prefer explicit DINING zone; otherwise fallback to right-side heuristic.
    dining_box = zones.get("dining")
    out = []
    for s in raw_segments:
        if s.kind.value != "rect" or len(s.centerline) < 2:
            continue
        if not _is_long_horizontal(s):
            continue
        if s.pixel_length < 90:
            continue
        if s.pixel_width is not None and not (5 <= s.pixel_width <= 40):
            continue
        mx = (s.bbox.x0 + s.bbox.x1) / 2
        my = (s.bbox.y0 + s.bbox.y1) / 2
        in_dining = False
        if dining_box is not None:
            if (dining_box.x0 - 220) <= mx <= (dining_box.x1 + 260) and (dining_box.y0 - 180) <= my <= (dining_box.y1 + 180):
                in_dining = True
        else:
            # right-side fallback
            in_dining = mx > 850
        if not in_dining:
            continue
        clone = s.model_copy(deep=True)
        clone.confidence = max(clone.confidence, 0.35)
        clone.geom_score = max(clone.geom_score, 0.35)
        out.append(clone)
    return out


def _merge_by_near_duplicate(primary, extra):
    out = list(primary)
    for cand in extra:
        cmx = (cand.bbox.x0 + cand.bbox.x1) / 2
        cmy = (cand.bbox.y0 + cand.bbox.y1) / 2
        dup = False
        for s in out:
            smx = (s.bbox.x0 + s.bbox.x1) / 2
            smy = (s.bbox.y0 + s.bbox.y1) / 2
            if math.hypot(cmx - smx, cmy - smy) < 35:
                dup = True
                break
        if not dup:
            out.append(cand)
    return out


def _exclude_hatched_room_noise(segments, native_text):
    """
    Exclude frequent false positives from dense hatch-filled room blocks
    (e.g. walk-in freezer/cooler), unless confidence is very high.
    """
    noisy_boxes = []
    keywords = ("WALK-IN FREEZER", "WALK-IN COOLER", "FREEZER", "COOLER")
    for txt, box in native_text:
        up = txt.upper()
        if any(k in up for k in keywords):
            noisy_boxes.append(box)
    if not noisy_boxes:
        return segments

    def in_noisy_zone(seg):
        mx = (seg.bbox.x0 + seg.bbox.x1) / 2
        my = (seg.bbox.y0 + seg.bbox.y1) / 2
        for b in noisy_boxes:
            # expand zone around room label to cover the full hatch block nearby
            x0 = b.x0 - 130
            y0 = b.y0 - 130
            x1 = b.x1 + 130
            y1 = b.y1 + 130
            if x0 <= mx <= x1 and y0 <= my <= y1:
                return True
        return False

    out = []
    for s in segments:
        if in_noisy_zone(s) and s.confidence < 0.80:
            continue
        out.append(s)
    return out


def _room_zones(native_text):
    zones = {}
    labels = {
        "DINING": "dining",
        "KITCHEN": "kitchen",
        "SCULLERY": "scullery",
        "WALK-IN FREEZER": "freezer",
        "WALK-IN COOLER": "cooler",
    }
    for txt, box in native_text:
        up = txt.upper()
        for k, name in labels.items():
            if k in up:
                zones[name] = box
    return zones


def _segment_zone(seg, zones, x_split=None):
    """
    Assign a named zone to the segment based on proximity to room labels.

    Expansion is intentionally small (±60 pts) so that zones don't bleed
    across the drawing and misclassify ducts in neighbouring rooms.
    The "scullery" label for example is very close to the duct-rich kitchen/
    service area; a wide expansion would incorrectly capture most segments.
    """
    mx = (seg.bbox.x0 + seg.bbox.x1) / 2
    my = (seg.bbox.y0 + seg.bbox.y1) / 2
    for name, b in zones.items():
        x0 = b.x0 - 60
        y0 = b.y0 - 60
        x1 = b.x1 + 60
        y1 = b.y1 + 60
        if x0 <= mx <= x1 and y0 <= my <= y1:
            return name
    if x_split is not None and mx >= x_split:
        return "dining"
    return "other"


def _is_long_horizontal(seg):
    if len(seg.centerline) < 2:
        return False
    a, b = seg.centerline[0], seg.centerline[-1]
    dx = abs(b.x - a.x)
    dy = abs(b.y - a.y)
    return dx > 85 and dy < max(10, 0.18 * dx)


def _is_long_vertical(seg):
    if len(seg.centerline) < 2:
        return False
    a, b = seg.centerline[0], seg.centerline[-1]
    dx = abs(b.x - a.x)
    dy = abs(b.y - a.y)
    return dy > 80 and dx < max(8, 0.12 * dy)


def _segments_for_visual_output(segments, native_text):
    """
    Keep only stable, human-meaningful duct runs for drawing overlay.

    Layout reference (page.rect = 2592 × 1728 PDF pts after rotation):
      - Drawing area: x ∈ [280, 1750], y ∈ [200, 1580]
      - Left border lines (gridline artifacts): x < 280
      - Notes / title block: y > 1580 or x > 1750
    """
    zones = _room_zones(native_text)
    out = []
    for s in segments:
        if s.kind.value != "rect" or len(s.centerline) < 2:
            continue

        a, b = s.centerline[0], s.centerline[-1]
        dx = abs(b.x - a.x)
        dy = abs(b.y - a.y)
        mx = (s.bbox.x0 + s.bbox.x1) / 2
        my = (s.bbox.y0 + s.bbox.y1) / 2

        # Hard exclusion: left/right border and bottom notes / title-block area.
        if mx < 280 or mx > 1750:
            continue
        if my > 1580:
            continue

        # Reject ONLY border/grid vertical lines (left margin x < 350).
        # Real vertical duct trunks are in x 900–1400 range and must be kept.
        is_vertical = dy > 80 and dx < max(8, 0.12 * dy)
        if is_vertical and mx < 350:
            continue

        zone = _segment_zone(s, zones)
        has_dim = s.dimensions is not None
        pdf_len = s.pixel_length or 0  # pixel_length is stored in PDF pts

        # Noisy rooms: only well-labelled, high-confidence segments pass.
        if zone in {"freezer", "cooler"}:
            if has_dim and s.confidence >= 0.65:
                out.append(s)
            continue

        # Dining zone: relaxed — important horizontal trunk rescue.
        if zone == "dining":
            if has_dim and s.confidence >= 0.40:
                out.append(s)
                continue
            if _is_long_horizontal(s) and s.confidence >= 0.30 and pdf_len >= 40:
                out.append(s)
            continue

        # Right-side rescue: canonical my > 1100 puts the segment in the
        # dining / corridor region (far-right in the rendered landscape view).
        # Accept long horizontal runs even without a dimension label; they
        # are almost certainly supply trunks leading to the dining area.
        if my > 1100 and _is_long_horizontal(s) and s.confidence >= 0.30 and pdf_len >= 50:
            out.append(s)
            continue

        # General drawing area (kitchen, scullery, other):
        if has_dim:
            # Labelled segment: keep if the dimension is consistent with the
            # measured pixel width (high consistency_score = reliable match),
            # OR if overall confidence is sufficiently high.
            keep = (
                (s.consistency_score >= 0.50 and pdf_len >= 10)
                or (s.confidence >= 0.50 and pdf_len >= 10)
            )
            if keep:
                out.append(s)
        else:
            # Unlabelled segment: only show long, high-confidence trunks.
            if s.confidence >= 0.55 and pdf_len >= 80:
                out.append(s)
    return out


def run_pipeline(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int = 300,
    scale_mode: str = "auto",
    classify: bool = True,
    report_csv: Path | None = None,
) -> AnnotationReport:
    rendered = load_pdf_page(input_pdf=input_pdf, dpi=dpi)
    native_text = extract_native_text(rendered.page)
    diffuser_tags = extract_diffuser_tags(native_text)
    symbol_anchors = detect_symbol_anchors_from_text(native_text)

    lines, circles = extract_raw_geometry(
        page=rendered.page,
        px_per_pdf_point=rendered.px_per_pdf_point,
        drawing_bbox_px=rendered.drawing_bbox_px,
    )
    z = rendered.px_per_pdf_point
    raw_segments, graph = build_centerline_segments(lines=lines, circles=circles)
    segments = filter_duct_candidates(rendered.image_bgr, raw_segments, z)
    segments = _merge_by_near_duplicate(segments, _dining_trunk_candidates(raw_segments, native_text))
    segments = _merge_by_near_duplicate(segments, _dining_hough_trunks(rendered.image_bgr, z, native_text))
    segments = _exclude_hatched_room_noise(segments, native_text)
    segments = _merge_collinear_segments(segments)

    enrich_round_ducts_with_hough(rendered.image_bgr, segments, z)

    tokens = run_ocr_near_segments(rendered.image_bgr, segments, z)
    labels = detect_dimension_labels(rendered.image_bgr, tokens)
    assign_dimensions_from_labels(segments, labels, z)
    associate_labels(segments, labels, z, image_bgr=rendered.image_bgr)
    propagate_dimensions(segments, graph)

    scale = detect_scale_info(
        rendered.image_bgr,
        dpi=dpi,
        segments=segments,
        px_per_pdf_point=z,
    )
    if scale_mode != "auto":
        scale.source = "manual"
        scale.expression = scale_mode
    measure_segments(segments, scale, z)

    if classify:
        classify_segments_by_diffuser_adjacency(segments, diffuser_tags)
        classify_with_symbol_anchors(segments, symbol_anchors)

    apply_validation(segments, scale.inches_per_pixel)

    visual_segments = _segments_for_visual_output(segments, native_text)
    report = annotate_pdf(input_pdf=input_pdf, output_pdf=output_pdf, segments=visual_segments, scale=scale)
    report.review_queue_count = sum(1 for s in segments if s.review_required)
    if report_csv:
        # CSV remains focused on what is actually annotated in the output PDF.
        write_csv_report(report, report_csv)
    rendered.doc.close()
    return report
