from __future__ import annotations

import csv
from pathlib import Path

import fitz

from hvac_duct_annotator.models import AnnotationReport, DuctClass, DuctKind, DuctSegment, ScaleInfo


def _label_text(seg: DuctSegment) -> str:
    dim = "?"
    if seg.dimensions is not None:
        if seg.dimensions.shape == "rect":
            dim = f'{int(seg.dimensions.width_in)}"x{int(seg.dimensions.height_in)}"'
        elif seg.dimensions.shape == "round":
            dim = f'{int(seg.dimensions.diameter_in)}"ø'
    ln = "?"
    if seg.length is not None:
        ln = f"{seg.length.feet}'-{seg.length.inches}\""
    return f"{dim} - {ln}"


def _color_for_class(cls: DuctClass) -> tuple[float, float, float]:
    # Match expected sample style: blue duct overlays.
    return (0.10, 0.45, 0.85)


def annotate_pdf(
    input_pdf: Path,
    output_pdf: Path,
    segments: list[DuctSegment],
    scale: ScaleInfo,
) -> AnnotationReport:
    doc = fitz.open(str(input_pdf))
    page = doc[0]
    summary = {"total": len(segments), "rect": 0, "round": 0}
    for seg in segments:
        if seg.kind == DuctKind.RECT:
            summary["rect"] += 1
        else:
            summary["round"] += 1

        color = _color_for_class(seg.classification)
        # Lighter tint for fills (approx 30% opacity equivalent)
        fill_color = tuple(0.6 + 0.4 * c for c in color)

        # Draw duct bounding-box overlay (semi-transparent fill + border).
        bbox = fitz.Rect(seg.bbox.x0, seg.bbox.y0, seg.bbox.x1, seg.bbox.y1)
        w = seg.bbox.x1 - seg.bbox.x0
        h = seg.bbox.y1 - seg.bbox.y0
        # Only fill the bbox when the segment has a real 2-D width
        # (avoid thick fills on near-point/tiny bboxes).
        if w > 4 and h > 4:
            page.draw_rect(bbox, color=color, fill=fill_color, width=1.2, overlay=True)

        # Thick centerline on top for clear visual emphasis.
        if len(seg.centerline) >= 2:
            p1 = fitz.Point(seg.centerline[0].x, seg.centerline[0].y)
            p2 = fitz.Point(seg.centerline[-1].x, seg.centerline[-1].y)
            page.draw_line(p1, p2, color=color, width=3.5, overlay=True)
            mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
        else:
            page.draw_rect(bbox, color=color, width=2.0, overlay=True)
            mx = (seg.bbox.x0 + seg.bbox.x1) / 2
            my = (seg.bbox.y0 + seg.bbox.y1) / 2

        # Dimension / length badge.
        label = _label_text(seg)
        badge = fitz.Rect(mx + 3, my - 11, mx + 130, my + 7)
        page.draw_rect(badge, color=(1, 1, 1), fill=(1, 1, 1), width=0.3, overlay=True)
        page.insert_text(
            (badge.x0 + 2, badge.y1 - 2),
            label,
            fontsize=6.0,
            color=(0.05, 0.05, 0.5),
            overlay=True,
        )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf))
    doc.close()
    return AnnotationReport(
        input_path=str(input_pdf),
        output_path=str(output_pdf),
        scale=scale,
        segments=segments,
        summary=summary,
    )


def write_csv_report(report: AnnotationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "id",
                "kind",
                "classification",
                "dimension",
                "length_ft_in",
                "confidence",
                "geom_score",
                "ocr_score",
                "consistency_score",
                "review_required",
                "review_reason",
            ]
        )
        for seg in report.segments:
            if seg.dimensions is None:
                dim = ""
            elif seg.dimensions.shape == "rect":
                dim = f'{seg.dimensions.width_in}x{seg.dimensions.height_in}'
            else:
                dim = f'{seg.dimensions.diameter_in}ø'
            length = ""
            if seg.length:
                length = f"{seg.length.feet}'-{seg.length.inches}\""
            writer.writerow(
                [
                    seg.id,
                    seg.kind.value,
                    seg.classification.value,
                    dim,
                    length,
                    f"{seg.confidence:.3f}",
                    f"{seg.geom_score:.3f}",
                    f"{seg.ocr_score:.3f}",
                    f"{seg.consistency_score:.3f}",
                    str(seg.review_required),
                    seg.review_reason or "",
                ]
            )
