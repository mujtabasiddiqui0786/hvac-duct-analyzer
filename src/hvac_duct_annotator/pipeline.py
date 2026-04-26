from __future__ import annotations

from pathlib import Path

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
    segments, graph = build_centerline_segments(lines=lines, circles=circles)
    segments = filter_duct_candidates(rendered.image_bgr, segments, z)

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

    report = annotate_pdf(input_pdf=input_pdf, output_pdf=output_pdf, segments=segments, scale=scale)
    report.review_queue_count = sum(1 for s in segments if s.review_required)
    if report_csv:
        write_csv_report(report, report_csv)
    rendered.doc.close()
    return report
