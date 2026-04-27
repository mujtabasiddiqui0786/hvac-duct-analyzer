from __future__ import annotations

from hvac_duct_annotator.models import BBox, DuctDimensions, DuctKind, DuctSegment, Point
from hvac_duct_annotator.propagate import propagate_dimensions


def test_propagate_dimensions_collinear_neighbor() -> None:
    src = DuctSegment(
        id="D-1",
        kind=DuctKind.RECT,
        bbox=BBox(x0=0, y0=0, x1=20, y1=4),
        centerline=[Point(x=0, y=2), Point(x=20, y=2)],
        dimensions=DuctDimensions(shape="rect", width_in=12, height_in=8),
    )
    dst = DuctSegment(
        id="D-2",
        kind=DuctKind.RECT,
        bbox=BBox(x0=24, y0=0, x1=44, y1=4),
        centerline=[Point(x=24, y=2), Point(x=44, y=2)],
    )
    segments = [src, dst]
    propagate_dimensions(segments)
    assert dst.dimensions is not None
    assert dst.review_required is True
