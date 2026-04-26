from __future__ import annotations

from hvac_duct_annotator.vector import RawLine, _line_projection_overlap


def test_overlap_computation() -> None:
    a = RawLine((0, 0), (100, 0), 1.0)
    b = RawLine((10, 2), (80, 2), 1.0)
    ov = _line_projection_overlap(a, b)
    assert ov > 0.6
