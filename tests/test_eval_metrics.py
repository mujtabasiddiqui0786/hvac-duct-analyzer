from __future__ import annotations

from hvac_duct_annotator.eval_metrics import evaluate_precision_recall
from hvac_duct_annotator.models import BBox, DuctDimensions, DuctKind, DuctSegment


def test_precision_recall_simple_match() -> None:
    pred = DuctSegment(
        id="D1",
        kind=DuctKind.RECT,
        bbox=BBox(x0=0, y0=0, x1=10, y1=4),
        dimensions=DuctDimensions(shape="rect", width_in=12, height_in=8),
    )
    gt = [
        {
            "kind": "rect",
            "bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 4},
            "dimensions": {"shape": "rect", "width_in": 12, "height_in": 8},
        }
    ]
    r = evaluate_precision_recall([pred], gt, iou_threshold=0.2)
    assert r.tp == 1
    assert r.fp == 0
    assert r.fn == 0
