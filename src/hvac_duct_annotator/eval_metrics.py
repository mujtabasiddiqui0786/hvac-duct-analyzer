from __future__ import annotations

from dataclasses import dataclass

from hvac_duct_annotator.models import DuctSegment


@dataclass(slots=True)
class EvalResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _bbox_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


def _dim_match(pred: DuctSegment, gt: dict) -> bool:
    pd = pred.dimensions
    gd = gt.get("dimensions")
    if pd is None or gd is None:
        return False
    if pd.shape != gd.get("shape"):
        return False
    if pd.shape == "round":
        return int(pd.diameter_in or 0) == int(gd.get("diameter_in", -1))
    return int(pd.width_in or 0) == int(gd.get("width_in", -1)) and int(pd.height_in or 0) == int(gd.get("height_in", -1))


def evaluate_precision_recall(
    predicted: list[DuctSegment],
    ground_truth_segments: list[dict],
    iou_threshold: float = 0.25,
) -> EvalResult:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred in predicted:
        best_idx = -1
        best_iou = 0.0
        for i, gt in enumerate(ground_truth_segments):
            if i in matched_gt:
                continue
            if pred.kind.value != gt.get("kind"):
                continue
            iou = _bbox_iou(pred.bbox.model_dump(), gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_threshold and _dim_match(pred, ground_truth_segments[best_idx]):
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(ground_truth_segments) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return EvalResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
