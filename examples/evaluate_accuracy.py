from __future__ import annotations

import json
from pathlib import Path

from hvac_duct_annotator.eval_metrics import evaluate_precision_recall
from hvac_duct_annotator.models import DuctSegment


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    report_path = root / "output" / "report_testset2.json"
    gt_path = root / "tests" / "fixtures" / "ground_truth.json"
    if not report_path.exists():
        print("Missing report:", report_path)
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    min_segments = int(gt["expected"]["min_detected_segments"])
    detected = len(report.get("segments", []))
    review_queue = int(report.get("review_queue_count", 0))
    pass_detect = detected >= min_segments
    print(f"Detected segments: {detected} (target >= {min_segments})")
    print(f"Review queue: {review_queue}")
    print(f"Detection gate: {'PASS' if pass_detect else 'FAIL'}")
    gt_segments = gt.get("segments", [])
    if gt_segments:
        predicted = [DuctSegment.model_validate(s) for s in report.get("segments", [])]
        eval_result = evaluate_precision_recall(predicted, gt_segments)
        print(
            "Precision={:.3f} Recall={:.3f} F1={:.3f} (TP={}, FP={}, FN={})".format(
                eval_result.precision,
                eval_result.recall,
                eval_result.f1,
                eval_result.tp,
                eval_result.fp,
                eval_result.fn,
            )
        )
    else:
        print("No fully-labeled ground truth segments present yet (ground_truth.json -> segments).")


if __name__ == "__main__":
    main()
