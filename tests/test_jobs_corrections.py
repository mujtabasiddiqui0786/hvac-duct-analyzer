from __future__ import annotations

import json
from pathlib import Path

from hvac_duct_annotator.web.jobs import JobStore


def test_apply_corrections_updates_report_and_csv(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "runs")
    fake_pdf = tmp_path / "input.pdf"
    fake_pdf.write_text("x", encoding="utf-8")
    state = store.create(fake_pdf)
    payload = {
        "segments": [
            {
                "id": "D-1",
                "kind": "rect",
                "classification": "unclassified",
                "dimensions": {"shape": "rect", "width_in": 12, "height_in": 8},
                "length": {"feet": 2, "inches": 3},
                "confidence": 0.2,
                "geom_score": 0.2,
                "ocr_score": 0.0,
                "consistency_score": 0.0,
                "review_required": True,
                "review_reason": "low confidence",
            }
        ],
        "review_queue_count": 1,
    }
    state.report_json.write_text(json.dumps(payload), encoding="utf-8")
    result = store.apply_corrections(
        state.id,
        [{"id": "D-1", "classification": "supply", "review_required": False, "review_reason": "approved"}],
    )
    updated = json.loads(state.report_json.read_text(encoding="utf-8"))
    assert result["applied"] == 1
    assert updated["segments"][0]["classification"] == "supply"
    assert updated["segments"][0]["review_required"] is False
    assert state.report_csv.exists()
