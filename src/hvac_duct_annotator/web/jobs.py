from __future__ import annotations

import json
import uuid
import csv
from dataclasses import dataclass
from pathlib import Path

from hvac_duct_annotator.pipeline import run_pipeline


@dataclass
class JobState:
    id: str
    status: str
    input_pdf: Path
    output_pdf: Path
    report_json: Path
    report_csv: Path
    review_csv: Path
    error: str | None = None


class JobStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}

    def create(self, input_pdf: Path) -> JobState:
        job_id = str(uuid.uuid4())
        out_dir = self.base_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        state = JobState(
            id=job_id,
            status="queued",
            input_pdf=input_pdf,
            output_pdf=out_dir / "annotated.pdf",
            report_json=out_dir / "report.json",
            report_csv=out_dir / "report.csv",
            review_csv=out_dir / "review_queue.csv",
        )
        self._jobs[job_id] = state
        return state

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def run(self, job_id: str) -> None:
        state = self._jobs[job_id]
        state.status = "running"
        try:
            report = run_pipeline(
                input_pdf=state.input_pdf,
                output_pdf=state.output_pdf,
                report_csv=state.report_csv,
            )
            state.report_json.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
            with state.review_csv.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["id", "confidence", "reason"])
                for seg in report.segments:
                    if seg.review_required:
                        writer.writerow([seg.id, f"{seg.confidence:.3f}", seg.review_reason or ""])
            state.status = "done"
        except Exception as exc:  # pragma: no cover
            state.status = "failed"
            state.error = str(exc)

    def apply_corrections(self, job_id: str, corrections: list[dict]) -> dict:
        state = self._jobs[job_id]
        payload = json.loads(state.report_json.read_text(encoding="utf-8"))
        by_id = {seg["id"]: seg for seg in payload.get("segments", [])}
        applied = 0
        for corr in corrections:
            seg_id = corr.get("id")
            if not seg_id or seg_id not in by_id:
                continue
            seg = by_id[seg_id]
            if "classification" in corr:
                seg["classification"] = corr["classification"]
            if "dimensions" in corr and isinstance(corr["dimensions"], dict):
                seg["dimensions"] = corr["dimensions"]
            if "review_required" in corr:
                seg["review_required"] = bool(corr["review_required"])
            if "review_reason" in corr:
                seg["review_reason"] = corr["review_reason"]
            applied += 1
        payload["review_queue_count"] = sum(1 for s in payload.get("segments", []) if s.get("review_required"))
        state.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with state.report_csv.open("w", newline="", encoding="utf-8") as fp:
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
            for seg in payload.get("segments", []):
                dim = ""
                d = seg.get("dimensions")
                if isinstance(d, dict):
                    if d.get("shape") == "round":
                        dim = f"{d.get('diameter_in','')}ø"
                    elif d.get("shape") == "rect":
                        dim = f"{d.get('width_in','')}x{d.get('height_in','')}"
                ln = seg.get("length") or {}
                length = ""
                if isinstance(ln, dict) and "feet" in ln and "inches" in ln:
                    length = f"{ln['feet']}'-{ln['inches']}\""
                writer.writerow(
                    [
                        seg.get("id", ""),
                        seg.get("kind", ""),
                        seg.get("classification", ""),
                        dim,
                        length,
                        f"{float(seg.get('confidence', 0.0)):.3f}",
                        f"{float(seg.get('geom_score', 0.0)):.3f}",
                        f"{float(seg.get('ocr_score', 0.0)):.3f}",
                        f"{float(seg.get('consistency_score', 0.0)):.3f}",
                        str(bool(seg.get("review_required", False))),
                        seg.get("review_reason", "") or "",
                    ]
                )
        return {"applied": applied, "review_queue_count": payload["review_queue_count"]}
