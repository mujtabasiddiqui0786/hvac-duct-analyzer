from __future__ import annotations

import json
import uuid
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
            state.status = "done"
        except Exception as exc:  # pragma: no cover
            state.status = "failed"
            state.error = str(exc)
