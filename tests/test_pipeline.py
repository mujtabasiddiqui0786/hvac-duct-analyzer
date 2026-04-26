from __future__ import annotations

from pathlib import Path

from hvac_duct_annotator.pipeline import run_pipeline


def test_pipeline_runs(sample_pdf: Path, tmp_path: Path) -> None:
    out_pdf = tmp_path / "annotated.pdf"
    out_csv = tmp_path / "report.csv"
    report = run_pipeline(input_pdf=sample_pdf, output_pdf=out_pdf, report_csv=out_csv, classify=True)
    assert out_pdf.exists()
    assert out_csv.exists()
    assert report.summary["total"] >= 1
