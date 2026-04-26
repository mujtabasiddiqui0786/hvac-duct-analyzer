from __future__ import annotations

import json
from pathlib import Path

import click

from hvac_duct_annotator.pipeline import run_pipeline


@click.command()
@click.argument("input_pdf", type=click.Path(path_type=Path, exists=True))
@click.option("-o", "--output", "output_pdf", required=True, type=click.Path(path_type=Path))
@click.option("--report", "report_json", type=click.Path(path_type=Path), default=None)
@click.option("--report-csv", "report_csv", type=click.Path(path_type=Path), default=None)
@click.option("--dpi", default=300, show_default=True, type=int)
@click.option("--scale", default="auto", show_default=True)
@click.option("--classify/--no-classify", default=True, show_default=True)
def main(
    input_pdf: Path,
    output_pdf: Path,
    report_json: Path | None,
    report_csv: Path | None,
    dpi: int,
    scale: str,
    classify: bool,
) -> None:
    """Annotate HVAC ducts in INPUT_PDF."""
    report = run_pipeline(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        dpi=dpi,
        scale_mode=scale,
        classify=classify,
        report_csv=report_csv,
    )
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    click.echo(f"Annotated PDF: {output_pdf}")
    if report_json:
        click.echo(f"JSON report: {report_json}")
    if report_csv:
        click.echo(f"CSV report: {report_csv}")
