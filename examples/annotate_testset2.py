from __future__ import annotations

from pathlib import Path

from hvac_duct_annotator.pipeline import run_pipeline


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    input_pdf = root / "testset2.pdf"
    if not input_pdf.exists():
        input_pdf = Path(
            "/Users/Mujtaba/Library/Application Support/Cursor/User/workspaceStorage/09d996dda1279ef33170d222662fa47d/pdfs/454cc657-0c59-48e6-89b7-502883ea0b05/testset2.pdf"
        )
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_pipeline(
        input_pdf=input_pdf,
        output_pdf=output_dir / "annotated_testset2.pdf",
        report_csv=output_dir / "report_testset2.csv",
    )
    (output_dir / "report_testset2.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"Annotated {report.summary['total']} segments")


if __name__ == "__main__":
    main()
