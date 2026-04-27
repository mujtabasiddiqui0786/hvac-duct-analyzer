# HVAC Duct Annotator

Python toolkit to detect, measure, and annotate HVAC ductwork from mechanical plan PDFs.

## Features

- Vector-first duct candidate extraction from CAD-like PDF drawings.
- OCR-based dimension extraction for flattened text (`12x14`, `14\"ø`, etc.).
- Duct length estimation from drawing scale (`1/4\" = 1'-0\"`) or calibrated fallback.
- Annotated PDF output with blue highlighted duct traces and label badges.
- JSON and CSV output reports.
- CLI and FastAPI web UI.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI usage

```bash
hvac-annotate /path/to/testset2.pdf -o output/annotated.pdf \
  --report output/report.json \
  --report-csv output/report.csv \
  --dpi 300 \
  --scale auto
```

### Tested sample command

```bash
python3 examples/annotate_testset2.py
```

Generated files:

- `output/annotated_testset2.pdf`
- `output/report_testset2.json`
- `output/report_testset2.csv`

## Web UI

```bash
hvac-annotate-web
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Pipeline modules

- `render.py`: render PDF page and coordinate transforms.
- `vector.py`: extract duct candidates from vector lines and circles.
- `detect.py`: hatch/line-density validation.
- `ocr.py`: OCR and duct-dimension parser.
- `associate.py`: assign labels to segments.
- `measure.py`: calculate feet/inches.
- `annotate.py`: output annotated PDF + reports.
- `propagate.py`: propagate dimensions over collinear continuity.
- `validate.py`: composite confidence + review flags.
- `symbols.py`: symbol anchor detection for supply/return logic.

## Testing

```bash
python3 -m pytest -q
```

If your sample PDF is in a different location, set:

```bash
export HVAC_SAMPLE_PDF="/absolute/path/to/testset2.pdf"
```

### Accuracy harness

```bash
python3 examples/evaluate_accuracy.py
```

For real precision/recall, populate `tests/fixtures/ground_truth.json -> segments` with fully-labeled ducts.

### Review corrections API

After a web job is complete, you can import manual fixes:

```bash
curl -X POST "http://127.0.0.1:8000/jobs/<job_id>/corrections" \
  -H "Content-Type: application/json" \
  -d '{
    "corrections":[
      {"id":"D-0007","classification":"supply","review_required":false,"review_reason":"approved manually"}
    ]
  }'
```

## OCR dependency note (`_lzma` missing)

If you see `OCR disabled due to missing module dependency: No module named '_lzma'`, your Python build is missing stdlib lzma support.
The app now tries a `pytesseract` fallback when EasyOCR initialization fails; for best results, use a Python build with lzma enabled.
