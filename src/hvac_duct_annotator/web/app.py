from __future__ import annotations

from pathlib import Path
import json
from pydantic import BaseModel

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from hvac_duct_annotator.web.jobs import JobStore

BASE = Path(".hvac_jobs")
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
store = JobStore(BASE / "runs")

app = FastAPI(title="HVAC Duct Annotator")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


class CorrectionItem(BaseModel):
    id: str
    classification: str | None = None
    dimensions: dict | None = None
    review_required: bool | None = None
    review_reason: str | None = None


class CorrectionPayload(BaseModel):
    corrections: list[CorrectionItem]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request, context={})


@app.post("/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF is supported")
    upload_path = UPLOADS / file.filename
    upload_path.write_bytes(await file.read())
    state = store.create(upload_path)
    background_tasks.add_task(store.run, state.id)
    return {"job_id": state.id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    payload = {"job_id": state.id, "status": state.status, "error": state.error}
    if state.status == "done" and state.report_json.exists():
        data = json.loads(state.report_json.read_text(encoding="utf-8"))
        payload["review_queue_count"] = int(data.get("review_queue_count", 0))
        payload["segments_total"] = len(data.get("segments", []))
    return payload


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if state.status != "done":
        raise HTTPException(status_code=409, detail="Not completed")
    return FileResponse(state.output_pdf, media_type="application/pdf", filename=f"{job_id}_annotated.pdf")


@app.get("/jobs/{job_id}/report.json")
async def get_report_json(job_id: str):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if state.status != "done":
        raise HTTPException(status_code=409, detail="Not completed")
    return FileResponse(state.report_json, media_type="application/json", filename=f"{job_id}.json")


@app.get("/jobs/{job_id}/report.csv")
async def get_report_csv(job_id: str):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if state.status != "done":
        raise HTTPException(status_code=409, detail="Not completed")
    return FileResponse(state.report_csv, media_type="text/csv", filename=f"{job_id}.csv")


@app.get("/jobs/{job_id}/review", response_class=HTMLResponse)
async def review_page(request: Request, job_id: str):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if state.status != "done":
        raise HTTPException(status_code=409, detail="Not completed")
    payload = json.loads(state.report_json.read_text(encoding="utf-8"))
    review_rows = [s for s in payload.get("segments", []) if s.get("review_required")]
    return templates.TemplateResponse(
        name="review.html",
        request=request,
        context={"job_id": job_id, "rows": review_rows},
    )


@app.get("/jobs/{job_id}/review.csv")
async def review_csv(job_id: str):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if state.status != "done":
        raise HTTPException(status_code=409, detail="Not completed")
    return FileResponse(state.review_csv, media_type="text/csv", filename=f"{job_id}_review.csv")


@app.post("/jobs/{job_id}/corrections")
async def apply_corrections(job_id: str, payload: CorrectionPayload):
    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if state.status != "done":
        raise HTTPException(status_code=409, detail="Not completed")
    result = store.apply_corrections(job_id, [c.model_dump() for c in payload.corrections])
    return {"job_id": job_id, **result}


def run_dev() -> None:
    uvicorn.run("hvac_duct_annotator.web.app:app", host="127.0.0.1", port=8000, reload=False)
