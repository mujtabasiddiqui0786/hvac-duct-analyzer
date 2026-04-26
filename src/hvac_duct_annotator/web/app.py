from __future__ import annotations

from pathlib import Path

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
    return {"job_id": state.id, "status": state.status, "error": state.error}


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


def run_dev() -> None:
    uvicorn.run("hvac_duct_annotator.web.app:app", host="127.0.0.1", port=8000, reload=False)
