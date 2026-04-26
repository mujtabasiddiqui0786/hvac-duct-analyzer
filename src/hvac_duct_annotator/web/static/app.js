const statusNode = document.getElementById("status");
const linksNode = document.getElementById("links");
const submitButton = document.getElementById("submit");
const fileInput = document.getElementById("file");
const pickedNode = document.getElementById("picked");
const jobIdNode = document.getElementById("jobId");
const segmentsNode = document.getElementById("segmentsTotal");
const reviewNode = document.getElementById("reviewQueue");
const dropzone = document.getElementById("dropzone");

function setPicked(file) {
  pickedNode.textContent = file ? `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)` : "";
}

async function pollJob(jobId) {
  submitButton.disabled = true;
  while (true) {
    const res = await fetch(`/jobs/${jobId}`);
    const data = await res.json();
    statusNode.textContent = `Job ${jobId}: ${data.status}`;
    jobIdNode.textContent = jobId;
    segmentsNode.textContent = data.segments_total ?? "-";
    reviewNode.textContent = data.review_queue_count ?? "-";
    if (data.status === "done") {
      linksNode.innerHTML = `
        <strong>Artifacts</strong><br/><br/>
        <a href="/jobs/${jobId}/result" target="_blank">Annotated PDF</a>
        <a href="/jobs/${jobId}/report.json" target="_blank">JSON report</a>
        <a href="/jobs/${jobId}/report.csv" target="_blank">CSV report</a>
        <a href="/jobs/${jobId}/review" target="_blank">Review Queue</a>
        <a href="/jobs/${jobId}/review.csv" target="_blank">Review CSV</a>
      `;
      submitButton.disabled = false;
      break;
    }
    if (data.status === "failed") {
      statusNode.textContent = `Job failed: ${data.error}`;
      submitButton.disabled = false;
      break;
    }
    await new Promise((r) => setTimeout(r, 1200));
  }
}

submitButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    statusNode.textContent = "Pick a PDF first.";
    return;
  }
  const form = new FormData();
  form.append("file", file);
  statusNode.textContent = "Uploading...";
  const res = await fetch("/jobs", { method: "POST", body: form });
  if (!res.ok) {
    statusNode.textContent = "Upload failed.";
    return;
  }
  const data = await res.json();
  statusNode.textContent = `Queued: ${data.job_id}`;
  jobIdNode.textContent = data.job_id;
  linksNode.innerHTML = "";
  await pollJob(data.job_id);
});

fileInput.addEventListener("change", () => setPicked(fileInput.files[0]));

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const files = e.dataTransfer?.files;
  if (!files || !files.length) return;
  fileInput.files = files;
  setPicked(files[0]);
});
