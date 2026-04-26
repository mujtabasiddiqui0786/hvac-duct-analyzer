const statusNode = document.getElementById("status");
const linksNode = document.getElementById("links");
const submitButton = document.getElementById("submit");
const fileInput = document.getElementById("file");

async function pollJob(jobId) {
  while (true) {
    const res = await fetch(`/jobs/${jobId}`);
    const data = await res.json();
    statusNode.textContent = `Job ${jobId}: ${data.status}`;
    if (data.status === "done") {
      linksNode.innerHTML = `
        <a href="/jobs/${jobId}/result" target="_blank">Annotated PDF</a>
        <a href="/jobs/${jobId}/report.json" target="_blank">JSON report</a>
        <a href="/jobs/${jobId}/report.csv" target="_blank">CSV report</a>
      `;
      break;
    }
    if (data.status === "failed") {
      statusNode.textContent = `Job failed: ${data.error}`;
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
  linksNode.innerHTML = "";
  await pollJob(data.job_id);
});
