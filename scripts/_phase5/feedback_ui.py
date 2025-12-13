from __future__ import annotations

import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

API_BASE = os.environ.get("SEARCH_API_BASE", "http://localhost:8000").rstrip("/")

app = FastAPI(title="IDR Feedback UI")

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>IDR Search Feedback</title>
  <style>
    body { font-family: system-ui, Arial; margin: 24px; max-width: 980px; }
    input[type=text] { width: 78%; padding: 10px; font-size: 16px; }
    button { padding: 10px 14px; font-size: 14px; margin-left: 6px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 12px 0; }
    .meta { color: #444; font-size: 13px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .small { font-size: 12px; color: #666; }
    .pill { display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; margin-right:6px; font-size:12px; }
    textarea { width: 100%; min-height: 54px; padding: 8px; }
  </style>
</head>
<body>
  <h2>IDR Search (Feedback build)</h2>
  <div class="row">
    <input id="q" type="text" placeholder="Type query (Hindi or Roman)…"/>
    <button onclick="doSearch()">Search</button>
    <button onclick="markNone()">Nothing relevant</button>
  </div>
  <div class="small" id="qid"></div>
  <div id="results"></div>

<script>
let lastQueryId = null;
let lastResults = [];

async function doSearch() {
  const q = document.getElementById("q").value.trim();
  if (!q) return;

  const res = await fetch("/proxy_search", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({query:q, per_page:10, explain:false})
  });

  const data = await res.json();
  lastQueryId = data.query_id;
  lastResults = data.results || [];

  document.getElementById("qid").innerText =
    "query_id: " + lastQueryId +
    " | mode: " + data.mode +
    " | query_used: " + data.query_used;

  const root = document.getElementById("results");
  root.innerHTML = "";

  lastResults.forEach(hit => {
    const card = document.createElement("div");
    card.className = "card";

    card.innerHTML = `
      <div><b>${hit.title || ""}</b></div>
      <div class="meta">
        Date: ${hit.date || ""} |
        ID: ${hit.id} |
        <a href="${hit.url || "#"}" target="_blank">open</a>
      </div>
      <div class="small">${(hit.summary || "").slice(0, 380)}</div>
      <div class="small">
        ${hit.primary_category ? `<span class="pill">Primary: ${hit.primary_category}</span>` : ""}
        ${hit.partner_label ? `<span class="pill">Partner: ${hit.partner_label}</span>` : ""}
        ${(hit.location || []).slice(0,4).map(x=>`<span class="pill">Loc: ${x}</span>`).join("")}
        ${(hit.tags || []).slice(0,4).map(x=>`<span class="pill">Tag: ${x}</span>`).join("")}
      </div>
    `;

    const note = document.createElement("textarea");
    note.placeholder = "Optional note (why relevant / irrelevant)…";

    const row = document.createElement("div");
    row.className = "row";

    const okBtn = document.createElement("button");
    okBtn.innerText = "Correct";
    okBtn.onclick = () => labelItem(hit.id, 1, note.value);

    const badBtn = document.createElement("button");
    badBtn.innerText = "Wrong";
    badBtn.onclick = () => labelItem(hit.id, 0, note.value);

    row.appendChild(okBtn);
    row.appendChild(badBtn);

    card.appendChild(note);
    card.appendChild(row);
    root.appendChild(card);
  });
}

async function labelItem(articleId, label, note) {
  if (!lastQueryId) return;
  await fetch("/proxy_label", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      query_id: lastQueryId,
      article_id: articleId,
      label: label,
      note: note || null
    })
  });
}

async function markNone() {
  if (!lastQueryId) return;
  const note = prompt("Optional note (what did you expect?)") || null;
  await fetch("/proxy_label_query", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      query_id: lastQueryId,
      label: 0,
      note: note
    })
  });
}
</script>

</body>
</html>
"""

class SearchReq(BaseModel):
    query: str
    per_page: int = 10
    explain: bool = False

class LabelReq(BaseModel):
    query_id: int
    article_id: str | None = None
    label: int
    note: str | None = None

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.post("/proxy_search")
def proxy_search(req: SearchReq):
    r = requests.post(f"{API_BASE}/search", json=req.model_dump(), timeout=120)
    return JSONResponse(r.json(), status_code=r.status_code)

@app.post("/proxy_label")
def proxy_label(req: LabelReq):
    r = requests.post(f"{API_BASE}/label", json=req.model_dump(), timeout=60)
    return JSONResponse(r.json(), status_code=r.status_code)

@app.post("/proxy_label_query")
def proxy_label_query(req: LabelReq):
    r = requests.post(f"{API_BASE}/label_query", json={"query_id": req.query_id, "label": 0, "note": req.note}, timeout=60)
    return JSONResponse(r.json(), status_code=r.status_code)

@app.get("/favicon.ico")
def favicon():
    return {}