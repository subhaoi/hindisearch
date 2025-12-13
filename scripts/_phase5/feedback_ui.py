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
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IDR Search Feedback</title>
  <style>
    :root{
      --red:#d93b3b;
      --text:#111;
      --muted:#6b6b6b;
      --line:#e6e6e6;
      --bg:#ffffff;
      --link:#1aa6a6;
    }
    body{
      margin:0;
      background:var(--bg);
      color:var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    .wrap{ max-width: 980px; margin: 0 auto; padding: 22px 18px 60px; }

    .searchbar{
      display:flex; gap:10px; align-items:stretch;
      margin-bottom: 14px;
    }
    .searchbar input{
      flex:1;
      border:1px solid var(--line);
      padding: 12px 12px;
      font-size:16px;
      outline:none;
    }
    .searchbar button{
      border:0;
      background:var(--red);
      color:white;
      padding: 0 18px;
      font-weight:700;
      letter-spacing:.08em;
      text-transform:uppercase;
      cursor:pointer;
    }

    .controls{
      display:flex;
      gap:18px;
      align-items:center;
      margin: 8px 0 18px;
    }
    .controls .label{
      width: 80px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing:.18em;
      text-transform:uppercase;
      color:#000;
    }
    .controls .opts{
      display:flex; gap:14px; align-items:center; flex-wrap:wrap;
      font-size:12px;
      letter-spacing:.10em;
      text-transform:uppercase;
      color:var(--muted);
    }
    .controls .opts a{
      color:inherit;
      text-decoration:none;
      cursor:pointer;
      padding: 2px 0;
      border-bottom: 2px solid transparent;
    }
    .controls .opts a.active{
      color:#000;
      border-bottom-color:#000;
    }
    .sep{ color: var(--line); }

    .qid{
      margin: 10px 0 14px;
      font-size: 12px;
      color: var(--muted);
    }

    .result{
      display:flex;
      gap: 22px;
      padding: 18px 0;
      border-top: 1px solid var(--line);
    }
    .result:first-child{ border-top: 0; }
    .left{ flex: 1; min-width: 0; }
    .right{ width: 0; display:none; } /* no image */

    .kicker{
      font-size: 12px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .date{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .title{
      font-size: 22px;
      line-height: 1.15;
      font-weight: 800;
      margin: 0 0 8px;
    }
    .summary{
      font-size: 14px;
      line-height: 1.45;
      color: #2a2a2a;
      margin: 0 0 10px;
      max-width: 72ch;
    }
    .meta{
      font-size: 12px;
      color: var(--muted);
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      align-items:center;
      margin-bottom: 10px;
    }
    .meta a{ color: var(--link); text-decoration:none; font-weight:700; }
    .pill{
      border:1px solid var(--line);
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      color: #333;
      background: #fff;
    }
    textarea{
      width:100%;
      border:1px solid var(--line);
      padding:10px;
      font-size:13px;
      resize:vertical;
      margin-top: 10px;
    }
    .actions{
      display:flex;
      gap:10px;
      margin-top: 10px;
    }
    .btn{
      border:1px solid var(--line);
      background:#fff;
      padding: 9px 12px;
      cursor:pointer;
      font-weight:700;
      text-transform:uppercase;
      letter-spacing:.08em;
      font-size:12px;
    }
    .btn.red{
      border-color: var(--red);
      color: var(--red);
    }
    .btn.black{
      border-color:#000;
      color:#000;
    }
    .footer-actions{
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 18px;
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
      flex-wrap:wrap;
    }
    .hint{ font-size:12px; color: var(--muted); }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="searchbar">
      <input id="q" placeholder="Search (Devanagari or Roman Hindi)..." />
      <button id="submitBtn" onclick="doSearch()">Submit</button>
    </div>

    <div class="controls">
      <div class="label">SORT BY</div>
      <div class="opts">
        <a id="sortRel" class="active" onclick="setSort('relevance')">RELEVANCE</a>
        <span class="sep">|</span>
        <a id="sortNew" onclick="setSort('newest')">NEWEST</a>
        <span class="sep">|</span>
        <a id="sortOld" onclick="setSort('oldest')">OLDEST</a>
      </div>
    </div>

    <div class="qid" id="qid">query_id: -</div>

    <div id="results"></div>

    <div class="footer-actions">
      <button class="btn black" onclick="markNone()">Nothing relevant</button>
      <div class="hint">Labels are saved immediately. Use “Wrong” even if result is close-but-not-right.</div>
    </div>
  </div>

  <script>
    let lastQueryId = null;
    let lastMode = null;
    let lastQueryUsed = null;
    let lastQuerySemantic = null;

    let lastResults = [];     // raw from API
    let currentSort = "relevance";

    function setActiveSortUI(){
      document.getElementById("sortRel").classList.toggle("active", currentSort==="relevance");
      document.getElementById("sortNew").classList.toggle("active", currentSort==="newest");
      document.getElementById("sortOld").classList.toggle("active", currentSort==="oldest");
    }

    function parseDateSafe(s){
      if(!s) return null;
      const t = Date.parse(s);
      return Number.isFinite(t) ? t : null;
    }

    function sortedResults(){
      const arr = [...lastResults];
      if(currentSort === "relevance") return arr;

      arr.sort((a,b)=>{
        const ta = parseDateSafe(a.date);
        const tb = parseDateSafe(b.date);
        if(ta === null && tb === null) return 0;
        if(ta === null) return 1;
        if(tb === null) return -1;
        return currentSort === "newest" ? (tb-ta) : (ta-tb);
      });
      return arr;
    }

    function setSort(mode){
      currentSort = mode;
      setActiveSortUI();
      renderResults();
    }

    function renderResults(){
      const root = document.getElementById("results");
      root.innerHTML = "";

      const results = sortedResults();

      results.forEach((hit, idx) => {
        const card = document.createElement("div");
        card.className = "result";

        const left = document.createElement("div");
        left.className = "left";

        const kicker = document.createElement("div");
        kicker.className = "kicker";
        kicker.textContent = (hit.primary_category || "").toUpperCase();

        const date = document.createElement("div");
        date.className = "date";
        date.textContent = hit.date || "";

        const title = document.createElement("div");
        title.className = "title";
        title.textContent = hit.title || "";

        const summary = document.createElement("div");
        summary.className = "summary";
        summary.textContent = hit.summary || "";

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `
          <span class="pill">ID: ${hit.id}</span>
          ${hit.url ? `<a href="${hit.url}" target="_blank" rel="noreferrer">Open</a>` : ``}
          ${hit.partner_label ? `<span class="pill">Partner: ${hit.partner_label}</span>` : ``}
          ${(hit.location || []).slice(0,3).map(x=>`<span class="pill">Loc: ${x}</span>`).join("")}
          ${(hit.tags || []).slice(0,3).map(x=>`<span class="pill">Tag: ${x}</span>`).join("")}
        `;

        const note = document.createElement("textarea");
        note.placeholder = "Optional note (why relevant / irrelevant)…";

        const actions = document.createElement("div");
        actions.className = "actions";

        const okBtn = document.createElement("button");
        okBtn.className = "btn black";
        okBtn.textContent = "Correct";
        okBtn.onclick = () => labelItem(hit.id, 1, note.value);

        const badBtn = document.createElement("button");
        badBtn.className = "btn red";
        badBtn.textContent = "Wrong";
        badBtn.onclick = () => labelItem(hit.id, 0, note.value);

        actions.appendChild(okBtn);
        actions.appendChild(badBtn);

        if(kicker.textContent) left.appendChild(kicker);
        left.appendChild(date);
        left.appendChild(title);
        left.appendChild(summary);
        left.appendChild(meta);
        left.appendChild(note);
        left.appendChild(actions);

        card.appendChild(left);
        root.appendChild(card);
      });
    }

    async function doSearch() {
      const q = document.getElementById("q").value.trim();
      if (!q) return;

      const res = await fetch("/proxy_search", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({query:q, per_page:50, explain:false})
      });

      const data = await res.json();
      lastQueryId = data.query_id;
      lastMode = data.mode;
      lastQueryUsed = data.query_used;
      lastQuerySemantic = data.query_semantic;

      document.getElementById("qid").innerText =
        `query_id: ${lastQueryId} | mode: ${lastMode} | query_used: ${lastQueryUsed}`;

      lastResults = data.results || [];
      currentSort = "relevance";
      setActiveSortUI();
      renderResults();
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
        body: JSON.stringify({ query_id: lastQueryId, label: 0, note })
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