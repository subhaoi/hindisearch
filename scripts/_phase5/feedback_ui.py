from __future__ import annotations

import json
import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

API_BASE = os.environ.get("SEARCH_API_BASE", "http://localhost:8000").rstrip("/")
# hi (default) or en: which site's search this page tests, and the language of its wording
SITE = os.environ.get("FEEDBACK_SITE", "hi").strip().lower()

app = FastAPI(title="IDR Feedback UI")

# Everything a tester reads. The testers are writers and editors, so plain words, no jargon.
WORDING = {
    "hi": {
        "lang": "hi",
        "locale": "hi-IN",
        "page_title": "आईडीआर हिंदी सर्च: टेस्टिंग",
        "heading": "आईडीआर हिंदी की नई सर्च को परखें",
        "intro": "मदद के लिए धन्यवाद! इसमें बस कुछ मिनट लगेंगे।",
        "steps": [
            "कुछ भी खोजें, जैसे आप वेबसाइट पर खोजते हैं। हिंदी या अंग्रेज़ी अक्षरों में, दोनों चलेगा।",
            "हर नतीजे के नीचे बताएं: 👍 काम का है या 👎 काम का नहीं।",
            "अगर कोई भी नतीजा काम का नहीं, तो नतीजों के नीचे वाला बटन दबाएं।",
        ],
        "placeholder": "जैसे: महिला किसान, mahila yojana bihar",
        "search": "खोजें",
        "examples_label": "ये आज़माएं:",
        "examples": ["महिला किसान", "शिक्षा नीति", "mahila yojana bihar", "जलवायु परिवर्तन"],
        "searching": "खोज रहे हैं…",
        "no_results": "कोई नतीजा नहीं मिला। दूसरे शब्दों में खोजकर देखें।",
        "error": "कुछ गड़बड़ हो गई। एक बार फिर कोशिश करें।",
        "count": "{n} नतीजे",
        "sort_best": "सबसे सटीक",
        "sort_newest": "सबसे नया पहले",
        "sort_oldest": "सबसे पुराना पहले",
        "useful": "👍 काम का है",
        "not_useful": "👎 काम का नहीं",
        "comment": "टिप्पणी जोड़ें (ज़रूरी नहीं)",
        "comment_placeholder": "क्यों? जैसे: पुराना है, विषय अलग है…",
        "saving": "सेव हो रहा है…",
        "saved_useful": "✓ सेव हो गया: काम का है",
        "saved_not_useful": "✓ सेव हो गया: काम का नहीं",
        "save_failed": "सेव नहीं हुआ। फिर से दबाएं।",
        "rated": "आपने इस खोज के {n} नतीजे परखे। धन्यवाद!",
        "show_more": "और नतीजे दिखाएं",
        "nothing_useful": "इनमें से कोई भी नतीजा काम का नहीं",
        "nothing_prompt": "आप क्या ढूंढ रहे थे? (ज़रूरी नहीं)",
        "nothing_saved": "✓ सेव हो गया: इस खोज का कोई नतीजा काम का नहीं। धन्यवाद!",
        "by": "लेखक:",
        "also_in": "इन भाषाओं में भी",
        "types": {},
    },
    "en": {
        "lang": "en",
        "locale": "en-GB",
        "page_title": "IDR search: testing",
        "heading": "Help us test IDR's new search",
        "intro": "Thank you for helping! It only takes a few minutes.",
        "steps": [
            "Search for anything you might look for on IDR.",
            "Under each result, tell us: 👍 Useful or 👎 Not useful for that search.",
            "If none of the results fit, press the button below the results.",
        ],
        "placeholder": "For example: water crisis in Rajasthan",
        "search": "Search",
        "examples_label": "Try:",
        "examples": ["menstrual health", "CSR funding", "podcast on philanthropy", "Rachita Vora"],
        "searching": "Searching…",
        "no_results": "No results. Try searching with other words.",
        "error": "Something went wrong. Please try again.",
        "count": "{n} results",
        "sort_best": "Best match",
        "sort_newest": "Newest first",
        "sort_oldest": "Oldest first",
        "useful": "👍 Useful",
        "not_useful": "👎 Not useful",
        "comment": "Add a comment (optional)",
        "comment_placeholder": "Why? For example: too old, different topic…",
        "saving": "Saving…",
        "saved_useful": "✓ Saved: useful",
        "saved_not_useful": "✓ Saved: not useful",
        "save_failed": "Not saved. Please press again.",
        "rated": "You've rated {n} results for this search. Thank you!",
        "show_more": "Show more results",
        "nothing_useful": "Nothing useful in these results",
        "nothing_prompt": "What were you looking for? (optional)",
        "nothing_saved": "✓ Saved: nothing useful for this search. Thank you!",
        "by": "By",
        "also_in": "Also in",
        "types": {"article": "Article", "feature": "Feature", "ground_up_story": "Ground Up story",
                  "post": "Post", "podcast": "Podcast"},
    },
}
T = WORDING.get(SITE, WORDING["hi"])

HTML = """
<!doctype html>
<html lang="__LANG__">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAGE_TITLE__</title>
  <style>
    :root{
      --red:#d93b3b; --red-soft:#fdeeee; --green:#0a7a3b; --green-soft:#e8f5ee;
      --text:#1a1a1a; --muted:#666; --line:#e4e4e4; --bg:#f6f6f4; --card:#ffffff;
    }
    *{ box-sizing:border-box; }
    body{ margin:0; background:var(--bg); color:var(--text); font-size:16px; line-height:1.5;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans Devanagari", Helvetica, Arial; }
    .topbar{ background:var(--red); height:6px; }
    .wrap{ max-width: 860px; margin: 0 auto; padding: 24px 16px 80px; }

    h1{ font-size: 26px; line-height:1.25; margin: 0 0 6px; }
    .intro{ color: var(--muted); margin: 0 0 14px; }
    .howto{ background: var(--card); border:1px solid var(--line); border-radius: 12px; padding: 14px 18px; margin-bottom: 18px; }
    .howto ol{ margin: 0; padding-left: 22px; }
    .howto li{ margin: 4px 0; }

    .searchbar{ display:flex; gap:10px; }
    .searchbar input{ flex:1; min-width:0; border:2px solid #cfcfcf; border-radius: 10px; padding: 14px 14px; font-size:18px; outline:none; background:#fff; }
    .searchbar input:focus{ border-color: var(--red); }
    .searchbar button{ border:0; border-radius: 10px; background:var(--red); color:#fff; padding: 0 22px; font-size:17px; font-weight:700; cursor:pointer; }
    .examples{ margin: 10px 0 0; display:flex; flex-wrap:wrap; gap:8px; align-items:center; color: var(--muted); font-size:14px; }
    .chip{ border:1px solid var(--line); background:#fff; border-radius: 999px; padding: 5px 12px; font-size:14px; cursor:pointer; color: var(--text); }
    .chip:hover{ border-color: var(--red); }

    .bar{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin: 22px 0 10px; }
    .count{ color: var(--muted); font-size: 15px; }
    .sort{ display:flex; border:1px solid var(--line); border-radius: 10px; overflow:hidden; background:#fff; }
    .sort button{ border:0; background:transparent; padding: 8px 14px; font-size:14px; cursor:pointer; color: var(--muted); }
    .sort button.active{ background: var(--text); color:#fff; }

    .message{ background: var(--card); border:1px solid var(--line); border-radius: 12px; padding: 18px; text-align:center; color: var(--muted); }
    .thanks{ background: var(--green-soft); color: var(--green); border-radius: 10px; padding: 10px 14px; margin: 0 0 12px; font-weight:600; display:none; }

    .result{ display:flex; gap:16px; background: var(--card); border:1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    .result.rated-1{ border-color: #9fd3b4; }
    .result.rated-0{ border-color: #efb3b3; }
    .thumb{ width:140px; flex:0 0 140px; height:96px; border-radius: 8px; overflow:hidden; background:#eee; }
    .thumb img{ width:100%; height:100%; object-fit:cover; display:block; }
    .body{ flex:1; min-width:0; }
    .meta-top{ display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center; font-size:14px; color: var(--muted); margin-bottom: 4px; }
    .date{ font-weight:700; color: var(--text); }
    .badge{ background:#f0f0f0; border-radius: 6px; padding: 1px 8px; font-size:13px; color:#333; }
    .langs{ display:inline-flex; align-items:center; gap:5px; margin-left:auto; }
    .langs .also{ font-size:13px; color: var(--muted); margin-right: 3px; }
    .lang{ display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:6px;
      color:#fff; font-weight:800; font-size:11px; letter-spacing:.04em; text-decoration:none; }
    .lang:hover{ filter: brightness(1.15); }
    .title{ font-size: 20px; line-height: 1.3; font-weight: 700; margin: 0 0 6px; }
    .title a{ color: var(--text); text-decoration:none; }
    .title a:hover{ text-decoration: underline; }
    .summary{ font-size: 15px; color: #333; margin: 0 0 6px; }
    .authors{ font-size: 14px; color: var(--muted); margin: 0 0 10px; }

    .actions{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .vote{ border:2px solid var(--line); background:#fff; border-radius: 10px; padding: 9px 16px; font-size:15px; font-weight:600; cursor:pointer; }
    .vote.yes:hover, .vote.yes.on{ border-color: var(--green); background: var(--green-soft); color: var(--green); }
    .vote.no:hover, .vote.no.on{ border-color: var(--red); background: var(--red-soft); color: var(--red); }
    .linkbtn{ border:0; background:none; color: var(--muted); text-decoration: underline; cursor:pointer; font-size:14px; padding: 4px; }
    .status{ font-size: 14px; }
    .status.ok{ color: var(--green); }
    .status.bad{ color: var(--red); }
    textarea{ width:100%; border:1px solid #cfcfcf; border-radius: 8px; padding:10px; font-size:15px; margin-top: 10px; display:none; font-family: inherit; }

    .more{ display:block; margin: 6px auto 0; }
    .nothing{ display:none; margin-top: 18px; text-align:center; }
    .nothing button{ border:2px solid var(--text); background:#fff; border-radius: 10px; padding: 12px 20px; font-size:16px; font-weight:700; cursor:pointer; }
    .nothing .status{ display:block; margin-top: 8px; }

    @media (max-width: 600px){
      h1{ font-size: 22px; }
      .result{ flex-direction: column; }
      .thumb{ width:100%; flex-basis:auto; height:160px; }
      .searchbar button{ padding: 0 16px; }
    }
  </style>
</head>

<body>
  <div class="topbar"></div>
  <div class="wrap">
    <h1 id="heading"></h1>
    <p class="intro" id="intro"></p>
    <div class="howto"><ol id="steps"></ol></div>

    <form class="searchbar" onsubmit="doSearch(); return false;">
      <input id="q" autocomplete="off" />
      <button type="submit" id="searchBtn"></button>
    </form>
    <div class="examples" id="examples"></div>

    <div class="bar" id="bar" style="display:none;">
      <div class="count" id="count"></div>
      <div class="sort">
        <button id="sortBest" class="active" onclick="setSort('best')"></button>
        <button id="sortNew" onclick="setSort('newest')"></button>
        <button id="sortOld" onclick="setSort('oldest')"></button>
      </div>
    </div>

    <div class="thanks" id="thanks"></div>
    <div id="results"></div>
    <button class="vote more" id="more" style="display:none;" onclick="showMore()"></button>

    <div class="nothing" id="nothing">
      <button onclick="markNone()" id="nothingBtn"></button>
      <span class="status" id="nothingStatus"></span>
    </div>
  </div>

  <script>
    const T = __WORDING__;

    let queryId = null;
    let results = [];
    let sortMode = "best";
    let visible = 10;
    let ratings = {};   // article id -> 1 / 0, for this search

    // The same piece in other languages (English site): colour and name in its own script
    const LANGS = {HI: ["हिंदी", "#B45309"], BN: ["বাংলা", "#15803D"], MR: ["मराठी", "#7E22CE"],
                   GU: ["ગુજરાતી", "#1D4ED8"], KN: ["ಕನ್ನಡ", "#BE185D"]};
    function langSquares(list){
      const items = (list || []).filter(x => LANGS[x.lang]);
      if(!items.length) return "";
      return `<span class="langs"><span class="also">${escapeHtml(T.also_in)}</span>` + items.map(x =>
        `<a class="lang" style="background:${LANGS[x.lang][1]}" href="${escapeHtml(x.url)}" target="_blank" rel="noreferrer" title="${LANGS[x.lang][0]}">${x.lang}</a>`
      ).join("") + `</span>`;
    }

    function el(id){ return document.getElementById(id); }
    function fmt(s, n){ return s.replace("{n}", n); }
    function escapeHtml(s){
      return String(s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
        .replaceAll('"',"&quot;").replaceAll("'","&#039;");
    }
    function dateValue(s){ const t = Date.parse(s || ""); return Number.isFinite(t) && t > 0 ? t : null; }
    function niceDate(s){
      const t = dateValue(s);
      if(t === null) return "";
      return new Date(t).toLocaleDateString(T.locale, {day:"numeric", month:"long", year:"numeric"});
    }

    function setup(){
      document.title = T.page_title;
      el("heading").textContent = T.heading;
      el("intro").textContent = T.intro;
      el("steps").innerHTML = T.steps.map(s => `<li>${escapeHtml(s)}</li>`).join("");
      el("q").placeholder = T.placeholder;
      el("searchBtn").textContent = T.search;
      el("examples").innerHTML = `<span>${escapeHtml(T.examples_label)}</span>` +
        T.examples.map(x => `<button class="chip" type="button">${escapeHtml(x)}</button>`).join("");
      el("examples").querySelectorAll(".chip").forEach(c => c.onclick = () => { el("q").value = c.textContent; doSearch(); });
      el("sortBest").textContent = T.sort_best;
      el("sortNew").textContent = T.sort_newest;
      el("sortOld").textContent = T.sort_oldest;
      el("more").textContent = T.show_more;
      el("nothingBtn").textContent = T.nothing_useful;
      el("q").focus();
      // A shared link like ...?q=CSR+funding runs that search straight away
      const q = new URLSearchParams(location.search).get("q");
      if(q){ el("q").value = q; doSearch(); }
    }

    function setSort(mode){
      sortMode = mode;
      el("sortBest").classList.toggle("active", mode === "best");
      el("sortNew").classList.toggle("active", mode === "newest");
      el("sortOld").classList.toggle("active", mode === "oldest");
      render();
    }

    function ordered(){
      const arr = [...results];
      if(sortMode === "newest"){
        arr.sort((a,b) => (dateValue(b.date) || 0) - (dateValue(a.date) || 0));
      }else if(sortMode === "oldest"){
        // Undated results go last, not first
        arr.sort((a,b) => (dateValue(a.date) || Infinity) - (dateValue(b.date) || Infinity));
      }
      return arr;
    }

    function updateThanks(){
      const n = Object.keys(ratings).length;
      el("thanks").style.display = n ? "block" : "none";
      el("thanks").textContent = fmt(T.rated, n);
    }

    function showMessage(text){
      el("results").innerHTML = `<div class="message">${escapeHtml(text)}</div>`;
      el("bar").style.display = "none";
      el("more").style.display = "none";
      el("nothing").style.display = "none";
    }

    function render(){
      const root = el("results");
      root.innerHTML = "";
      const list = ordered().slice(0, visible);
      list.forEach(hit => root.appendChild(card(hit)));
      el("more").style.display = results.length > visible ? "block" : "none";
      el("nothing").style.display = results.length ? "block" : "none";
    }

    function card(hit){
      const div = document.createElement("div");
      div.className = "result" + (hit.id in ratings ? " rated-" + ratings[hit.id] : "");
      const type = T.types[hit.post_type] || "";
      const category = hit.primary_category || "";
      const authors = (hit.contributors || []).slice(0, 3).join(", ");
      div.innerHTML = `
        ${hit.image_url ? `<div class="thumb"><img src="${escapeHtml(hit.image_url)}" alt="" loading="lazy"></div>` : ``}
        <div class="body">
          <div class="meta-top">
            ${niceDate(hit.date) ? `<span class="date">${escapeHtml(niceDate(hit.date))}</span>` : ``}
            ${type ? `<span class="badge">${escapeHtml(type)}</span>` : ``}
            ${category ? `<span>${escapeHtml(category)}</span>` : ``}
            ${langSquares(hit.translations)}
          </div>
          <div class="title">${hit.url ? `<a href="${escapeHtml(hit.url)}" target="_blank" rel="noreferrer">${escapeHtml(hit.title)}</a>` : escapeHtml(hit.title)}</div>
          ${hit.summary ? `<p class="summary">${escapeHtml(hit.summary)}</p>` : ``}
          ${authors ? `<p class="authors">${escapeHtml(T.by)} ${escapeHtml(authors)}</p>` : ``}
          <div class="actions">
            <button class="vote yes" type="button"></button>
            <button class="vote no" type="button"></button>
            <button class="linkbtn" type="button"></button>
            <span class="status"></span>
          </div>
          <textarea rows="2"></textarea>
        </div>`;
      const yes = div.querySelector(".vote.yes"), no = div.querySelector(".vote.no");
      const commentBtn = div.querySelector(".linkbtn"), note = div.querySelector("textarea");
      const status = div.querySelector(".status");
      yes.textContent = T.useful; no.textContent = T.not_useful;
      commentBtn.textContent = T.comment; note.placeholder = T.comment_placeholder;
      if(hit.id in ratings){ (ratings[hit.id] ? yes : no).classList.add("on"); }
      commentBtn.onclick = () => { note.style.display = "block"; note.focus(); commentBtn.style.display = "none"; };
      yes.onclick = () => rate(hit, 1, {div, yes, no, note, status});
      no.onclick = () => rate(hit, 0, {div, yes, no, note, status});
      // A comment written after voting is saved with the same vote when the box is left
      note.onchange = () => { if(hit.id in ratings) rate(hit, ratings[hit.id], {div, yes, no, note, status}); };
      return div;
    }

    async function rate(hit, label, ui){
      if(!queryId) return;
      ui.status.className = "status";
      ui.status.textContent = T.saving;
      try{
        const res = await fetch("proxy_label", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query_id: queryId, article_id: hit.id, label: label, note: ui.note.value || null})
        });
        if(!res.ok) throw new Error("save failed");
        ratings[hit.id] = label;
        ui.yes.classList.toggle("on", label === 1);
        ui.no.classList.toggle("on", label === 0);
        ui.div.className = "result rated-" + label;
        ui.status.className = "status ok";
        ui.status.textContent = label === 1 ? T.saved_useful : T.saved_not_useful;
        updateThanks();
      }catch(e){
        ui.status.className = "status bad";
        ui.status.textContent = T.save_failed;
      }
    }

    function showMore(){ visible += 10; render(); }

    async function doSearch(){
      const q = el("q").value.trim();
      if(!q) return;
      showMessage(T.searching);
      try{
        const res = await fetch("proxy_search", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: q, per_page: 50, explain: false})
        });
        if(!res.ok) throw new Error("search failed");
        const data = await res.json();
        queryId = data.query_id;
        results = data.results || [];
      }catch(e){
        showMessage(T.error);
        return;
      }
      ratings = {};
      visible = 10;
      updateThanks();
      el("nothingStatus").textContent = "";
      if(!results.length){ showMessage(T.no_results); return; }
      el("bar").style.display = "flex";
      el("count").textContent = fmt(T.count, results.length);
      setSort("best");
    }

    async function markNone(){
      if(!queryId) return;
      const note = prompt(T.nothing_prompt) || null;
      const st = el("nothingStatus");
      try{
        const res = await fetch("proxy_label_query", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query_id: queryId, label: 0, note: note})
        });
        if(!res.ok) throw new Error("save failed");
        st.className = "status ok"; st.textContent = T.nothing_saved;
      }catch(e){
        st.className = "status bad"; st.textContent = T.save_failed;
      }
    }

    setup();
  </script>
</body>
</html>
"""
HTML = (HTML.replace("__LANG__", T["lang"]).replace("__PAGE_TITLE__", T["page_title"])
        .replace("__WORDING__", json.dumps(T, ensure_ascii=False)))


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
