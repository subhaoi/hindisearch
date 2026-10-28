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
# The page itself is the search results design proposed for the website, plus 👍/👎 per result.
WORDING = {
    "hi": {
        "lang": "hi",
        "locale": "hi-IN",
        "page_title": "आईडीआर हिंदी सर्च: टेस्टिंग",
        "banner_title": "आप आईडीआर हिंदी की नई सर्च परख रहे हैं।",
        "banner_hint": "हर नतीजे पर 👍 या 👎 दबाकर बताएं कि वह काम का है या नहीं। क्लिक करते ही सेव हो जाता है।",
        "brand_sub": "सर्च प्रीव्यू",
        "placeholder": "जैसे: महिला किसान, mahila yojana bihar",
        "search": "खोजें",
        "examples_label": "ये आज़माएं:",
        "examples": ["महिला किसान", "शिक्षा नीति", "mahila yojana bihar", "जलवायु परिवर्तन"],
        "searching": "खोज रहे हैं…",
        "no_results": "कोई नतीजा नहीं मिला। दूसरे शब्दों में खोजकर देखें।",
        "error": "कुछ गड़बड़ हो गई। एक बार फिर कोशिश करें।",
        "results_for": "{q} के नतीजे",
        "sort_best": "सबसे सटीक",
        "sort_newest": "सबसे नया",
        "sort_oldest": "सबसे पुराना",
        "all": "सभी",
        "useful": "👍 काम का है",
        "not_useful": "👎 काम का नहीं",
        "comment": "टिप्पणी",
        "comment_placeholder": "क्यों? जैसे: पुराना है, विषय अलग है…",
        "saving": "सेव हो रहा है…",
        "saved": "✓ सेव हो गया",
        "save_failed": "सेव नहीं हुआ। फिर से दबाएं।",
        "show_more": "और नतीजे दिखाएं",
        "nothing_question": "इनमें से कुछ भी काम का नहीं?",
        "nothing_button": "बताएं",
        "nothing_prompt": "आप क्या ढूंढ रहे थे? (ज़रूरी नहीं)",
        "nothing_saved": "✓ सेव हो गया। धन्यवाद!",
        "by": "लेखक:",
        "also_in": "इन भाषाओं में भी",
        "author_line": "लेखक · आईडीआर पर {n} लेख",
        "author_link": "प्रोफ़ाइल देखें →",
        "types": {}, "types_plural": {},
    },
    "en": {
        "lang": "en",
        "locale": "en-GB",
        "page_title": "IDR search: testing",
        "banner_title": "You're testing IDR's new search.",
        "banner_hint": "Rate each result with 👍 or 👎. That's all; it saves as you click.",
        "brand_sub": "Search preview",
        "placeholder": "For example: water crisis in Rajasthan",
        "search": "Search",
        "examples_label": "Try:",
        "examples": ["menstrual health", "CSR funding", "podcast on philanthropy", "Rachita Vora"],
        "searching": "Searching…",
        "no_results": "No results. Try searching with other words.",
        "error": "Something went wrong. Please try again.",
        "results_for": "Results for {q}",
        "sort_best": "Best match",
        "sort_newest": "Newest",
        "sort_oldest": "Oldest",
        "all": "All",
        "useful": "👍 Useful",
        "not_useful": "👎 Not useful",
        "comment": "Comment",
        "comment_placeholder": "Why? For example: too old, different topic…",
        "saving": "Saving…",
        "saved": "✓ Saved",
        "save_failed": "Not saved. Please press again.",
        "show_more": "Load more results",
        "nothing_question": "Nothing useful in these results?",
        "nothing_button": "Tell us",
        "nothing_prompt": "What were you looking for? (optional)",
        "nothing_saved": "✓ Saved. Thank you!",
        "by": "By",
        "also_in": "Also in",
        "author_line": "Author · {n} pieces on IDR",
        "author_link": "See profile on IDR →",
        "types": {"article": "Article", "feature": "Feature", "ground_up_story": "Ground Up story",
                  "post": "Post", "podcast": "Podcast"},
        "types_plural": {"article": "Articles", "ground_up_story": "Ground Up stories", "feature": "Features",
                         "post": "Posts", "podcast": "Podcasts"},
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
      --red:#d93b3b; --teal:#3f9e96; --text:#1d1d1d; --muted:#6b6b6b; --line:#e7e7e7;
      --green:#0a7a3b; --green-soft:#e8f5ee; --no:#b42318; --no-soft:#fdecea;
    }
    *{ box-sizing:border-box; }
    body{ margin:0; background:#fff; color:var(--text); font-size:16px; line-height:1.55;
      font-family: "Open Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans", "Noto Sans Devanagari", Arial; }
    a{ color:inherit; text-decoration:none; }
    button{ font: inherit; }

    .banner{ background:#1d1d1d; color:#fff; font-size:14px; }
    .banner .in{ max-width:920px; margin:0 auto; padding: 9px 24px; display:flex; flex-wrap:wrap; gap:4px 10px; }
    .banner .hint{ color:#cfcfcf; }
    .wrap{ max-width: 920px; margin: 0 auto; padding: 0 24px 60px; }
    .head{ display:flex; align-items:baseline; gap:14px; padding: 18px 0 14px; border-bottom: 2px solid #111; }
    .logo{ font-weight:800; font-size:44px; line-height:1; letter-spacing:-2px; position:relative; }
    .logo:before{ content:""; position:absolute; left:2px; top:1px; width:8px; height:8px; background:var(--red); }
    .head .sub{ font-size: 14px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color: var(--muted); }

    .searchbox{ margin: 26px 0 8px; display:flex; gap:10px; }
    .field{ flex:1; min-width:0; display:flex; align-items:center; gap:10px; border:2px solid #cfcfcf; border-radius: 10px; padding: 0 12px 0 14px; height: 58px; background:#fff; }
    .field:focus-within{ border-color: var(--red); }
    .mag{ width:18px; height:18px; border:2.5px solid #999; border-radius:50%; position:relative; flex:0 0 18px; }
    .mag:after{ content:""; position:absolute; width:7px; height:2.5px; background:#999; right:-6px; bottom:-2px; transform:rotate(45deg); }
    .field input{ flex:1; min-width:0; border:0; outline:none; font-size:20px; background:transparent; font-family: inherit; }
    .clear{ border:0; background:none; color:#999; font-size:24px; line-height:1; cursor:pointer; padding: 4px 6px; display:none; }
    .go{ border:0; border-radius:10px; background:var(--red); color:#fff; padding: 0 26px; font-weight:700; font-size:17px; cursor:pointer; }
    .tries{ font-size:14px; color: var(--muted); display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom: 18px; }
    .tries button{ border:1px solid var(--line); border-radius:999px; background:#fff; padding: 3px 11px; cursor:pointer; color: var(--text); font-size:14px; }
    .tries button:hover{ border-color: var(--red); }

    .resline{ display:none; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin: 6px 0 12px; }
    .for{ font-size: 17px; } .for b{ font-weight:700; }
    .seg{ display:flex; border:1px solid var(--line); border-radius: 10px; overflow:hidden; }
    .seg button{ border:0; background:#fff; padding: 8px 14px; color: var(--muted); font-size:14px; cursor:pointer; }
    .seg button.on{ background:#1d1d1d; color:#fff; }
    .chips{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom: 8px; position:relative; }
    .chips button{ border:1px solid #d6d6d6; border-radius: 999px; padding: 6px 14px; font-size:14px; background:#fff; cursor:pointer; white-space:nowrap; }
    .chips button.on{ background:#1d1d1d; border-color:#1d1d1d; color:#fff; }

    .person{ display:flex; align-items:center; gap:16px; border:1px solid var(--line); border-radius:14px; padding: 14px 18px; margin: 10px 0 8px; }
    .avatar{ width:58px; height:58px; border-radius:50%; background: var(--teal); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:21px; flex:0 0 58px; }
    .person .nm{ font-size: 19px; font-weight:800; } .person .ds{ color: var(--muted); font-size:14px; }
    .person .link{ margin-left:auto; border:2px solid #1d1d1d; border-radius:10px; padding: 7px 14px; font-weight:700; white-space:nowrap; font-size:15px; }

    .message{ border-top:1px solid var(--line); padding: 28px 0; text-align:center; color: var(--muted); }

    .card{ display:flex; gap:20px; padding: 18px 0; border-top: 1px solid var(--line); }
    .card.r1{ background: linear-gradient(90deg, var(--green-soft), #fff 40%); }
    .card.r0{ background: linear-gradient(90deg, var(--no-soft), #fff 40%); }
    .thumb{ width:160px; height:106px; flex:0 0 160px; border-radius:8px; overflow:hidden; background:#eee; }
    .thumb img{ width:100%; height:100%; object-fit:cover; display:block; }
    .body{ flex:1; min-width:0; }
    .meta{ display:flex; align-items:center; gap:6px 12px; flex-wrap:wrap; font-size:13.5px; color: var(--muted); margin-bottom: 3px; }
    .meta .date{ color: var(--text); font-weight:600; }
    .badge{ background:#f0f0f0; color:#333; border-radius:6px; padding: 1px 8px; font-size: 12.5px; }
    .langs{ display:inline-flex; align-items:center; gap:5px; margin-left:auto; }
    .langs .also{ font-size:12.5px; color: var(--muted); margin-right:2px; }
    .lang{ display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border-radius:6px; color:#fff; font-weight:800; font-size:10.5px; }
    .lang:hover{ filter: brightness(1.15); }
    .title{ font-size: 19.5px; line-height:1.3; font-weight:700; margin: 0 0 5px; }
    .title a:hover{ text-decoration: underline; }
    .sum{ font-size: 15px; color:#333; margin: 0 0 6px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
    mark{ background: #fff3c4; color: inherit; font-weight:700; padding: 0 1px; }
    .ex{ font-size: 14px; color: var(--muted); margin: 0 0 7px; padding-left: 10px; border-left: 3px solid #e2e2e2; }
    .foot{ display:flex; align-items:center; gap:8px 10px; flex-wrap:wrap; }
    .by{ font-size: 14px; color: var(--muted); flex:1; min-width: 180px; } .by .who{ color: var(--teal); font-weight:700; }
    .rate{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .v{ border:1.5px solid #d6d6d6; border-radius: 999px; padding: 4px 12px; font-size:14px; font-weight:600; background:#fff; cursor:pointer; white-space:nowrap; }
    .v.yes.on, .v.yes:hover{ border-color: var(--green); background: var(--green-soft); color: var(--green); }
    .v.no.on, .v.no:hover{ border-color: var(--no); background: var(--no-soft); color: var(--no); }
    .cm{ border:0; background:none; font-size: 13px; color: var(--muted); text-decoration: underline; cursor:pointer; padding: 2px; }
    .status{ font-size: 13px; font-weight:600; } .status.ok{ color: var(--green); } .status.bad{ color: var(--no); }
    textarea{ width:100%; border:1px solid #cfcfcf; border-radius: 8px; padding:10px; font-size:15px; margin-top: 10px; display:none; font-family: inherit; }

    .end{ border-top: 1px solid var(--line); padding: 22px 0 10px; display:none; flex-direction:column; align-items:center; gap:14px; }
    .more{ border:2px solid #1d1d1d; border-radius:10px; padding: 10px 22px; font-weight:700; background:#fff; cursor:pointer; }
    .none{ font-size: 15px; color: var(--muted); text-align:center; }
    .none button{ border:1.5px solid #bdbdbd; border-radius:10px; padding: 7px 16px; color: var(--text); font-weight:600; margin-left: 8px; background:#fff; cursor:pointer; }
    .none .status{ display:block; margin-top: 8px; }

    @media (max-width: 640px){
      .wrap, .banner .in{ padding-left:16px; padding-right:16px; }
      .banner .hint{ display:none; }
      .logo{ font-size: 36px; } .head .sub{ font-size: 12px; }
      .searchbox{ margin: 16px 0 8px; } .field{ height: 50px; } .field input{ font-size: 17px; }
      .go{ padding: 0 14px; font-size: 15px; }
      .seg button{ padding: 6px 10px; font-size: 13px; }
      .chips{ flex-wrap:nowrap; overflow-x:auto; scrollbar-width:none; }
      .chips::-webkit-scrollbar{ display:none; }
      .chipwrap{ position:relative; }
      .chipwrap:after{ content:""; position:absolute; right:0; top:0; bottom:8px; width:40px; background: linear-gradient(90deg, rgba(255,255,255,0), #fff); pointer-events:none; }
      .card{ display:block; padding: 14px 0; }
      .card:after{ content:""; display:block; clear:both; }
      .thumb{ float:right; width:84px; height:84px; flex:none; margin: 2px 0 8px 12px; }
      .title{ font-size: 17px; }
      .langs{ margin-left:0; width:100%; margin-top:4px; }
      .sum{ font-size: 14.5px; }
      .foot{ clear:both; }
      .person .link{ margin-left: 0; }
      .person{ flex-wrap: wrap; }
      .none button{ display:inline-block; margin: 8px 0 0; }
    }
  </style>
</head>

<body>
  <div class="banner"><div class="in"><b id="bannerTitle"></b><span class="hint" id="bannerHint"></span></div></div>
  <div class="wrap">
    <div class="head"><span class="logo">idr</span><span class="sub" id="brandSub"></span></div>

    <form class="searchbox" onsubmit="doSearch(); return false;">
      <div class="field">
        <span class="mag"></span>
        <input id="q" autocomplete="off" />
        <button type="button" class="clear" id="clear" aria-label="Clear">×</button>
      </div>
      <button type="submit" class="go" id="searchBtn"></button>
    </form>
    <div class="tries" id="tries"></div>

    <div class="resline" id="resline">
      <span class="for" id="resFor"></span>
      <span class="seg">
        <button id="sortBest" class="on" onclick="setSort('best')"></button>
        <button id="sortNew" onclick="setSort('newest')"></button>
        <button id="sortOld" onclick="setSort('oldest')"></button>
      </span>
    </div>
    <div class="chipwrap"><div class="chips" id="chips"></div></div>
    <div id="person"></div>
    <div id="results"></div>

    <div class="end" id="end">
      <button class="more" id="more" onclick="showMore()"></button>
      <div class="none"><span id="nothingQ"></span><button onclick="markNone()" id="nothingBtn"></button>
        <span class="status" id="nothingStatus"></span></div>
    </div>
  </div>

  <script>
    const T = __WORDING__;
    // The same piece in other languages (English site): colour and name in its own script
    const LANGS = {HI: ["हिंदी", "#B45309"], BN: ["বাংলা", "#15803D"], MR: ["मराठी", "#7E22CE"],
                   GU: ["ગુજરાતી", "#1D4ED8"], KN: ["ಕನ್ನಡ", "#BE185D"]};
    const TYPE_ORDER = ["article", "ground_up_story", "feature", "post", "podcast"];
    // Newest / Oldest reorder only the strongest matches; weak matches never jump to the top by date
    const STRONG = 30;
    const STOP = new Set(["the","a","an","and","or","of","in","on","for","to","with","by","is","are","what","how","why",
                          "के","की","का","में","और","से","को","पर","है"]);

    let queryId = null, results = [], author = null;
    let sortMode = "best", typeFilter = null, visible = 10;
    let ratings = {};   // article id -> 1 / 0, for this search
    let wordRe = null;

    function el(id){ return document.getElementById(id); }
    function fmt(s, k, v){ return s.replace("{" + k + "}", v); }
    function escapeHtml(s){
      return String(s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
        .replaceAll('"',"&quot;").replaceAll("'","&#039;");
    }
    function dateValue(s){ const t = Date.parse(s || ""); return Number.isFinite(t) && t > 0 ? t : null; }
    function niceDate(s){
      const t = dateValue(s);
      return t === null ? "" : new Date(t).toLocaleDateString(T.locale, {day:"numeric", month:"long", year:"numeric"});
    }

    // Search words to highlight: plain words of the query (plurals and -es forms match too)
    function buildWordRe(q){
      const words = (q.toLowerCase().match(/[\\p{L}\\p{M}\\p{N}]+/gu) || [])
        .filter(w => !STOP.has(w) && (w.length >= 3 || /[\\u0900-\\u097F]/.test(w)));
      if(!words.length) return null;
      const esc = words.map(w => w.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"));
      return new RegExp("(?<![\\\\p{L}\\\\p{M}\\\\p{N}])((?:" + esc.join("|") + ")(?:s|es)?)(?![\\\\p{L}\\\\p{M}\\\\p{N}])", "giu");
    }
    function highlight(text){
      const safe = escapeHtml(text);
      return wordRe ? safe.replace(wordRe, "<mark>$1</mark>") : safe;
    }
    function hasWord(text){
      if(!wordRe) return false;
      wordRe.lastIndex = 0;
      const found = wordRe.test(text || "");
      wordRe.lastIndex = 0;
      return found;
    }
    // A sentence from the matching passage, only when the card's visible text (title and about two
    // lines of summary) doesn't show the search words: then it explains why the result is here.
    // Not for an author search: the match is the byline ("Rachita" in the text is someone else).
    function excerpt(hit){
      if(!wordRe || author || hasWord((hit.title || "") + " " + (hit.summary || "").slice(0, 160))) return "";
      const snip = (hit.snippet || "").replace(hit.title || "", "");
      for(const s of snip.split(/(?<=[.?!।])\\s+/)){
        if(hasWord(s)){
          const t = s.trim();
          return "…" + (t.length < 190 ? t : t.slice(0, 187).replace(/\\s+\\S*$/, "") + "…");
        }
      }
      return "";
    }

    function setup(){
      document.title = T.page_title;
      el("bannerTitle").textContent = T.banner_title;
      el("bannerHint").textContent = T.banner_hint;
      el("brandSub").textContent = T.brand_sub;
      el("q").placeholder = T.placeholder;
      el("searchBtn").textContent = T.search;
      el("tries").innerHTML = `<span>${escapeHtml(T.examples_label)}</span>` +
        T.examples.map(x => `<button type="button">${escapeHtml(x)}</button>`).join("");
      el("tries").querySelectorAll("button").forEach(b => b.onclick = () => { el("q").value = b.textContent; doSearch(); });
      el("sortBest").textContent = T.sort_best;
      el("sortNew").textContent = T.sort_newest;
      el("sortOld").textContent = T.sort_oldest;
      el("more").textContent = T.show_more;
      el("nothingQ").textContent = T.nothing_question;
      el("nothingBtn").textContent = T.nothing_button;
      el("q").oninput = () => { el("clear").style.display = el("q").value ? "block" : "none"; };
      el("clear").onclick = () => { el("q").value = ""; el("clear").style.display = "none"; el("q").focus(); };
      el("q").focus();
      // A shared link like ...?q=CSR+funding runs that search straight away
      const q = new URLSearchParams(location.search).get("q");
      if(q){ el("q").value = q; el("clear").style.display = "block"; doSearch(); }
    }

    function setSort(mode){
      sortMode = mode;
      el("sortBest").classList.toggle("on", mode === "best");
      el("sortNew").classList.toggle("on", mode === "newest");
      el("sortOld").classList.toggle("on", mode === "oldest");
      visible = 10;
      render();
    }

    function setType(t){
      typeFilter = t;
      visible = 10;
      renderChips();
      render();
    }

    function ordered(){
      const base = typeFilter ? results.filter(h => h.post_type === typeFilter) : results.slice();
      if(sortMode === "best") return base;
      const strong = base.slice(0, STRONG), rest = base.slice(STRONG);
      const sign = sortMode === "newest" ? -1 : 1;
      // Undated results go last either way
      strong.sort((a, b) => {
        const ta = dateValue(a.date), tb = dateValue(b.date);
        if(ta === null || tb === null) return (ta === null) - (tb === null);
        return sign * (ta - tb);
      });
      return strong.concat(rest);
    }

    function renderChips(){
      const present = TYPE_ORDER.filter(t => results.some(h => h.post_type === t) && T.types_plural[t]);
      if(present.length < 2){ el("chips").innerHTML = ""; return; }
      el("chips").innerHTML = [[null, T.all]].concat(present.map(t => [t, T.types_plural[t]]))
        .map(([t, label]) => `<button type="button" class="${typeFilter === t ? "on" : ""}" data-t="${t || ""}">${escapeHtml(label)}</button>`).join("");
      el("chips").querySelectorAll("button").forEach(b => b.onclick = () => setType(b.dataset.t || null));
    }

    function renderPerson(){
      if(!author){ el("person").innerHTML = ""; return; }
      const initials = author.name.split(/\\s+/).filter(Boolean).map(w => w[0]).slice(0, 2).join("").toUpperCase();
      el("person").innerHTML = `<div class="person"><div class="avatar">${escapeHtml(initials)}</div>
        <div><div class="nm">${escapeHtml(author.name)}</div><div class="ds">${escapeHtml(fmt(T.author_line, "n", author.count))}</div></div>
        <a class="link" href="${escapeHtml(author.url)}" target="_blank" rel="noreferrer">${escapeHtml(T.author_link)}</a></div>`;
    }

    function langSquares(list){
      const items = (list || []).filter(x => LANGS[x.lang]);
      if(!items.length) return "";
      return `<span class="langs"><span class="also">${escapeHtml(T.also_in)}</span>` + items.map(x =>
        `<a class="lang" style="background:${LANGS[x.lang][1]}" href="${escapeHtml(x.url)}" target="_blank" rel="noreferrer" title="${LANGS[x.lang][0]}">${x.lang}</a>`
      ).join("") + `</span>`;
    }

    function card(hit){
      const div = document.createElement("div");
      div.className = "card" + (hit.id in ratings ? " r" + ratings[hit.id] : "");
      const type = T.types[hit.post_type] || "";
      const authors = (hit.contributors || []).slice(0, 3).map(a => `<span class="who">${escapeHtml(a)}</span>`).join(", ");
      const ex = excerpt(hit);
      div.innerHTML = `
        ${hit.image_url ? `<div class="thumb"><img src="${escapeHtml(hit.image_url)}" alt="" loading="lazy"></div>` : ``}
        <div class="body">
          <div class="meta">
            ${niceDate(hit.date) ? `<span class="date">${escapeHtml(niceDate(hit.date))}</span>` : ``}
            ${type ? `<span class="badge">${escapeHtml(type)}</span>` : ``}
            ${hit.primary_category ? `<span>${escapeHtml(hit.primary_category)}</span>` : ``}
            ${langSquares(hit.translations)}
          </div>
          <div class="title">${hit.url ? `<a href="${escapeHtml(hit.url)}" target="_blank" rel="noreferrer">${escapeHtml(hit.title)}</a>` : escapeHtml(hit.title)}</div>
          ${hit.summary ? `<p class="sum">${highlight(hit.summary)}</p>` : ``}
          ${ex ? `<p class="ex">${highlight(ex)}</p>` : ``}
          <div class="foot">
            <span class="by">${authors ? `${escapeHtml(T.by)} ${authors}` : ``}</span>
            <span class="rate">
              <button type="button" class="v yes"></button>
              <button type="button" class="v no"></button>
              <button type="button" class="cm"></button>
              <span class="status"></span>
            </span>
          </div>
          <textarea rows="2"></textarea>
        </div>`;
      const yes = div.querySelector(".v.yes"), no = div.querySelector(".v.no");
      const cm = div.querySelector(".cm"), note = div.querySelector("textarea"), status = div.querySelector(".status");
      yes.textContent = T.useful; no.textContent = T.not_useful;
      cm.textContent = T.comment; note.placeholder = T.comment_placeholder;
      if(hit.id in ratings){ (ratings[hit.id] ? yes : no).classList.add("on"); status.className = "status ok"; status.textContent = T.saved; }
      cm.onclick = () => { note.style.display = "block"; note.focus(); cm.style.display = "none"; };
      const ui = {div, yes, no, note, status};
      yes.onclick = () => rate(hit, 1, ui);
      no.onclick = () => rate(hit, 0, ui);
      // A comment written after rating is saved with the same rating when the box is left
      note.onchange = () => { if(hit.id in ratings) rate(hit, ratings[hit.id], ui); };
      return div;
    }

    function render(){
      const list = ordered();
      el("results").innerHTML = "";
      list.slice(0, visible).forEach(h => el("results").appendChild(card(h)));
      el("more").style.display = list.length > visible ? "inline-block" : "none";
    }

    function showMessage(text){
      el("results").innerHTML = `<div class="message">${escapeHtml(text)}</div>`;
      el("resline").style.display = "none";
      el("chips").innerHTML = "";
      el("person").innerHTML = "";
      el("end").style.display = "none";
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
        ui.div.className = "card r" + label;
        ui.status.className = "status ok";
        ui.status.textContent = T.saved;
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
        author = data.author || null;
      }catch(e){
        showMessage(T.error);
        return;
      }
      wordRe = buildWordRe(q);
      ratings = {};
      typeFilter = null;
      el("nothingStatus").textContent = "";
      if(!results.length){ showMessage(T.no_results); return; }
      el("resline").style.display = "flex";
      el("resFor").innerHTML = fmt(escapeHtml(T.results_for), "q", `<b>${escapeHtml(q)}</b>`);
      el("end").style.display = "flex";
      renderChips();
      renderPerson();
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
