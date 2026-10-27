# GPU session, 27 October 2026: Hindi model, English search, measurement

One g5.xlarge (NVIDIA A10G) in us-east-1, on-demand, about 12 hours (terminated the same
night). All model work was done offline on the GPU or the laptop; nothing calls a paid service
when readers search. The results below were deployed the same evening, step by step, each
with a rollback.

## What is live now (`main` 252789c on the search server)

| Change | Live since (27 Oct, IST) | Undo |
|---|---|---|
| Hindi speed fixes: cache, logging after the reply, per-step timings, 50 logged results per search | 22:20 | `git checkout pre-gpu-2026-10 && cp .env.pre-gpu-2026-10 .env`, restart `hindisearch-api` |
| English site: API at `https://api.idronline.org/en/`, test page at `/en-test/` (password) | 22:40 | `sudo systemctl disable --now englishsearch-api englishsearch-test-ui`; nginx backup `/etc/nginx/default.pre-en-2026-10-27` |
| Hindi on the trained model (`models/e5-base-hi-idr-v1`, int8 query encoder, Qdrant `idr_*_vec_v2`) | 22:57 | `bash deploy/hindi_model_switch.sh rollback` |
| Partial author names (both sites), English place aliases and synonyms | late evening | `git checkout pre-names-2026-10-27`, rerun `scripts/en/04_index_english.py --prune`, restart both APIs |
| Hindi indexes published posts only (new Status column in the Hindi exports) | takes effect at the weekly refresh, Sat 31 Oct 18:30 UTC | `git revert` the commit; the next refresh restores the items |

## Summary

| | Before | After |
|---|---|---|
| Hindi, live server, median time per search (same 50 searches) | 604 ms | **167 ms** |
| Hindi, live server, slowest 10% | 1.3 s | **0.8 s** |
| Hindi: turning a search into a vector, 2 cores | 363 ms | **11 ms** |
| Hindi search finds the question's article in its top 10 (1,500 test questions, whole search) | 66.7% | **70.9%** |
| Hindi: the article ranked first | 37.1% | 39.9% |
| Hindi real reader searches, first result relevant (graded, same code) | 60% | **64%** |
| Hindi hand-made test: authors in top 10 | 98% | **98%** (93% after the model switch, back to 98% with the partial-name fix) |
| **English** search (new), top 10, whole search | | **61.1%** (58% with the untrained model) |

Most of what Hindi readers will notice comes from the code fixes of the past weeks: on real
searches, the first result is relevant 60% of the time with today's code against 42% for what
readers actually got before. The trained model adds a further step and makes the search
several times faster.

Tested and not adopted (details below): a reranker (+4 points, too slow for the t3 server),
learned ranking weights, embedding Hinglish as typed, broad English synonyms (no gain).

## Open items

1. **Sunday 1 November:** check that the weekly refresh left out the 13 unpublished Hindi
   items (`grep -E "left out by status|Canonical articles|WARNING" logs/refresh/<latest>.log`).
2. **After a quiet week:** drop the old `idr_articles_vec_v1` / `idr_chunks_vec_v1` collections
   (~0.3-0.4 GB); rollback of the model switch is no longer possible after that.
3. **When English goes public:** the int8 English query encoder (plan below), the English
   weekly refresh (needs the English WP All Export links), and the admin panel's English tab
   (`ENGLISH_SEARCH_USAGE_URL=https://api.idronline.org/en/api/usage` on the tools server).
4. **Server:** stay on t3.large and watch the CPU credit balance; if it still throttles, or
   when English goes public, `m7g.large` (2 full cores, 8 GB, similar price) after a one-hour
   ARM test.
5. **Next GPU session** (optional): stricter grading (the grader sees no authors and is
   generous), more English training, and the reranker if the server gets full cores.

## How it was measured

- **Test questions.** A local model (Gemma 3 12B via Ollama) read every article and wrote
  realistic searches for it: 19,310 Hindi (991 articles) and 61,067 English (2,990 items).
  Styles: Hindi keywords, Hinglish, English words, full questions, names, misspellings; for
  English keywords, phrases, questions, names, Indian terms, misspellings.
- **Split once, fixed seed.** 15% of articles held out entirely (stand-in for next week's new
  articles); a quarter of each other article's questions held out; the rest trains. Test
  questions are never trained on.
- **Strict score**: does the question's own article come back in the top 10 (or first)?
  Understates quality, since many short questions have several right answers.
- **Graded score**: the same model judged each (search, article) pair 0/1/2, from title and
  summary. Spot check (24 pairs read by hand): reliable on "not relevant", generous on
  "relevant", especially for junk searches ("H by"). It does not see authors, so it marks an
  author's article "not relevant" for a search of the author's name. Use graded figures to
  compare versions, not as absolute quality.
- **Whole-search tests** ran the real search code (keywords, entities, vectors, ranking) on the
  laptop against local copies of Typesense/Qdrant built from the same data as production
  (`scripts/29_eval_search_on_questions.py`), and on the live server with logging off.

## Hindi

### The model

`intfloat/multilingual-e5-base`, fine-tuned on 12,305 training questions (in-batch and hard
negatives, 2 epochs). Saved as `models/e5-base-hi-idr-v1`.

Vector search alone (6,958 test questions):

| Question type | Live model (e5-large) | Small model, untrained | **Trained** |
|---|---|---|---|
| Full questions | 73% | 66% | 77% |
| Hindi keywords | 58% | 55% | 67% |
| English words | 39% | 20% | 63% |
| Names | 35% | 34% | 38% |
| Misspelt | 33% | 27% | 55% |
| Hinglish | 31% | 15% | 51% |
| **All** (known / held-out articles) | 43% / 46% | 34% / 34% | **57% / 60%** |

Held-out articles score as well as the rest: it learned the skill, not the articles.

### Whole search (1,500 test questions, same code)

| Question type | Live model | Trained (int8) |
|---|---|---|
| Hindi keywords | 73% | 78% |
| Misspelt | 69% | 76% |
| Full questions | 75% | 78% |
| Names | 77% | 75% |
| Hinglish | 61% | 66% |
| English words | 56% | 61% |
| **All, top 10** | **66.7%** | **70.9%** |
| All, first | 37.1% | 39.9% |
| Median time per search (laptop, 16 cores) | 431 ms | 253 ms |

Graded (same questions; any article the grader judged relevant counts; all top-10 results graded):

| | Live model | Trained |
|---|---|---|
| A relevant article in the top 10 | 92.0% | **94.7%** |
| First result relevant | 71.3% | **73.3%** |
| Relevant articles in the top 3 (average) | 1.58 | **1.77** |
| nDCG@10 | 0.861 | 0.869 |

The grader is generous, so the levels run high; the direction agrees with the strict test.
Name searches looked weak here (first result relevant ~40%), but that is the grader not
seeing authors: 40 of 41 author-name test questions return that author's article first.

### Real reader searches (846 logged searches, junk removed; graded)

The searches people actually typed (2025-12 to 2026-10), each run three ways and every top-10
result graded:

| | What readers got at the time | Today's code, live model | Today's code, trained model |
|---|---|---|---|
| A relevant article in the top 10 | 76% | 83% | **87%** |
| First result relevant | 42% | 60% | **64%** |
| Relevant articles in the top 3 (average) | 1.0 | 1.4 | **1.6** |

With the same code, the trained model's first result is better on 86 searches and worse on 52
(p ≈ 0.005). Of the 52, 29 have a relevant article second and 5 third; several are junk
searches the filter missed. Genuine misses include "dastwejo ke makadjaal me fsta" (a
misspelt Hinglish title) and "aser".

### Speed (query encoding, 2 cores)

| Version | Time | Top 10 (vectors alone) |
|---|---|---|
| Live e5-large (torch) | 363 ms | 44.1% |
| Trained, torch | 80 ms | 58.1% |
| Trained, ONNX fp32 | 23 ms | 58.1% |
| **Trained, ONNX int8 (reduce_range)** | **11 ms** | **57.8%** |
| Trained, ONNX int8 without reduce_range | 11 ms | 54.8% |

`reduce_range` matters on x86 CPUs without VNNI (the int8 maths saturates otherwise). It is
now the default in `scripts/25_export_onnx_query_encoder.py`.

On the live server (same 50 test searches, logging off), the whole search went from median
604 ms to 167 ms, slowest 10% 1.3 s to 0.8 s, average 807 ms to 361 ms. Hindi API memory
went from 2.3 to 1.7 GB, less than estimated: the libraries and article data dominate.

### Name searches

Full author names already worked. The gap was part of a longer name: "चन्द्र प्रताप" (चंद्र
प्रताप सिंह) and "दीप ज्योति" (दीप ज्योति सोनू ब्रह्मा) were not recognised, because only a
full name or a single word counted. Now any query made only of consecutive words of an
author's name boosts that author, in Devanagari, Hinglish and on the English site; topic
phrases ("महिला किसान") are not affected. Hand-made test, authors: top 10 93% → 98%, mrr
0.96 → 1.00; test questions unchanged (70.9%).

### Unpublished items

13 Hindi items had no real permalink (`?post_type=…&p=…`): 10 copies of published articles
(e.g. "बैगा समुदाय…" twice) and 3 empty. The Hindi exports had no Status column. It has been
added to the three export templates, and `scripts/00_concat_raw_exports.py` now keeps
`publish` only (without the column, nothing changes and the refresh warns).

### Tried, not adopted

| Experiment | Result | Verdict |
|---|---|---|
| Embed Hinglish as typed instead of converting to Devanagari | 66.9% (Hinglish 66% → 55%) | Keep conversion |
| Both, averaged | 70.9% | No gain |
| Learned ranking weights (coordinate search; pairwise logistic regression) | 71.0-71.5%; LR lands on the current weights | Keep current weights |
| Cross-encoder reranker (mMiniLM, 12 layers, trained), blended 50/50, top 20-30 | **74.1-75.1%**, first 42.7-43.7% | +4 points but ~510 ms per search on 2 cores even in int8 |
| Same, 6 layers | 72.4% | Most of the gain lost |

The reranker is saved (`models/hi-reranker-v1`, int8 at `models/hi-reranker-v1-onnx`); worth
re-timing on full cores (m7g) if the server changes.

## English

- Data: 2,990 items (1,237 articles, 1,065 posts, 551 Ground Up stories, 79 features, 58
  podcasts). The 315 regional-language translations (Marathi, Gujarati, Bengali, Kannada) are
  left out: English search is English only. Two Bengali items have broken permalinks
  (`%primary_category_article%`).
- Model comparison (untrained, vectors alone, first 5,154 questions): e5-base-v2 52%, 60 ms;
  bge-base 51%, 97 ms; bge-small 49%, 27 ms; nomic-embed 49%, 87 ms; multilingual-e5-large
  53%, 314 ms; arctic-embed did not load (needs xformers). Chosen: **e5-base-v2**.
- Fine-tuned on 37,835 training questions (`models/e5-base-v2-en-idr-v1`): vectors alone
  49.4% → 54.9%; whole search 57.8% → **61.1%** (first 27.3% → 29.0%), 122 ms per search on
  the laptop.
- Indian terms (anganwadi, gram panchayat, MGNREGA) score 20% strict but 79% first-result
  relevant graded: many items match them, so the "one right article" test understates them.
  Some generated "Indian term" questions are Hinglish nobody types on the English site
  ("khet ki pump"); to be dropped from English targets.

Graded (1,500 test questions, all top-10 results graded):

| | Base e5-base-v2 | Trained |
|---|---|---|
| A relevant item in the top 10 | 94.5% | **94.7%** |
| First result relevant | 71.8% | **73.5%** |
| Relevant items in the top 3 (average) | 1.85 | **1.89** |

### What the English search does (code: `scripts/_phase4/english_search_api.py`)

Same request/response contract as Hindi (`scripts/_phase4/api_common.py`), own Typesense and
Qdrant collections, own log database, `EN_*` settings. Authors named in full and non-national
places become filters; one word, or consecutive words, of an author's name boost their items;
place aliases (Orissa → Odisha, Kashmir → Jammu and Kashmir, from `config/location_aliases.json`)
count for queries and for places named in text; "podcast"/"ground up" filter by type; a bare
year lists that year; junk is rejected; results are cached; the log is written after the
reply; usage feed for the admin panel; test page (`FEEDBACK_SITE=en`).

Synonyms (`config/synonyms_en_v1.json`, loaded by `scripts/en/04_index_english.py`): 29
groups of acronyms, spellings and old city names (CSR, SHG, FCRA, MGNREGA spellings,
Bombay/Mumbai…). On the English test questions: trained 61.1%, + place aliases 60.8%, + tight
synonyms 60.7%, + all synonyms 60.3% (within noise). Broad groups (jobs/employment,
funding/grants, agriculture/farming…) widened every search that used those words and were
left out.

### Next for English (no GPU), when it goes public

1. Int8 query encoder: `scripts/25_export_onnx_query_encoder.py --model
   models/e5-base-v2-en-idr-v1`, an `EN_EMBED_BACKEND` switch in the English API (as Hindi),
   test with `29_eval_search_on_questions.py` on `split_en`, then copy ~110 MB to the server.
   Expected: ~80 → ~10 ms per query, 0.3-0.5 GB less memory.
2. Weekly refresh for English (needs the English export links).
3. A small hand-written English test set (known items, authors, series).
4. Ranking weights from the English grades (`scripts/37_learn_ranker_weights.py`); low
   expectation (no gain for Hindi), quick to check.

## Deployment record

All on 27 October, with the user's go-ahead for each batch.

**Step 1, code and speed fixes (22:20).** Tagged the running version `pre-gpu-2026-10`,
backed up `.env` to `.env.pre-gpu-2026-10`, pulled `main`, set `LOG_CANDIDATES_TOPN=50`,
restarted `hindisearch-api`.

**Step 3, English site (22:40).** Copied the English model and `data/en/` from the laptop
(`data/en/phase_3_ft/` vectors into `data/en/phase_3/`), created the `searchlogs_en` database,
added the `EN_*` settings, ran `scripts/en/04_index_english.py --prune`, installed
`englishsearch-api` and `englishsearch-test-ui`, added nginx locations `/en/` and `/en-test/`
(basic auth). Testers are added with:

```bash
sudo sh -c "printf 'idrtest:%s\n' \"\$(openssl passwd -apr1)\" >> /etc/nginx/.htpasswd-en-test"
```

**Step 2, trained Hindi model (22:57).** Copied `models/e5-base-hi-idr-v1/`,
`models/e5-base-hi-idr-v1-onnx/` (without `onnx/model.onnx`) and `data/phase_3_ft/` to the
server, then `bash deploy/hindi_model_switch.sh apply`: backs up `.env` and the vector files to
`data/backups/pre-model-switch`, builds `*_v2` beside the current collections, switches the
settings, restarts and smoke-tests (rolling back by itself on failure). Check with `status`,
undo with `rollback`. If a weekly refresh ran in between, run one refresh after a rollback so
the old collections get the newer articles.

**Names, English aliases and synonyms (late evening).** Tag `pre-names-2026-10-27`, pull,
`scripts/en/04_index_english.py --prune`, restart both APIs (run by the user).

**Published-only filter.** Pulled on the server; applies at the next weekly refresh.

## Files

| What | Where (laptop, `~/Downloads/IDR Search/hindi_search`) |
|---|---|
| Generated questions and splits | `data/synthetic/` |
| LLM grades | `data/judgments/hi.parquet`, `en.parquet` |
| Hindi trained model, int8, vectors | `models/e5-base-hi-idr-v1`, `models/e5-base-hi-idr-v1-onnx`, `data/phase_3_ft/` |
| English trained model, vectors | `models/e5-base-v2-en-idr-v1`, `data/en/phase_3_ft/` |
| Rerankers | `models/hi-reranker-v1`, `models/hi-reranker-v1-onnx` |
| All measurements | `runs/gpu/*.json`; live before/after: `runs/eval_before_model_switch.json`, `runs/eval_after_model_switch.json` (on the server) |
| GPU job logs and scripts | `logs/gpu-session/` |
| Local test copies (search engines, database, Python) | `~/idr-en-preview/` (stopped) |

Scripts: `scripts/gpu/` (setup, question split, vector scoring, fine-tuning, grading, reranker,
job chains), `scripts/29_eval_search_on_questions.py` (whole search on test questions),
`scripts/37_learn_ranker_weights.py`, `scripts/en/` (English pipeline),
`deploy/hindi_model_switch.sh` (Hindi model switch and rollback).
