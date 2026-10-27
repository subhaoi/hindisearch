# GPU session, 27 October 2026: Hindi model, English search, measurement

One g5.xlarge (NVIDIA A10G) in us-east-1, on-demand, about 12 hours. Everything here was
prepared offline; nothing calls a paid service when readers search, and nothing has been
deployed. Code is on the `multisite` branch; models and data are on the laptop (not in git).

## Summary

| | Before | After (ready, not live) |
|---|---|---|
| **Hindi** search finds the question's article in its top 10 (1,500 test questions, whole search) | 66.7% | **70.9%** |
| Hindi: the article ranked first | 37.1% | 39.9% |
| Hindi: turning a search into a vector, 2 cores | 363 ms | **11 ms** |
| Hindi query model size | 2.2 GB | 0.28 GB |
| **English** search (new), top 10, whole search | | **61.1%** (58% with the untrained model) |

Real reader searches (graded, same code): first result relevant 60% with the live model, 64%
with the trained one; 42% for what readers actually got before this month's fixes.

**Deployed 27 October, 22:20-22:40:** step 1 (code, speed fixes) and step 3 (English site:
API at `https://api.idronline.org/en/`, password-protected test page at `/en-test/`). Hindi still
switched to the trained model at 22:57 (step 2, `deploy/hindi_model_switch.sh apply`).

Measured on the live server, same 50 test searches (log off), before and after the switch:
median 604 → **167 ms**, slowest 10% 1.3 → 0.8 s, average 807 → 361 ms. Hand-made test set
(`runs/eval_before_model_switch.json`, `..._after_...`): known items better (three move up),
authors 98% → 93% (one name, "चन्द्र प्रताप", already weak, now missing), series unchanged.
Hindi API memory 2.3 → 1.7 GB (less saving than estimated: libraries and data dominate). The
`*_v1` collections stay for rollback; drop them after a quiet week (~0.3-0.4 GB).

Tested and not adopted (details below): a reranker (+4 points, too slow for the t3 server),
learned ranking weights and embedding Hinglish as typed (no gain).

## Decisions for you

Done on 27 October: `multisite` merged into `main`; step 1 (speed fixes) and step 3 (English
site) deployed. Still open:

1. **Tester login** for `/en-test/` (one command, under "Deploy", step 3), then share it.
2. **Hindi trained model: switched 27 October 22:57.** Undo any time with
   `bash deploy/hindi_model_switch.sh rollback`; drop the `*_v1` collections after a quiet week.
3. **English WP All Export links** for the English weekly refresh, and the admin panel's
   English tab (`ENGLISH_SEARCH_USAGE_URL=https://api.idronline.org/en/api/usage` on the tools server).
4. **Server**: stay on t3.large after the model switch and watch the CPU credit balance; if it
   still throttles, or when English goes public, `m7g.large` (2 full cores, 8 GB, similar price).
5. **Next GPU session** (later, optional): stricter grading, more training for English, and the
   reranker if the server gets full cores.

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
- **Graded score**: the same model judged each (search, article) pair 0/1/2. Spot check (24
  pairs read by hand): reliable on "not relevant", generous on "relevant", especially for junk
  searches ("H by"). Use graded figures to compare versions, not as absolute quality.
- **Whole-search tests** ran the real search code (keywords, entities, vectors, ranking) on the
  laptop against local copies of Typesense/Qdrant built from the same data as production.

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
Weak spot for both: searches that are a name (person, organisation): first result relevant
only ~40% of the time.

### Real reader searches (846 logged searches, junk removed; graded)

The searches people actually typed (2025-12 to 2026-10), each run three ways and every top-10
result graded:

| | What readers got at the time | Today's code, live model | Today's code, trained model |
|---|---|---|---|
| A relevant article in the top 10 | 76% | 83% | **87%** |
| First result relevant | 42% | 60% | **64%** |
| Relevant articles in the top 3 (average) | 1.0 | 1.4 | **1.6** |

Most of the gain comes from the code fixes of the past weeks (live since 27 October, 22:20);
the trained model adds a further step on the same code (first result better on 86 searches,
worse on 52), and is ~30x faster at its step. Grades are generous: compare columns, not levels.

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
- Weakest type: Indian terms (anganwadi, gram panchayat, MGNREGA) at 20%, partly because many
  items match them. Some generated "Indian term" questions are Hinglish nobody types on the
  English site ("khet ki pump"); to be dropped from English targets.

Graded (1,500 test questions, all top-10 results graded):

| | Base e5-base-v2 | Trained |
|---|---|---|
| A relevant item in the top 10 | 94.5% | **94.7%** |
| First result relevant | 71.8% | **73.5%** |
| Relevant items in the top 3 (average) | 1.85 | **1.89** |

Indian-term searches: first result relevant 79% (graded) against 20% strict, so the strict
figure was mostly the "one right article" effect. Names are the weak spot (first result
relevant ~50%), as in Hindi.

### What the English search does (code: `scripts/_phase4/english_search_api.py`)

Same request/response contract as Hindi (`scripts/_phase4/api_common.py`), own Typesense and
Qdrant collections, own log database, `EN_*` settings. Authors named in full and non-national
places become filters; one word of an author's name boosts their items; "podcast"/"ground up"
filter by type; a bare year lists that year; junk is rejected; results are cached; the log is
written after the reply; usage feed for the admin panel; test page (`FEEDBACK_SITE=en`).

### Next for English (no GPU), in order

1. **Name searches** (authors, organisations): the first result is relevant only ~50% of the
   time (Hindi ~40%). Find why they miss (spelling variants, organisation names that are not
   in the author list, partial names) and fix for both sites.
2. ONNX int8 query encoder for the English API (needs an `EN_EMBED_BACKEND` switch, as Hindi).
3. Synonyms from the English members of `config/synonyms_v1.json` (NREGA/MGNREGA, CSR, ASHA,
   SHG, FCRA…) and place aliases from `config/location_aliases.json` (Orissa → Odisha,
   Bombay → Mumbai). Cheap; modest gain, since Indian-term searches already do well graded.
4. A small hand-written English test set (known items, authors, series).
5. Ranking weights from the English grades (`scripts/37_learn_ranker_weights.py`); low
   expectation (no gain for Hindi), quick to check.
6. Weekly refresh for English (needs the English export links), and author aliases.

## Deploy

Every step can be undone. Two steps, a day or two apart, so each shows its own effect in the
per-step timings.

### Step 1: code (speed fixes, model setting, English service), after `multisite` is merged

```bash
cd ~/hindisearch
git tag pre-gpu-2026-10            # the version running now, to come back to
cp .env .env.pre-gpu-2026-10
git pull
sed -i 's/^LOG_CANDIDATES_TOPN=200/LOG_CANDIDATES_TOPN=50/' .env
sudo systemctl restart hindisearch-api
```

Behaviour changes only by the speed fixes: the model settings default to today's model.
**Undo:** `git checkout pre-gpu-2026-10 && cp .env.pre-gpu-2026-10 .env && sudo systemctl restart hindisearch-api`
(then `git checkout main` again when ready).

### Step 2: the trained Hindi model (`deploy/hindi_model_switch.sh`)

Copy from the laptop to `~/hindisearch` on the server (about 1.4 GB):

```bash
# on the laptop, from the repo folder
rsync -a models/e5-base-hi-idr-v1 ubuntu@13.127.105.2:hindisearch/models/
rsync -a --exclude onnx/model.onnx models/e5-base-hi-idr-v1-onnx ubuntu@13.127.105.2:hindisearch/models/
rsync -a data/phase_3_ft ubuntu@13.127.105.2:hindisearch/data/
```

Then on the server:

```bash
bash deploy/hindi_model_switch.sh status     # what is live now
bash deploy/hindi_model_switch.sh apply      # backup, build *_v2 beside the current collections,
                                             # switch settings, restart, smoke-test
```

`apply` keeps the current model, collections and vector files, and rolls back by itself if
the test searches fail. The API also refuses to start if model and collections don't match.
**Undo at any time:** `bash deploy/hindi_model_switch.sh rollback`. If a weekly refresh ran
in between, run one refresh afterwards so the old collections get the newer articles.
Not run end to end yet (needs the server's service); each step in it was run locally tonight.

### Step 3: English test site (with either step; separate from Hindi)

`README.md`, "English site search": a log database, `EN_*` settings
(`EN_EMBED_MODEL=models/e5-base-v2-en-idr-v1`, collections `idr_en_*_v1`), the model and
`data/en/` copied from the laptop (`data/en/phase_3_ft/` vectors into `data/en/phase_3/`), then
`scripts/en/04_index_english.py --prune`, two services, and an nginx location with a password.
**Undo:** `sudo systemctl disable --now englishsearch-api englishsearch-test-ui` and remove
the nginx location. Hindi is not touched.

Done on 27 October: API at `https://api.idronline.org/en/` (e.g. `POST /en/search`), test page
at `https://api.idronline.org/en-test/`, nginx backup at `/etc/nginx/default.pre-en-2026-10-27`.
The test page refuses everyone until a login is added (no reload needed; repeat per tester):

```bash
sudo sh -c "printf 'idrtest:%s\n' \"\$(openssl passwd -apr1)\" >> /etc/nginx/.htpasswd-en-test"
```

## Files

| What | Where (laptop, `~/Downloads/IDR Search/hindi_search`) |
|---|---|
| Generated questions and splits | `data/synthetic/` |
| LLM grades | `data/judgments/hi.parquet`, `en.parquet` |
| Hindi trained model, int8, vectors | `models/e5-base-hi-idr-v1`, `models/e5-base-hi-idr-v1-onnx`, `data/phase_3_ft/` |
| English trained model, vectors | `models/e5-base-v2-en-idr-v1`, `data/en/phase_3_ft/` |
| Rerankers | `models/hi-reranker-v1`, `models/hi-reranker-v1-onnx`, `models/hi-reranker-l6-v1` (GPU only) |
| All measurements | `runs/gpu/*.json` |
| Local preview (search engines, database, Python) | `~/idr-en-preview/` |

Scripts: `scripts/gpu/` (setup, question split, vector scoring, fine-tuning, grading, reranker,
job chains), `scripts/29_eval_search_on_questions.py` (whole search on test questions),
`scripts/37_learn_ranker_weights.py`, `scripts/en/` (English pipeline).
