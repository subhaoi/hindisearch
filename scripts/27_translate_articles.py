from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from utils import Paths, ensure_dir, read_parquet, write_parquet, write_json, is_nullish, clean_title

# IndicTrans2 Hindi->English, distilled 200M (MIT licence). Gated on Hugging Face: accept the
# terms once with a free account and put HF_TOKEN in .env. The repo ships its own model code
# (trust_remote_code), so it is pinned to a reviewed commit.
MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_REVISION = "eb9e49d81077cfc5311e82ff36d8c1fc11557b5d"
FIELDS = ["title_hi", "summary_hi"]
_SENT_SPLIT = re.compile(r"(?<=[।?!])\s+")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def segments(text: str) -> List[str]:
    """Sentences, so long summaries translate one sentence at a time (better quality)."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class IndicTrans2:
    def __init__(self, batch_size: int, num_beams: int) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=True).eval()
        self.ip = IndicProcessor(inference=True)
        self.batch_size = batch_size
        self.num_beams = num_beams

    def translate(self, texts: List[str]) -> List[str]:
        out: List[str] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Translating"):
            batch = self.ip.preprocess_batch(texts[i:i + self.batch_size], src_lang="hin_Deva", tgt_lang="eng_Latn")
            enc = self.tok(batch, truncation=True, padding="longest", max_length=256, return_tensors="pt")
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, num_beams=self.num_beams, max_length=256, use_cache=True)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang="eng_Latn"))
        return out


def main() -> None:
    """
    Translate article titles and summaries to English, once, offline, so English queries
    ("labour law", "women farmers") match English text at search time with no model in the
    request path. Incremental: only new or edited texts are translated (cached by text hash).

      python scripts/27_translate_articles.py
    Output: data/phase_3/translations.parquet (id, title_en, summary_en), read by 06_typesense_ingest.py.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--input", default="data/final/articles_canonical.parquet")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=3)
    args = ap.parse_args()

    load_dotenv()
    paths = Paths(root=Path(args.root).resolve())
    ensure_dir(paths.data / "phase_3")
    out_path = paths.data / "phase_3" / "translations.parquet"
    cache_path = paths.data / "phase_3" / "translation_cache.parquet"

    df = read_parquet(Path(args.input).resolve())
    cache: Dict[str, str] = {}
    if cache_path.exists():
        c = read_parquet(cache_path)
        cache = dict(zip(c["src_sha1"], c["text_en"]))

    # Segments per article field
    plan: List[Tuple[str, str, List[str]]] = []
    for _, r in df.iterrows():
        for field in FIELDS:
            v = r.get(field)
            text = "" if is_nullish(v) else (clean_title(v) if field == "title_hi" else str(v))
            plan.append((str(r.get("id")), field, segments(text)))

    all_segs = {s for _, _, segs in plan for s in segs}
    todo = sorted(s for s in all_segs if sha1(s) not in cache)
    print(f"Distinct segments: {len(all_segs)} | cached {len(all_segs) - len(todo)} | to translate {len(todo)}")

    if todo:
        translator = IndicTrans2(args.batch_size, args.num_beams)
        for src, en in zip(todo, translator.translate(todo)):
            cache[sha1(src)] = en.strip()
        write_parquet(pd.DataFrame({"src_sha1": list(cache), "text_en": list(cache.values())}), cache_path)

    rows: Dict[str, Dict[str, str]] = {}
    for aid, field, segs in plan:
        rows.setdefault(aid, {"id": aid})[field.replace("_hi", "_en")] = " ".join(cache.get(sha1(s), "") for s in segs).strip()
    out = pd.DataFrame(list(rows.values()))
    write_parquet(out, out_path)

    samples = out[out["title_en"] != ""].head(10)
    write_json(paths.logs / "translation_report.json", {
        "model": MODEL, "revision": MODEL_REVISION, "articles": int(len(out)), "translated_now": len(todo),
        "samples": [{"id": r.id, "title_hi": df.loc[df["id"].astype(str) == r.id, "title_hi"].iloc[0], "title_en": r.title_en}
                    for r in samples.itertuples()],
    })
    print(f"Wrote: {out_path} ({len(out)} articles)")


if __name__ == "__main__":
    main()
