from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from utils import Paths, ensure_dir, read_parquet, write_parquet, write_json, is_nullish, build_tokenizer_for_mpnet, count_tokens, safe_join


def split_into_paragraphs(text: str) -> List[str]:
    t = "" if is_nullish(text) else str(text)
    # Preserve paragraph boundaries (Phase 1 already normalized newlines)
    paras = [p.strip() for p in t.split("\n\n")]
    paras = [p for p in paras if p]
    return paras


def chunk_paragraphs(tokenizer, paragraphs: List[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Paragraph-aware chunking:
    - Builds chunks by appending paragraphs until max_tokens.
    - If a single paragraph is too long, it is split by sentence-ish separators.
    - Overlap is applied only to long splits.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
        current = []
        current_tokens = 0

    def split_long_text(t: str) -> List[str]:
        # Conservative split for Hindi: use danda and sentence punctuation, fallback to whitespace windows
        seps = ["।", "?", "!", "\n", ";", ":"]
        parts = [t]
        for sep in seps:
            new_parts = []
            for part in parts:
                if count_tokens(tokenizer, part) <= max_tokens:
                    new_parts.append(part)
                    continue
                # split on sep
                segs = [s.strip() for s in part.split(sep) if s.strip()]
                if len(segs) <= 1:
                    new_parts.append(part)
                else:
                    # put sep back lightly
                    rebuilt = [s + (sep if sep in ["।", "?", "!"] else "") for s in segs]
                    new_parts.extend(rebuilt)
            parts = new_parts

        # If still too big, do a token-window fallback
        final_parts: List[str] = []
        for part in parts:
            if count_tokens(tokenizer, part) <= max_tokens:
                final_parts.append(part.strip())
                continue
            # token window
            ids = tokenizer.encode(part, add_special_tokens=False)
            i = 0
            while i < len(ids):
                window = ids[i:i+max_tokens]
                txt = tokenizer.decode(window)
                final_parts.append(txt.strip())
                i += max_tokens - max(1, overlap_tokens)
        return [p for p in final_parts if p]

    for para in paragraphs:
        ptoks = count_tokens(tokenizer, para)

        if ptoks > max_tokens:
            # flush current then split this long paragraph
            flush()
            splits = split_long_text(para)
            for s in splits:
                stoks = count_tokens(tokenizer, s)
                if stoks <= max_tokens:
                    chunks.append(s.strip())
                else:
                    # extreme fallback: truncate by tokens
                    ids = tokenizer.encode(s, add_special_tokens=False)[:max_tokens]
                    chunks.append(tokenizer.decode(ids).strip())
            continue

        if current_tokens + ptoks <= max_tokens:
            current.append(para)
            current_tokens += ptoks
        else:
            flush()
            current.append(para)
            current_tokens = ptoks

    flush()
    return [c for c in chunks if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/final/articles_canonical.parquet", help="Canonical parquet path")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--max-tokens", type=int, default=240)
    ap.add_argument("--overlap-tokens", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    ensure_dir(paths.data / "phase_3")
    ensure_dir(paths.logs)

    df = read_parquet(Path(args.input).resolve())

    tokenizer = build_tokenizer_for_mpnet()

    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "articles": len(df),
        "chunks": 0,
        "max_tokens": args.max_tokens,
        "overlap_tokens": args.overlap_tokens,
        "empty_content": 0,
        "long_content_split": 0,
        "samples": [],
    }

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking articles"):
        article_id = str(r.get("id"))
        title = "" if is_nullish(r.get("title_hi")) else str(r.get("title_hi"))
        summary = "" if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi"))
        content = "" if is_nullish(r.get("content_hi")) else str(r.get("content_hi"))

        if not content.strip():
            report["empty_content"] += 1

        # Prefer content; but prepend title/summary to improve chunk context for retrieval/snippets
        base_text = safe_join([title, summary, content], sep="\n\n").strip()
        paras = split_into_paragraphs(base_text)

        # chunk
        chunks = chunk_paragraphs(tokenizer, paras, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
        if any(count_tokens(tokenizer, p) > args.max_tokens for p in paras):
            report["long_content_split"] += 1

        for idx, ch in enumerate(chunks):
            chunk_id = f"{article_id}::c{idx:04d}"
            rows.append({
                "chunk_id": chunk_id,
                "article_id": article_id,
                "chunk_index": idx,
                "chunk_text": ch,
                "chunk_tokens": count_tokens(tokenizer, ch),
                "url": None if is_nullish(r.get("url")) else str(r.get("url")),
                "published_date": None if is_nullish(r.get("published_date")) else str(r.get("published_date")),
                "published_ts": int(r.get("published_ts")) if "published_ts" in df.columns and not is_nullish(r.get("published_ts")) else 0,
                "title_hi": title,
            })

        report["chunks"] += len(chunks)
        if len(report["samples"]) < 10:
            report["samples"].append({"article_id": article_id, "n_chunks": len(chunks), "title": title[:80]})

    out_df = pd.DataFrame(rows)
    out_path = paths.data / "phase_3" / "chunks.parquet"
    write_parquet(out_df, out_path)

    write_json(paths.logs / "phase3_chunk_report.json", report)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {paths.logs / 'phase3_chunk_report.json'}")
    print(f"Total chunks: {len(out_df)}")


if __name__ == "__main__":
    main()
