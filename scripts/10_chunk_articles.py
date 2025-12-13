from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from utils import (
    Paths,
    ensure_dir,
    read_parquet,
    write_parquet,
    write_json,
    is_nullish,
    build_tokenizer_for_mpnet,
    count_tokens,
    safe_join,
)


def split_into_paragraphs(text: str) -> List[str]:
    t = "" if is_nullish(text) else str(text)
    paras = [p.strip() for p in t.split("\n\n")]
    return [p for p in paras if p]


def split_long_text_sentenceish(text: str) -> List[str]:
    """
    Conservative Hindi-friendly splitting. This is not perfect; it exists only to
    reduce extreme paragraph sizes before token-window fallback.
    """
    t = text.strip()
    if not t:
        return []
    seps = ["।", "?", "!", "\n", ";", ":"]
    parts = [t]
    for sep in seps:
        new_parts: List[str] = []
        for part in parts:
            segs = [s.strip() for s in part.split(sep) if s.strip()]
            if len(segs) <= 1:
                new_parts.append(part)
            else:
                if sep in ["।", "?", "!"]:
                    new_parts.extend([s + sep for s in segs])
                else:
                    new_parts.extend(segs)
        parts = new_parts
    return [p.strip() for p in parts if p.strip()]


def token_window_split(tokenizer, text: str, hard_max_tokens: int, overlap_tokens: int) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= hard_max_tokens:
        return [text.strip()] if text.strip() else []

    step = hard_max_tokens - max(1, overlap_tokens)
    out: List[str] = []
    i = 0
    while i < len(ids):
        window = ids[i : i + hard_max_tokens]
        txt = tokenizer.decode(window).strip()
        if txt:
            out.append(txt)
        i += step
    return out


def chunk_paragraphs(
    tokenizer,
    paragraphs: List[str],
    max_tokens: int,
    overlap_tokens: int,
    hard_max_tokens: int,
) -> List[str]:
    """
    Paragraph-aware chunking with a HARD token cap.

    - Build chunks by appending paragraphs until `max_tokens`.
    - If a paragraph is too long, split sentence-ish first, then token windows.
    - After chunk assembly, run a post-pass HARD CAP that guarantees
      chunk_tokens <= hard_max_tokens.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if current:
            txt = "\n\n".join(current).strip()
            if txt:
                chunks.append(txt)
        current = []
        current_tokens = 0

    for para in paragraphs:
        ptoks = count_tokens(tokenizer, para)

        if ptoks > max_tokens:
            flush()
            # sentence-ish split first
            parts = split_long_text_sentenceish(para)
            if not parts:
                parts = [para]
            for part in parts:
                if count_tokens(tokenizer, part) <= max_tokens:
                    if part.strip():
                        chunks.append(part.strip())
                else:
                    # final fallback: token windows with HARD cap
                    chunks.extend(token_window_split(tokenizer, part, hard_max_tokens, overlap_tokens))
            continue

        if current_tokens + ptoks <= max_tokens:
            current.append(para)
            current_tokens += ptoks
        else:
            flush()
            current.append(para)
            current_tokens = ptoks

    flush()

    # HARD CAP post-pass: guarantees no chunk exceeds hard_max_tokens
    capped: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        ctoks = count_tokens(tokenizer, c)
        if ctoks <= hard_max_tokens:
            capped.append(c)
        else:
            capped.extend(token_window_split(tokenizer, c, hard_max_tokens, overlap_tokens))

    return [c for c in capped if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/final/articles_canonical.parquet", help="Canonical parquet path")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--max-tokens", type=int, default=240, help="Soft target tokens per chunk")
    ap.add_argument("--overlap-tokens", type=int, default=40, help="Overlap for token-window splits")
    ap.add_argument("--hard-max-tokens", type=int, default=480, help="Hard cap (must be <= 512 for MPNet)")
    args = ap.parse_args()

    if args.hard_max_tokens > 512:
        raise ValueError("hard-max-tokens must be <= 512 for paraphrase-multilingual-mpnet-base-v2")
    if args.max_tokens > args.hard_max_tokens:
        raise ValueError("max-tokens must be <= hard-max-tokens")

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
        "hard_max_tokens": args.hard_max_tokens,
        "empty_content": 0,
        "hard_capped_chunks": 0,
        "max_chunk_tokens_observed": 0,
        "samples": [],
    }

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking articles"):
        article_id = str(r.get("id"))
        title = "" if is_nullish(r.get("title_hi")) else str(r.get("title_hi"))
        summary = "" if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi"))
        content = "" if is_nullish(r.get("content_hi")) else str(r.get("content_hi"))

        if not content.strip():
            report["empty_content"] += 1

        base_text = safe_join([title, summary, content], sep="\n\n").strip()
        paras = split_into_paragraphs(base_text)

        chunks = chunk_paragraphs(
            tokenizer=tokenizer,
            paragraphs=paras,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            hard_max_tokens=args.hard_max_tokens,
        )

        for idx, ch in enumerate(chunks):
            tok_ct = count_tokens(tokenizer, ch)
            report["max_chunk_tokens_observed"] = max(report["max_chunk_tokens_observed"], tok_ct)
            if tok_ct > args.max_tokens:
                report["hard_capped_chunks"] += 1

            chunk_id = f"{article_id}::c{idx:04d}"
            rows.append({
                "chunk_id": chunk_id,
                "article_id": article_id,
                "chunk_index": idx,
                "chunk_text": ch,
                "chunk_tokens": tok_ct,
                "url": None if is_nullish(r.get("url")) else str(r.get("url")),
                "published_date": None if is_nullish(r.get("published_date")) else str(r.get("published_date")),
                "published_ts": int(r.get("published_ts")) if "published_ts" in df.columns and not is_nullish(r.get("published_ts")) else 0,
                "title_hi": title,
            })

        report["chunks"] += len(chunks)
        if len(report["samples"]) < 10:
            report["samples"].append({"article_id": article_id, "n_chunks": len(chunks), "title": title[:80]})

    out_df = pd.DataFrame(rows)

    # Hard assertion: never exceed hard max
    if len(out_df) > 0 and int(out_df["chunk_tokens"].max()) > args.hard_max_tokens:
        raise RuntimeError("Hard cap failed: some chunks exceed hard-max-tokens")

    out_path = paths.data / "phase_3" / "chunks.parquet"
    write_parquet(out_df, out_path)
    write_json(paths.logs / "phase3_chunk_report.json", report)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {paths.logs / 'phase3_chunk_report.json'}")
    print(f"Total chunks: {len(out_df)}")
    print(f"Max chunk tokens observed: {report['max_chunk_tokens_observed']}")


if __name__ == "__main__":
    main()
