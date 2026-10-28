from __future__ import annotations

import argparse
import glob
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import strip_wp_html_to_text, is_nullish, parse_date_to_iso  # noqa: E402

# WP All Export files for the English site, one per post type (see README "English search").
# The file name says which type it is.
POST_TYPES = {"articles": "article", "features": "feature", "ground_up": "ground_up_story",
              "posts": "post", "podcasts": "podcast"}
SUMMARY_COLS = ["general_info_summary", "general_info_podcast_summary", "_yoast_wpseo_metadesc"]
# Translations into Indian languages live under a language path (idronline.org/mr-in/...,
# /gu-in/, /bn/, /kn-in/). English search covers English only: its models don't read those
# languages, so they would be keyword-only noise among English results.
_TRANSLATION_URL_RE = re.compile(r"idronline\.org/[a-z]{2}(-[a-z]{2})?/", re.IGNORECASE)
# Translations shown on English results, in this order (HI from the export's Hindi link; the rest
# matched to their English original by the last part of the web address, which they share)
TRANSLATION_ORDER = ["HI", "BN", "MR", "GU", "KN"]
# " | IDR", " | Ground Up | India Development Review" and similar SEO suffixes
_SEO_SUFFIX_RE = re.compile(r"(\s*\|\s*(IDR|India Development Review|Ground Up|Features?))+\s*$", re.IGNORECASE)


def text(v: Any) -> str:
    return "" if is_nullish(v) else html.unescape(str(v)).strip()


def clean_body(raw: Any) -> str:
    """HTML -> text. Entities in the WordPress source are often escaped twice (&amp;nbsp;)."""
    body, _ = strip_wp_html_to_text("" if is_nullish(raw) else str(raw))
    body = html.unescape(html.unescape(body)).replace("\xa0", " ")
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n[ \t]*(?=\n)", "\n", body)          # whitespace-only lines
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def split_pipe(v: Any) -> List[str]:
    return [x.strip() for x in text(v).split("|") if x.strip()]


def url_slug(url: str) -> str:
    return url.strip().rstrip("/").split("/")[-1].lower()


def translation_links(english: pd.DataFrame, translations: pd.DataFrame, codes: pd.Series,
                      hindi_published: Optional[Set[str]]) -> Tuple[List[List[Dict[str, str]]], Dict[str, Any]]:
    """Per English item, its translations as [{"lang": "HI", "url": ...}, ...] in TRANSLATION_ORDER.
    Hindi links are kept only when they point to a published Hindi article (when that list is known)."""
    by_slug: Dict[str, Dict[str, str]] = {}
    for url, code in zip(translations["url"], codes):
        by_slug.setdefault(url_slug(url), {}).setdefault(code[:2].upper(), url)
    norm = lambda u: u.strip().rstrip("/").lower()  # noqa: E731
    links, broken_hindi = [], 0
    for url, hindi in zip(english["url"], english["hindi_url"]):
        langs = dict(by_slug.get(url_slug(url), {}))
        if hindi:
            if hindi_published is None or norm(hindi) in hindi_published:
                langs["HI"] = hindi
            else:
                broken_hindi += 1
        links.append([{"lang": k, "url": langs[k]} for k in TRANSLATION_ORDER if k in langs])
    counts = {k: sum(any(x["lang"] == k for x in item) for item in links) for k in TRANSLATION_ORDER}
    return links, {"items_with_any": sum(bool(x) for x in links), "by_language": counts,
                   "hindi_links_not_published": broken_hindi,
                   "translations_without_english": sum(s not in {url_slug(u) for u in english["url"]} for s in by_slug)}


def post_type_of(path: str) -> str:
    name = Path(path).name.lower()
    for key, kind in POST_TYPES.items():
        if name.startswith(key):
            return kind
    raise SystemExit(f"Can't tell the post type of {path}; name it articles_*, features_*, ground_up_*, posts_* or podcasts_*")


def main() -> None:
    """
    English exports -> one clean table, the English counterpart of data/final/articles_canonical.parquet.

      python scripts/en/01_prepare_english.py --input ~/Downloads/fresh_english_data

    Keeps published items only (Status column). Writes data/en/final/articles_canonical.parquet
    and logs/en_prepare_report.json.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with the five English WP All Export CSVs")
    ap.add_argument("--out", default="data/en/final/articles_canonical.parquet")
    ap.add_argument("--report", default="logs/en_prepare_report.json")
    ap.add_argument("--hindi-articles", default="data/final/articles_canonical.parquet",
                    help="Published Hindi articles; Hindi links pointing elsewhere are left out")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"files": {}}
    for path in sorted(glob.glob(str(Path(args.input).expanduser() / "*.csv"))):
        kind = post_type_of(path)
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        id_col = "ID" if "ID" in df.columns else "id"
        status = df["Status"].fillna("").str.strip().str.lower() if "Status" in df.columns else pd.Series("publish", index=df.index)
        kept = df[status == "publish"]
        report["files"][Path(path).name] = {"rows": len(df), "published": len(kept),
                                            "dropped": status[status != "publish"].value_counts().to_dict()}
        for r in kept.to_dict("records"):
            content = clean_body(r.get("Content"))
            summary = next((text(r.get(c)) for c in SUMMARY_COLS if text(r.get(c))), "")
            seo = _SEO_SUFFIX_RE.sub("", text(r.get("_yoast_wpseo_title")))
            rows.append({
                "id": text(r.get(id_col)),
                "post_type": kind,
                "title": text(r.get("Title")),
                "seo_title": "" if seo == text(r.get("Title")) else seo,
                "summary": summary,
                "content": content,
                "url": text(r.get("Permalink")),
                "published_date": parse_date_to_iso(r.get("Date")),
                "categories": split_pipe(r.get("Categories")),
                "tags": split_pipe(r.get("Tags")),
                "locations": split_pipe(r.get("Locations")),
                "contributors": split_pipe(r.get("Contributor")),
                "podcast_series": split_pipe(r.get("Podcast Titles")),
                "image_url": text(r.get("Image Featured")),
                "hindi_url": text(r.get("hindi_info_article_link")),
            })

    out = pd.DataFrame(rows)
    lang = out["url"].str.extract(r"idronline\.org/([a-z]{2}(?:-[a-z]{2})?)/", flags=re.IGNORECASE)[0]
    translated = out["url"].str.contains(_TRANSLATION_URL_RE)
    report["translations_dropped"] = lang[translated].value_counts().to_dict()
    translations, codes = out[translated], lang[translated]
    out = out[~translated].reset_index(drop=True)
    hindi_path = Path(args.hindi_articles)
    hindi_published = ({u.strip().rstrip("/").lower() for u in pd.read_parquet(hindi_path, columns=["url"])["url"].dropna()}
                       if hindi_path.exists() else None)
    out["translations"], report["translation_links"] = translation_links(out, translations, codes, hindi_published)
    dup = out["id"].duplicated(keep="first")
    report["duplicate_ids_dropped"] = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    words = out["content"].str.split().str.len().fillna(0)
    report.update({
        "articles": len(out),
        "by_type": out["post_type"].value_counts().to_dict(),
        "no_text": int((words == 0).sum()),
        "no_summary": int((out["summary"] == "").sum()),
        "median_words": int(words.median()),
        "newest": out["published_date"].dropna().max(),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
