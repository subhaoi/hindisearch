from __future__ import annotations

import argparse
from pathlib import Path

from scripts.utils import Paths, ensure_dir, write_json

DEFAULT_QUERIES = [
    "महिला सशक्तिकरण",
    "बिहार स्वास्थ्य",
    "आशा कार्यकर्ताओं का प्रशिक्षण बिहार",
    "शिक्षा कार्यक्रम",
    "किशोरियों के लिए योजना",
    "rural health workers training",
    "csr education impact",
    "mahila yojana bihar",
    "जलवायु परिवर्तन कृषि",
    "स्थानीय शासन पंचायत",
    "आदिवासी समुदाय अधिकार",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    ensure_dir(paths.data / "phase_4")

    out = {"version": "core_query_set_v1", "queries": DEFAULT_QUERIES}
    out_path = paths.data / "phase_4" / "core_queries.json"
    write_json(out_path, out)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
