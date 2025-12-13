from __future__ import annotations

import argparse
from pathlib import Path

from utils import Paths, write_json, canonicalize_query_for_search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Raw user query")
    ap.add_argument("--root", default=".", help="Project root")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    out = canonicalize_query_for_search(args.q)

    print("raw:", out["raw"])
    print("mode:", out["mode"])
    print("query_used:", out["q"])
    print("roman_norm:", out.get("roman_norm", ""))

    log_path = paths.logs / "roman_canonicalize_report.json"
    write_json(log_path, {"input": args.q, "output": out})
    print(f"Wrote: {log_path}")


if __name__ == "__main__":
    main()
