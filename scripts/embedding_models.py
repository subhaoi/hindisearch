"""Query/passage prefixes per embedding-model family (each family was trained with its own)."""
from __future__ import annotations

from typing import Dict, Tuple

# (query prefix, passage prefix, needs trust_remote_code)
PREFIXES: Dict[str, Tuple[str, str, bool]] = {
    "e5": ("query: ", "passage: ", False),
    "bge-": ("Represent this sentence for searching relevant passages: ", "", False),
    "arctic-embed": ("query: ", "", True),
    "nomic-embed": ("search_query: ", "search_document: ", True),
    "gte-": ("", "", True),
}


def prefixes(model_name: str) -> Tuple[str, str, bool]:
    """(query prefix, passage prefix, trust_remote_code) for a model name or local path."""
    name = model_name.lower()
    for key, val in PREFIXES.items():
        if key in name:
            return val
    return "", "", False
