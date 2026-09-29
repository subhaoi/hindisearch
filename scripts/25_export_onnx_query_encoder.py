from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from utils import e5_prefix_text

MODEL_NAME = "intfloat/multilingual-e5-large"

SAMPLE_QUERIES = [
    "महिला सशक्तिकरण",
    "बिहार स्वास्थ्य",
    "आशा कार्यकर्ताओं का प्रशिक्षण",
    "किसान आंदोलन",
    "rural health workers training",
    "csr education impact",
    "जलवायु परिवर्तन कृषि",
    "लड़कियों की पढ़ाई",
]


def encode(model, queries: List[str]) -> np.ndarray:
    return model.encode([e5_prefix_text(q, "query") for q in queries], normalize_embeddings=True)


def mean_latency_ms(model, queries: List[str], repeats: int = 3) -> float:
    encode(model, queries[:1])  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        for q in queries:
            encode(model, [q])
    return (time.perf_counter() - t0) * 1000 / (repeats * len(queries))


def main() -> None:
    """
    Export the e5-large query encoder to ONNX with int8 dynamic quantization for faster
    CPU inference on the API host. Needs `optimum[onnxruntime]`. Exporting takes ~6 GB RAM,
    so run it on a dev machine and copy --out to the server.

    Prints how closely the int8 query vectors agree with the original model; only switch
    the API over (EMBED_BACKEND=onnx) if the agreement is high and 23_evaluate_search.py
    shows no quality drop. Document vectors in Qdrant stay as they are.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/e5-large-onnx")
    ap.add_argument("--config", default="avx2", choices=["avx2", "avx512", "avx512_vnni", "arm64"],
                    help="Quantization target; avx2 runs on every x86 server (t3 included)")
    ap.add_argument("--queries", default="data/phase_4/core_queries.json", help="Extra queries for the check")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

    out = Path(args.out).resolve()
    print(f"Exporting {MODEL_NAME} to ONNX at {out} ...")
    onnx_model = SentenceTransformer(MODEL_NAME, backend="onnx")
    onnx_model.save(str(out))
    export_dynamic_quantized_onnx_model(onnx_model, args.config, str(out))

    candidates = sorted((out / "onnx").glob(f"*qint8*{args.config}*.onnx"))
    if not candidates:
        raise SystemExit(f"Quantized model not found under {out / 'onnx'}")
    file_name = str(candidates[0].relative_to(out))

    queries = list(SAMPLE_QUERIES)
    qpath = Path(args.queries)
    if qpath.exists():
        queries += json.loads(qpath.read_text(encoding="utf-8")).get("queries", [])

    ref = SentenceTransformer(MODEL_NAME)
    quant = SentenceTransformer(str(out), backend="onnx", model_kwargs={"file_name": file_name})
    cos = np.sum(encode(ref, queries) * encode(quant, queries), axis=1)
    print(f"\nQuery-vector agreement (cosine, int8 vs original) over {len(queries)} queries: "
          f"mean {cos.mean():.4f}, min {cos.min():.4f}")
    print(f"Latency per query: original {mean_latency_ms(ref, queries):.0f} ms | int8 {mean_latency_ms(quant, queries):.0f} ms")
    print("\nTo use it, set in .env on the API host:")
    print("  EMBED_BACKEND=onnx")
    print(f"  EMBED_ONNX_DIR={args.out}")
    print(f"  EMBED_ONNX_FILE={file_name}")


if __name__ == "__main__":
    main()
