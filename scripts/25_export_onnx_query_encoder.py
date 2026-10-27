from __future__ import annotations

import argparse
import json
import shutil
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
    Export the query encoder to ONNX with int8 dynamic quantization (weights) for faster CPU
    inference on the API host. Needs `optimum[onnxruntime]`. Memory peaks at about 5x the model
    file (measured: 5.3 GB for e5-base, ~10 GB for e5-large), so not on the API host; then copy
    --out to the server. Temporary files go to TMPDIR (set it to a real disk if /tmp is small).

    Prints how closely the int8 query vectors agree with the original model; only switch
    the API over (EMBED_BACKEND=onnx) if the agreement is high and 23_evaluate_search.py
    shows no quality drop. Document vectors in Qdrant stay as they are.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/e5-large-onnx")
    ap.add_argument("--model", default=MODEL_NAME, help="Model name or local folder (e.g. a fine-tuned model)")
    ap.add_argument("--config", default="avx2", choices=["avx2", "avx512", "avx512_vnni", "arm64"],
                    help="Quantization target; avx2 runs on every x86 server (t3 included)")
    ap.add_argument("--queries", default="data/phase_4/core_queries.json", help="Extra queries for the check")
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sentence_transformers import SentenceTransformer

    out = Path(args.out).resolve()
    print(f"Exporting {args.model} to ONNX at {out} ...")
    # Hub models publish onnx/model.onnx; a local (fine-tuned) folder has none and is converted
    local = Path(args.model)
    load_kwargs = {"export": True} if local.is_dir() and not (local / "onnx" / "model.onnx").exists() \
        else {"file_name": "onnx/model.onnx"}
    SentenceTransformer(args.model, backend="onnx", model_kwargs=load_kwargs).save(str(out))
    # Quantize the saved copy (the Hugging Face cache keeps model.onnx_data as a symlink, which
    # newer onnx versions refuse to load), with onnxruntime directly: optimum's quantizer needed
    # more memory than a 16 GB laptop has. The file name records the target CPU family.
    shutil.copy(out / "config.json", out / "onnx" / "config.json")
    quantize_dynamic(str(out / "onnx" / "model.onnx"), str(out / "onnx" / f"model_qint8_{args.config}.onnx"),
                     weight_type=QuantType.QInt8, per_channel=True)

    candidates = sorted((out / "onnx").glob(f"*qint8*{args.config}*.onnx"))
    if not candidates:
        raise SystemExit(f"Quantized model not found under {out / 'onnx'}")
    file_name = str(candidates[0].relative_to(out))

    queries = list(SAMPLE_QUERIES)
    qpath = Path(args.queries)
    if qpath.exists():
        queries += json.loads(qpath.read_text(encoding="utf-8")).get("queries", [])

    ref = SentenceTransformer(args.model)
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
