"""
Retrieval evaluation: measure Recall@k and MRR against a Q&A test set.

Usage:
    python eval/evaluate.py                        # uses eval/qa_pairs.json, k=5
    python eval/evaluate.py --qa eval/qa_pairs.json --k 10

qa_pairs.json format — each entry must have:
    "question"        : the question string
    "expected_source" : filename (e.g. "report.pdf") that must appear in top-k

    Optional:
    "expected_text"   : substring that must appear in the retrieved chunk text;
                        if provided, takes precedence over expected_source matching.

Metrics:
    Recall@k  — fraction of questions where a relevant chunk appears in top-k
    MRR       — mean reciprocal rank of the first relevant result
"""

import argparse
import json
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from rag_utils import COLLECTION_NAME, get_embedder, get_qdrant_client

DEFAULT_K = 5


def _is_relevant(hit, expected_source: str, expected_text: str) -> bool:
    payload = hit.payload or {}
    if expected_text:
        return expected_text.lower() in payload.get("text", "").lower()
    return payload.get("source", "") == expected_source


def evaluate(qa_path: str, k: int) -> None:
    with open(qa_path, encoding="utf-8") as f:
        pairs = [p for p in json.load(f) if "question" in p]

    if not pairs:
        print(f"No valid Q&A pairs found in {qa_path}")
        sys.exit(1)

    embedder = get_embedder()
    client = get_qdrant_client()

    reciprocal_ranks: list[float] = []
    hits: list[int] = []

    print(f"Evaluating {len(pairs)} questions at k={k}\n")

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        expected_source = pair.get("expected_source", "")
        expected_text = pair.get("expected_text", "")

        query_vector = embedder.encode(question).tolist()
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=k,
            ).points
        except (ValueError, RuntimeError):
            print("ERROR: collection not found — run ingest.py first.")
            sys.exit(1)

        first_relevant_rank = None
        for pos, hit in enumerate(results, 1):
            if _is_relevant(hit, expected_source, expected_text):
                first_relevant_rank = pos
                break

        rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        reciprocal_ranks.append(rr)
        hits.append(1 if first_relevant_rank else 0)

        status = f"rank {first_relevant_rank}" if first_relevant_rank else "MISS"
        print(f"  [{i:3d}] {status:<10}  {question}")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    recall = sum(hits) / len(hits)

    print(f"\n{'─' * 50}")
    print(f"  Questions : {len(pairs)}")
    print(f"  Recall@{k:<3}: {recall:.2%}  ({sum(hits)}/{len(pairs)})")
    print(f"  MRR       : {mrr:.4f}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--qa", default="eval/qa_pairs.json", help="Path to Q&A pairs JSON")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Top-k to retrieve (default: {DEFAULT_K})")
    args = parser.parse_args()
    evaluate(args.qa, args.k)