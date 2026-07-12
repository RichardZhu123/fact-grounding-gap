#!/usr/bin/env python3
"""
Run multi-hop QA evaluation on MuSiQue dataset.


Saves detailed trajectories and computes metrics.
"""

import time
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict
from tqdm import tqdm


from simple_multihop_qa_contriever import SimpleMultiHopQA, Trajectory




def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s




def compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)




def compute_em(pred: str, gold: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(pred) == normalize_answer(gold))




def load_musique_data(filepath: str) -> List[Dict]:
    """Load MuSiQue JSONL data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data




def run_evaluation(
    input_file: str,
    output_file: str,
    corpus: str = "musique",
    model: str = "gpt-4.1-mini",
    max_hops: int = 4,
    limit: int = None,
    es_host: str = "localhost",
    es_port: int = 9200,
):
    """Run evaluation on MuSiQue dataset."""

    # Load data
    print(f"Loading data from {input_file}...")
    data = load_musique_data(input_file)

    if limit:
        data = data[:limit]

    print(f"Evaluating on {len(data)} examples...")

    # Initialize QA system
    qa = SimpleMultiHopQA(
        es_host=es_host,
        es_port=es_port,
        corpus_name=corpus,
        model=model,
        max_hops=max_hops,
    )

    # Run evaluation
    results = []
    total_f1 = 0.0
    total_em = 0.0

    for item in tqdm(data, desc="Running multi-hop QA"):
        question = item.get("question_text") or item.get("question")
        gold_answer = item.get("answer", "")
        qid = item.get("question_id", item.get("id", "unknown"))

        # Get trajectory with retry
        max_retries = 5
        for attempt in range(max_retries):
            try:
                trajectory = qa.answer(question, gold_answer=gold_answer)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"Error on {qid} (attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"FAILED after {max_retries} attempts on {qid}: {e}")
                    trajectory = Trajectory(
                        question=question,
                        steps=[],
                        final_answer="ERROR",
                        gold_answer=gold_answer,
                    )

        # Compute metrics
        f1 = compute_f1(trajectory.final_answer, gold_answer)
        em = compute_em(trajectory.final_answer, gold_answer)

        total_f1 += f1
        total_em += em

        # Store result
        result = {
            "qid": qid,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": trajectory.final_answer,
            "f1": f1,
            "em": em,
            "num_hops": len(trajectory.steps),
            "trajectory": asdict(trajectory),
        }
        results.append(result)

    # Compute aggregate metrics
    n = len(results)
    metrics = {
        "count": n,
        "avg_f1": total_f1 / n if n > 0 else 0,
        "avg_em": total_em / n if n > 0 else 0,
        "avg_hops": sum(r["num_hops"] for r in results) / n if n > 0 else 0,
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metrics": metrics,
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"\nMetrics:")
    print(f"  Count: {metrics['count']}")
    print(f"  Avg F1: {metrics['avg_f1']:.3f}")
    print(f"  Avg EM: {metrics['avg_em']:.3f}")
    print(f"  Avg Hops: {metrics['avg_hops']:.2f}")

    return metrics, results




def main():
    parser = argparse.ArgumentParser(description="Run Multi-Hop QA Evaluation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, default="results/multihop_results.json", help="Output JSON file")
    parser.add_argument("--corpus", type=str, default="musique", help="Elasticsearch index name")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--max-hops", type=int, default=4, help="Maximum retrieval hops")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--es-host", type=str, default="localhost", help="Elasticsearch host")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch port")
    args = parser.parse_args()

    run_evaluation(
        input_file=args.input,
        output_file=args.output,
        corpus=args.corpus,
        model=args.model,
        max_hops=args.max_hops,
        limit=args.limit,
        es_host=args.es_host,
        es_port=args.es_port,
    )




if __name__ == "__main__":
    main()

