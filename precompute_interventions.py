#!/usr/bin/env python3
"""
Precompute intervention lookup table for re-retrieval experiment.

For each (qid, MuSiQue step), computes:
  - oracle_flag: from existing LLM judge labels
  - deberta_flag: from running DeBERTa on (natural sub-question, baseline accumulated passages)
  - deberta_prob: DeBERTa probability of ANSWERABLE
  - reformulated_query: ONE reformulation if EITHER flag is True (shared across conditions)
  - reformulated_passages: top-3 ES results from reformulated query

Output: results/intervention_lookup.json
Format:
{
  "qid_xxx": {
    "1": {"oracle_flag": 0, "deberta_flag": 1, "deberta_prob": 0.23,
          "reformulated_query": "...", "reformulated_passages": [...]},
    "2": {...}
  }
}
"""

import json
import argparse
from tqdm import tqdm

from deberta_inference import DebertaPredictor
from query_reformulator import QueryReformulator
from es_retriever import ESRetriever


def format_passages_for_deberta(passages, max_chars=4000):
    parts, total = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"[{i}] {p.get('title','')}: {p.get('text','')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def get_accumulated_passages(trajectory):
    """Same dedup logic as fact_grounding_final.py — full accumulation across all hops."""
    seen, accumulated = set(), []
    for step in trajectory.get('steps', []):
        for p in step.get('retrieved_passages', []):
            t = p.get('text', '')
            if t and t not in seen:
                seen.add(t)
                accumulated.append(p)
    return accumulated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_results", default="results/full_run_all_4.1.json")
    parser.add_argument("--raw_data", default="raw_data/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--subquestions", default="results/musique_dev_subquestions.json")
    parser.add_argument("--fact_grounded", default="results/fact_grounded_final_dev.jsonl")
    parser.add_argument("--deberta_dir", default="models/deberta_fg_v2_best")
    parser.add_argument("--output", default="results/intervention_lookup.json")
    parser.add_argument("--deberta_threshold", type=float, default=0.5)
    parser.add_argument("--retrieval_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Process only N questions (for testing)")
    args = parser.parse_args()

    print("Loading components...")
    deberta = DebertaPredictor(args.deberta_dir)
    reformulator = QueryReformulator()
    retriever = ESRetriever()

    print("Loading data...")
    with open(args.qa_results) as f:
        qa = json.load(f)

    raw = {}
    with open(args.raw_data) as f:
        for line in f:
            d = json.loads(line)
            raw[d['id']] = d

    with open(args.subquestions) as f:
        subq_lookup = json.load(f)

    # Oracle labels: qid -> {sub_question_natural -> label_binary}
    oracle = {}
    with open(args.fact_grounded) as f:
        for line in f:
            e = json.loads(line)
            if e['label'] == 'UNKNOWN':
                continue
            oracle.setdefault(e['qid'], {})[e['sub_question']] = e['label_binary']

    results_list = qa['results']
    if args.limit:
        results_list = results_list[:args.limit]

    print(f"Building lookup for {len(results_list)} questions\n")

    lookup = {}
    stats = {
        'total_steps': 0,
        'oracle_flagged': 0,
        'deberta_flagged': 0,
        'both_flagged': 0,
        'either_flagged': 0,
        'reformulations': 0,
        'skipped_no_subq': 0,
        'skipped_no_raw': 0,
    }

    for r in tqdm(results_list, desc="Precompute"):
        qid = r.get('qid', '')
        question = r.get('question', '')
        if not qid:
            continue
        if qid not in subq_lookup:
            stats['skipped_no_subq'] += 1
            continue
        if qid not in raw:
            stats['skipped_no_raw'] += 1
            continue

        # Baseline accumulated passages (what DeBERTa and reformulator see)
        accumulated = get_accumulated_passages(r.get('trajectory', {}))
        passages_str = format_passages_for_deberta(accumulated)

        natural_subqs = subq_lookup[qid].get('natural_sub_questions', [])
        oracle_for_q = oracle.get(qid, {})

        per_step = {}
        for i, sub_q in enumerate(natural_subqs):
            step_num = i + 1  # 1-indexed; maps to system hop number
            stats['total_steps'] += 1

            # Oracle flag from existing LLM judge labels
            oracle_label = oracle_for_q.get(sub_q)
            if oracle_label is None:
                oracle_flag = 0  # missing label = treat as ANSWERABLE (conservative)
            else:
                oracle_flag = 1 if oracle_label == 0 else 0  # label_binary 0 = NOT-ANSWERABLE

            # DeBERTa flag
            try:
                _, prob = deberta.predict(sub_q, passages_str)
                deberta_flag = 1 if prob < args.deberta_threshold else 0
            except Exception as e:
                print(f"DeBERTa failed on {qid} step {step_num}: {e}")
                deberta_flag = 0
                prob = 1.0

            if oracle_flag:
                stats['oracle_flagged'] += 1
            if deberta_flag:
                stats['deberta_flagged'] += 1
            if oracle_flag and deberta_flag:
                stats['both_flagged'] += 1
            either_flagged = oracle_flag or deberta_flag
            if either_flagged:
                stats['either_flagged'] += 1

            # Reformulate ONCE if either flag is set, reuse for both conditions
            reformulated_query = None
            reformulated_passages = []
            if either_flagged:
                try:
                    reformulated_query = reformulator.reformulate(
                        question, sub_q, accumulated
                    )
                    reformulated_passages = retriever.retrieve(reformulated_query, k=args.retrieval_k)
                    stats['reformulations'] += 1
                except Exception as e:
                    print(f"Reformulation failed on {qid} step {step_num}: {e}")

            per_step[str(step_num)] = {
                'sub_question': sub_q,
                'oracle_flag': oracle_flag,
                'deberta_flag': deberta_flag,
                'deberta_prob': float(prob),
                'reformulated_query': reformulated_query,
                'reformulated_passages': reformulated_passages,
            }

        lookup[qid] = per_step

    # Save
    with open(args.output, 'w') as f:
        json.dump(lookup, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print(f"PRECOMPUTE COMPLETE")
    print(f"{'='*60}")
    print(f"Questions processed: {len(lookup)}")
    print(f"Skipped (no subq):   {stats['skipped_no_subq']}")
    print(f"Skipped (no raw):    {stats['skipped_no_raw']}")
    print(f"Total steps:         {stats['total_steps']}")
    if stats['total_steps'] > 0:
        print(f"  Oracle flagged:    {stats['oracle_flagged']} ({stats['oracle_flagged']/stats['total_steps']*100:.1f}%)")
        print(f"  DeBERTa flagged:   {stats['deberta_flagged']} ({stats['deberta_flagged']/stats['total_steps']*100:.1f}%)")
        print(f"  Both flagged:      {stats['both_flagged']} ({stats['both_flagged']/stats['total_steps']*100:.1f}%)")
        print(f"  Either flagged:    {stats['either_flagged']} ({stats['either_flagged']/stats['total_steps']*100:.1f}%)")
        print(f"  Reformulations:    {stats['reformulations']}")

        # Agreement
        if stats['oracle_flagged'] > 0 or stats['deberta_flagged'] > 0:
            agree_pos = stats['both_flagged']
            print(f"\nDeBERTa-Oracle agreement on flagged steps: {agree_pos}/{stats['either_flagged']} = {agree_pos/stats['either_flagged']*100:.1f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
