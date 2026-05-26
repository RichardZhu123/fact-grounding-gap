#!/usr/bin/env python3
"""
Test: Does DeBERTa accuracy hold on system SEARCH queries (vs natural sub-questions)?

For each system hop in baseline trajectories:
  - Get the system's SEARCH query (or initial question for hop 1)
  - Get accumulated passages up to that hop
  - Run DeBERTa
  - Compare against existing LLM judge label for matching MuSiQue step (step N -> hop N)
"""

import json
import argparse
import random
from deberta_inference import DebertaPredictor


def format_passages(passages, max_chars=4000):
    parts, total = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"[{i}] {p.get('title','')}: {p.get('text','')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def get_passages_up_to_hop(trajectory, hop_num):
    """Accumulate passages from hops 1..hop_num (inclusive), deduped by text."""
    seen, accumulated = set(), []
    for step in trajectory.get('steps', []):
        if step.get('hop_number', 0) > hop_num:
            break
        for p in step.get('retrieved_passages', []):
            t = p.get('text', '')
            if t and t not in seen:
                seen.add(t)
                accumulated.append(p)
    return accumulated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_results", default="results/full_run_all_4.1.json")
    parser.add_argument("--fact_grounded", default="results/fact_grounded_final_dev.jsonl")
    parser.add_argument("--subquestions", default="results/musique_dev_subquestions.json")
    parser.add_argument("--deberta_dir", default="models/deberta_fg_v2_best")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading components...")
    deberta = DebertaPredictor(args.deberta_dir)

    with open(args.qa_results) as f:
        qa = json.load(f)

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

    # Build (qid, hop_num, system_query, oracle_label) tuples
    candidates = []
    for r in qa['results']:
        qid = r.get('qid', '')
        if qid not in subq_lookup or qid not in oracle:
            continue
        natural_subqs = subq_lookup[qid].get('natural_sub_questions', [])
        steps = r.get('trajectory', {}).get('steps', [])
        # Map step N -> hop N
        for i, sub_q in enumerate(natural_subqs):
            step_num = i + 1
            if step_num > len(steps):
                continue  # System didn't reach this hop
            if sub_q not in oracle[qid]:
                continue
            label = oracle[qid][sub_q]
            system_query = steps[step_num - 1].get('query', '')
            candidates.append({
                'qid': qid,
                'hop_num': step_num,
                'system_query': system_query,
                'natural_sub_question': sub_q,
                'oracle_label': label,  # 1 = ANSWERABLE, 0 = NOT-ANSWERABLE
                'trajectory': r['trajectory'],
            })

    print(f"Total candidate (hop, label) pairs: {len(candidates)}")
    random.seed(args.seed)
    random.shuffle(candidates)
    sample = candidates[:args.n_samples]

    print(f"Testing DeBERTa on {len(sample)} system queries...\n")

    # Also test on natural sub-questions (sanity check — should match dev 79.4%)
    natural_correct = 0
    system_correct = 0
    natural_pos = 0
    system_pos = 0
    matches_between = 0

    for i, c in enumerate(sample):
        passages = get_passages_up_to_hop(c['trajectory'], c['hop_num'])
        passages_str = format_passages(passages)

        # DeBERTa on natural sub-question (the trained distribution)
        _, prob_nat = deberta.predict(c['natural_sub_question'], passages_str)
        pred_nat = 1 if prob_nat >= 0.5 else 0

        # DeBERTa on system query (the runtime distribution)
        _, prob_sys = deberta.predict(c['system_query'], passages_str)
        pred_sys = 1 if prob_sys >= 0.5 else 0

        if pred_nat == c['oracle_label']:
            natural_correct += 1
        if pred_sys == c['oracle_label']:
            system_correct += 1
        if pred_nat == 1: natural_pos += 1
        if pred_sys == 1: system_pos += 1
        if pred_nat == pred_sys:
            matches_between += 1

        if i < 5:
            print(f"[{i+1}] hop {c['hop_num']}")
            print(f"    natural subq: {c['natural_sub_question'][:80]}")
            print(f"    system query: {c['system_query'][:80]}")
            print(f"    oracle label: {'ANSWERABLE' if c['oracle_label']==1 else 'NOT-ANSWERABLE'}")
            print(f"    natural pred: {'ANS' if pred_nat==1 else 'NO '} ({prob_nat:.2f})")
            print(f"    system pred:  {'ANS' if pred_sys==1 else 'NO '} ({prob_sys:.2f})")
            print()

    n = len(sample)
    print(f"\n{'='*60}")
    print(f"DEBERTA DISTRIBUTION SHIFT TEST (n={n})")
    print(f"{'='*60}")
    print(f"DeBERTa on natural sub-questions: {natural_correct}/{n} = {natural_correct/n*100:.1f}% (expected ~79%)")
    print(f"DeBERTa on system queries:        {system_correct}/{n} = {system_correct/n*100:.1f}%")
    print(f"Agreement between the two:        {matches_between}/{n} = {matches_between/n*100:.1f}%")
    print(f"\nDecision:")
    sys_acc = system_correct / n
    if sys_acc >= 0.70:
        print(f"  ✓ DeBERTa stays as primary ({sys_acc*100:.1f}% >= 70%)")
    elif sys_acc <= 0.60:
        print(f"  ✗ Use LLM judge as primary ({sys_acc*100:.1f}% <= 60%)")
    else:
        print(f"  ? Borderline ({sys_acc*100:.1f}%) — discuss with professor")


if __name__ == "__main__":
    main()
