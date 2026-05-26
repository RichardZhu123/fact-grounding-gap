#!/usr/bin/env python3
"""
Chain-Aware Ablation: Add Previous Hop Answers to DeBERTa Input

Tests whether chain-level signal improves fact presence prediction.

Original input:
    Sub-question: Who is the spouse of Steve Hillage?
    Passages: [1] ... [2] ...

Chain-aware input:
    Previous answers: Steve Hillage
    Sub-question: Who is the spouse of Steve Hillage?
    Passages: [1] ... [2] ...

If accuracy jumps with this change, chain-level context matters.
If not, fact presence prediction is just hard from local features.

Usage:
    # On VM, regenerate CSVs with chain context
    python format_chain_aware.py \
        --train_jsonl results/fact_grounded_final_train.jsonl \
        --dev_jsonl results/fact_grounded_final_dev.jsonl \
        --train_subq results/musique_train_subquestions.json \
        --dev_subq results/musique_dev_subquestions.json \
        --train_out data/deberta_fg_chain_train.csv \
        --dev_out data/deberta_fg_chain_dev.csv
"""

import json
import csv
import argparse
from collections import defaultdict


def format_text_with_chain(sub_question, passages_combined, previous_answers, max_passage_chars=1500):
    """Format input text with chain context."""
    if previous_answers:
        prev_str = ", ".join(previous_answers)
        prev_part = f"Previous answers: {prev_str}\n\n"
    else:
        prev_part = ""

    passages_truncated = passages_combined[:max_passage_chars]
    return f"{prev_part}Sub-question: {sub_question}\n\nPassages:\n{passages_truncated}"


def process_split(jsonl_path, subq_path, output_path):
    """Process one split (train or dev)."""
    # Load sub-question lookup for previous answers
    with open(subq_path) as f:
        subq_lookup = json.load(f)

    # Load fact-grounded examples
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            e = json.loads(line)
            if e['label'] != 'UNKNOWN':
                examples.append(e)

    # Group by qid to get hop sequences
    by_qid = defaultdict(list)
    for e in examples:
        by_qid[e['qid']].append(e)

    # For each qid, sort by hop and add previous answers
    written = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])

        for qid, hops in by_qid.items():
            hops.sort(key=lambda h: h['hop_number'])

            # Get intermediate answers from subquestion lookup
            subq_data = subq_lookup.get(qid, {})
            intermediate_answers = subq_data.get('intermediate_answers', [])

            for i, hop in enumerate(hops):
                # Previous answers are intermediate answers from hops before this one
                prev_answers = intermediate_answers[:i]  # answers from hops 1..i-1

                text = format_text_with_chain(
                    sub_question=hop['sub_question'],
                    passages_combined=hop.get('passage_text_combined', ''),
                    previous_answers=prev_answers,
                )
                writer.writerow([text, hop['label_binary']])
                written += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--train_subq", required=True)
    parser.add_argument("--dev_subq", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--dev_out", required=True)
    args = parser.parse_args()

    print("Processing train set...")
    n_train = process_split(args.train_jsonl, args.train_subq, args.train_out)
    print(f"  Saved {n_train} examples to {args.train_out}")

    print("Processing dev set...")
    n_dev = process_split(args.dev_jsonl, args.dev_subq, args.dev_out)
    print(f"  Saved {n_dev} examples to {args.dev_out}")

    # Show samples
    print("\nSample inputs (first 3 from train):")
    with open(args.train_out) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"\n--- Example {i+1} (label={row['label']}) ---")
            text = row['text']
            print(text[:500] + ("..." if len(text) > 500 else ""))


if __name__ == "__main__":
    main()
