#!/usr/bin/env python3
"""
Format Sufficiency Data for DeBERTa Classification Fine-Tuning

Converts the extracted JSONL sufficiency data into CSV format
suitable for DeBERTa sequence classification.

Input format (JSONL from extract_training_data.py):
    {"sub_question": "...", "passages": [...], "label_binary": 0/1, ...}

Output format (CSV):
    text,label
    "Sub-question: ... [SEP] Passages: ...", 0

The text field concatenates the sub-question and passages into a single
string. DeBERTa's tokenizer will handle the [SEP] token insertion
between segments during tokenization.

Usage:
    python format_deberta_data.py \
        --input results/sufficiency_data_train.jsonl \
        --output data/deberta_train.csv

    python format_deberta_data.py \
        --input results/sufficiency_data_dev.jsonl \
        --output data/deberta_dev.csv
"""

import json
import csv
import argparse
from pathlib import Path
from collections import Counter


def format_example(example: dict, max_passage_chars: int = 1500) -> str:
    """
    Format a single example into a text string for DeBERTa.

    Structure:
        Sub-question: [sub_question]
        Passages:
        [1] title: text
        [2] title: text
        ...

    We limit total passage characters to ~1500 to stay within
    DeBERTa's 512 token limit after tokenization (~4 chars per token,
    512 tokens ≈ 2048 chars, minus ~200 chars for sub-question and
    special tokens).
    """
    sub_question = example.get('sub_question', '').strip()
    passages = example.get('passages', [])

    # Format passages with truncation
    passage_parts = []
    total_chars = 0
    for i, p in enumerate(passages, 1):
        title = p.get('title', '').strip()
        text = p.get('text', '').strip()
        passage_str = f"[{i}] {title}: {text}"

        # Truncate if we're running out of space
        remaining = max_passage_chars - total_chars
        if remaining <= 0:
            break
        if len(passage_str) > remaining:
            passage_str = passage_str[:remaining] + "..."
        passage_parts.append(passage_str)
        total_chars += len(passage_str)

    passages_text = "\n".join(passage_parts) if passage_parts else "No passages retrieved."

    # Combine into single text
    # Using a clear structure that DeBERTa can learn from
    text = f"Sub-question: {sub_question}\n\nPassages:\n{passages_text}"

    return text


def main():
    parser = argparse.ArgumentParser(description="Format sufficiency data for DeBERTa")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--max_passage_chars", type=int, default=1500,
                        help="Max chars for passages (default 1500)")
    args = parser.parse_args()

    # Load JSONL
    examples = []
    with open(args.input) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {args.input}")

    # Format and write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])

        for ex in examples:
            sub_q = ex.get('sub_question', '').strip()
            if not sub_q:
                skipped += 1
                continue

            text = format_example(ex, max_passage_chars=args.max_passage_chars)
            label = ex['label_binary']
            writer.writerow([text, label])
            written += 1

    # Statistics
    labels = Counter(ex['label_binary'] for ex in examples if ex.get('sub_question', '').strip())
    print(f"\nWritten {written} examples to {args.output}")
    if skipped > 0:
        print(f"Skipped {skipped} examples with empty sub-questions")
    print(f"  SUFFICIENT (1): {labels[1]} ({labels[1]/written*100:.1f}%)")
    print(f"  INSUFFICIENT (0): {labels[0]} ({labels[0]/written*100:.1f}%)")

    # Sample text length statistics
    sample_lengths = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            ex = json.loads(line)
            if ex.get('sub_question', '').strip():
                text = format_example(ex, max_passage_chars=args.max_passage_chars)
                sample_lengths.append(len(text))

    if sample_lengths:
        avg_len = sum(sample_lengths) / len(sample_lengths)
        max_len = max(sample_lengths)
        print(f"\n  Avg text length: {avg_len:.0f} chars (~{avg_len/4:.0f} tokens)")
        print(f"  Max text length: {max_len} chars (~{max_len/4:.0f} tokens)")


if __name__ == "__main__":
    main()
