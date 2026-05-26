#!/usr/bin/env python3
"""
Extract Training Data from Counterfactual Analysis Results

Converts counterfactual SUFFICIENT/INSUFFICIENT labels into clean
(sub-question, passages, label) examples for:
  - Simple baseline experiments (retrieval score, lexical overlap, NLI)
  - DeBERTa fine-tuning
  - SFT on larger models

Each example contains:
  - original_question: the full multi-hop question
  - sub_question: the query at this specific hop
  - passages: list of retrieved passages (title + text)
  - passage_texts: concatenated passage text (for simple feature extraction)
  - retrieval_scores: BM25 scores per passage
  - retrieval_recall: fraction of gold passages retrieved
  - hop_number: which hop (1, 2, 3, 4)
  - label: SUFFICIENT or INSUFFICIENT
  - label_binary: 1 (sufficient) or 0 (insufficient)

Usage:
    # Extract from dev counterfactual (for evaluation)
    python extract_training_data.py \
        --input results/counterfactual_all_4.1.json \
        --output results/sufficiency_data_dev.jsonl

    # Extract from train counterfactual (for training)
    python extract_training_data.py \
        --input results/counterfactual_train_4.1.json \
        --output results/sufficiency_data_train.jsonl

    # Show statistics
    python extract_training_data.py \
        --input results/counterfactual_train_4.1.json \
        --output results/sufficiency_data_train.jsonl \
        --stats
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter


def normalize_text(s: str) -> str:
    """Normalize text for feature computation."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def compute_lexical_overlap(query: str, passage_text: str) -> float:
    """Compute word-level Jaccard overlap between query and passage."""
    query_words = set(normalize_text(query).split())
    passage_words = set(normalize_text(passage_text).split())
    if not query_words or not passage_words:
        return 0.0
    intersection = query_words & passage_words
    union = query_words | passage_words
    return len(intersection) / len(union)


def compute_query_coverage(query: str, passage_text: str) -> float:
    """What fraction of query words appear in the passage?"""
    query_words = set(normalize_text(query).split())
    passage_words = set(normalize_text(passage_text).split())
    if not query_words:
        return 0.0
    return len(query_words & passage_words) / len(query_words)


def compute_features(sub_question: str, passages: list) -> dict:
    """
    Compute simple baseline features for a (sub_question, passages) pair.
    These are used to test whether simple signals predict sufficiency.
    """
    # Concatenate all passage texts
    all_text = " ".join(p.get('text', '') for p in passages)

    # Feature 1: Average retrieval score
    scores = [p.get('score', 0.0) for p in passages]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    # Feature 2: Lexical overlap between sub-question and passages
    lexical_overlap = compute_lexical_overlap(sub_question, all_text)

    # Feature 3: Query coverage (what % of query words appear in passages)
    query_coverage = compute_query_coverage(sub_question, all_text)

    # Feature 4: Total passage length (word count)
    passage_word_count = len(all_text.split())

    # Feature 5: Number of passages retrieved
    num_passages = len(passages)

    return {
        'avg_retrieval_score': round(avg_score, 4),
        'max_retrieval_score': round(max_score, 4),
        'lexical_overlap': round(lexical_overlap, 4),
        'query_coverage': round(query_coverage, 4),
        'passage_word_count': passage_word_count,
        'num_passages': num_passages,
    }


def extract_examples(counterfactual_path: str) -> list:
    """Extract clean examples from counterfactual analysis results."""

    with open(counterfactual_path) as f:
        data = json.load(f)

    examples = []

    for result in data['results']:
        original_question = result.get('question', '')
        qid = result.get('qid', '')

        for hop_result in result.get('hop_results', []):
            sub_question = hop_result.get('sub_question', '')
            passages = hop_result.get('retrieved_passages', [])
            label = hop_result.get('label', '')
            retrieval_recall = hop_result.get('retrieval_recall', 0.0)
            hop_number = hop_result.get('hop', 0)

            # Skip examples with empty sub-questions
            if not sub_question.strip():
                continue

            # Compute simple features for baseline experiments
            features = compute_features(sub_question, passages)

            # Format passage text for models
            passage_texts = []
            for p in passages:
                title = p.get('title', '')
                text = p.get('text', '')
                passage_texts.append(f"{title}: {text}")

            example = {
                'qid': qid,
                'original_question': original_question,
                'sub_question': sub_question,
                'passages': passages,
                'passage_text_combined': "\n\n".join(passage_texts),
                'retrieval_recall': round(retrieval_recall, 4),
                'hop_number': hop_number,
                'label': label,
                'label_binary': 1 if label == 'SUFFICIENT' else 0,
                'features': features,
            }

            examples.append(example)

    return examples


def print_statistics(examples: list):
    """Print detailed statistics about the extracted data."""
    print("\n" + "=" * 60)
    print("EXTRACTED DATA STATISTICS")
    print("=" * 60)

    # Basic counts
    total = len(examples)
    labels = Counter(e['label'] for e in examples)
    print(f"\nTotal examples: {total}")
    print(f"  SUFFICIENT:   {labels['SUFFICIENT']} ({labels['SUFFICIENT']/total*100:.1f}%)")
    print(f"  INSUFFICIENT: {labels['INSUFFICIENT']} ({labels['INSUFFICIENT']/total*100:.1f}%)")

    # Per-hop breakdown
    print(f"\nPer-hop breakdown:")
    hops = sorted(set(e['hop_number'] for e in examples))
    for h in hops:
        hop_examples = [e for e in examples if e['hop_number'] == h]
        hop_suf = sum(1 for e in hop_examples if e['label'] == 'SUFFICIENT')
        hop_total = len(hop_examples)
        print(f"  Hop {h}: {hop_total} examples ({hop_suf/hop_total*100:.1f}% sufficient)")

    # Feature statistics (for baseline feasibility)
    print(f"\nFeature statistics (for baseline prediction):")

    for feature_name in ['avg_retrieval_score', 'lexical_overlap', 'query_coverage', 'passage_word_count']:
        suf_vals = [e['features'][feature_name] for e in examples if e['label'] == 'SUFFICIENT']
        insuf_vals = [e['features'][feature_name] for e in examples if e['label'] == 'INSUFFICIENT']

        suf_mean = sum(suf_vals) / len(suf_vals) if suf_vals else 0
        insuf_mean = sum(insuf_vals) / len(insuf_vals) if insuf_vals else 0
        gap = abs(suf_mean - insuf_mean)

        print(f"\n  {feature_name}:")
        print(f"    SUFFICIENT mean:   {suf_mean:.4f}")
        print(f"    INSUFFICIENT mean: {insuf_mean:.4f}")
        print(f"    Gap: {gap:.4f}", end="")
        if gap < 0.05 * max(suf_mean, insuf_mean, 0.001):
            print(" (very small — poor predictor)")
        elif gap < 0.15 * max(suf_mean, insuf_mean, 0.001):
            print(" (moderate — weak predictor)")
        else:
            print(" (large — potential predictor)")

    # Example samples
    print(f"\n\nSample SUFFICIENT example:")
    suf_ex = next(e for e in examples if e['label'] == 'SUFFICIENT')
    print(f"  Question: {suf_ex['original_question'][:80]}...")
    print(f"  Sub-question: {suf_ex['sub_question'][:80]}...")
    print(f"  Passages: {len(suf_ex['passages'])} retrieved")
    print(f"  Retrieval recall: {suf_ex['retrieval_recall']}")

    print(f"\nSample INSUFFICIENT example:")
    insuf_ex = next(e for e in examples if e['label'] == 'INSUFFICIENT')
    print(f"  Question: {insuf_ex['original_question'][:80]}...")
    print(f"  Sub-question: {insuf_ex['sub_question'][:80]}...")
    print(f"  Passages: {len(insuf_ex['passages'])} retrieved")
    print(f"  Retrieval recall: {insuf_ex['retrieval_recall']}")


def main():
    parser = argparse.ArgumentParser(description="Extract training data from counterfactual results")
    parser.add_argument("--input", required=True, help="Path to counterfactual analysis JSON")
    parser.add_argument("--output", required=True, help="Output path for extracted data (JSONL)")
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics")
    args = parser.parse_args()

    print(f"Extracting from {args.input}...")
    examples = extract_examples(args.input)

    # Save as JSONL (one example per line — standard for training data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Saved {len(examples)} examples to {args.output}")

    if args.stats:
        print_statistics(examples)

    # Always print basic summary
    labels = Counter(e['label'] for e in examples)
    print(f"\nSummary: {len(examples)} total — "
          f"SUFFICIENT: {labels['SUFFICIENT']} ({labels['SUFFICIENT']/len(examples)*100:.1f}%), "
          f"INSUFFICIENT: {labels['INSUFFICIENT']} ({labels['INSUFFICIENT']/len(examples)*100:.1f}%)")


if __name__ == "__main__":
    main()
