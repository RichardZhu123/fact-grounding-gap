#!/usr/bin/env python3
"""
Baseline Classifiers for Fact-Grounded Answerability

Tests whether simple features can predict if the system's accumulated
retrieved passages contain enough information to answer each sub-question.

Uses labels from fact_grounding_final.py (ANSWERABLE / NOT-ANSWERABLE).

Since we don't have separate train/dev for these labels yet (only dev),
we run leave-one-out style evaluation and also report feature correlations.

Usage:
    python baselines_fact_grounded.py \
        --input results/fact_grounded_final_dev.jsonl
"""

import json
import argparse
import re
import numpy as np
from collections import Counter


def load_data(path: str):
    """Load fact-grounded data and extract features + labels."""
    examples = []
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e['label'] == 'UNKNOWN':
                continue
            examples.append(e)
    return examples


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }


def best_threshold(values, labels, directions=[1, -1]):
    """Find best threshold for a single feature."""
    best_acc = 0
    best_thresh = 0
    best_dir = 1
    best_preds = None

    percentiles = np.percentile(values, list(range(5, 96, 5)))
    for thresh in percentiles:
        for d in directions:
            if d == 1:
                preds = (values >= thresh).astype(int)
            else:
                preds = (values < thresh).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                best_dir = d
                best_preds = preds

    return best_acc, best_thresh, best_dir, best_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fact-grounded JSONL from fact_grounding_final.py")
    args = parser.parse_args()

    examples = load_data(args.input)
    labels = np.array([e['label_binary'] for e in examples])

    print(f"{'='*70}")
    print(f"BASELINES FOR FACT-GROUNDED ANSWERABILITY")
    print(f"{'='*70}")
    print(f"Total examples: {len(examples)}")
    print(f"  ANSWERABLE: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  NOT-ANSWERABLE: {len(labels)-sum(labels)} ({(1-labels.mean())*100:.1f}%)")

    majority_class = 1 if labels.mean() > 0.5 else 0
    majority_acc = max(labels.mean(), 1 - labels.mean())
    majority_preds = np.full_like(labels, majority_class)

    # ── Baseline 1: Majority ──
    print(f"\n1. Majority baseline:")
    m = compute_metrics(labels, majority_preds)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}")

    # ── Baseline 2: Overall retrieval recall ──
    print(f"\n2. Overall retrieval recall:")
    recalls = np.array([e['overall_recall'] for e in examples])
    acc, thresh, direction, preds = best_threshold(recalls, labels)
    m = compute_metrics(labels, preds)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}  (threshold: {thresh:.2f})")

    # Mean recall comparison
    recall_ans = recalls[labels == 1].mean()
    recall_not = recalls[labels == 0].mean()
    print(f"   Avg recall when ANSWERABLE: {recall_ans:.3f}")
    print(f"   Avg recall when NOT-ANSWERABLE: {recall_not:.3f}")
    print(f"   Gap: {abs(recall_ans - recall_not):.3f}")

    # ── Baseline 3: Gold paragraph present ──
    print(f"\n3. Gold paragraph present (oracle-ish):")
    gold_present = np.array([1 if e['gold_para_present'] else 0 for e in examples])
    m = compute_metrics(labels, gold_present)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}")

    # ── Baseline 4: String match ──
    print(f"\n4. String match (intermediate answer in passages):")
    string_labels = np.array([e['string_match_label'] for e in examples])
    m = compute_metrics(labels, string_labels)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}")

    # ── Baseline 5: Hop number ──
    print(f"\n5. Hop number (predict answerable if hop <= threshold):")
    hops = np.array([e['hop_number'] for e in examples])
    acc, thresh, direction, preds = best_threshold(hops, labels)
    m = compute_metrics(labels, preds)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}  (threshold: hop {'<=' if direction == -1 else '>='} {thresh:.0f})")

    # ── Baseline 6: Number of accumulated passages ──
    print(f"\n6. Number of accumulated passages:")
    num_passages = np.array([e['num_accumulated_passages'] for e in examples])
    acc, thresh, direction, preds = best_threshold(num_passages, labels)
    m = compute_metrics(labels, preds)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}  (threshold: {thresh:.0f})")

    # ── Baseline 7: Logistic regression on all features ──
    print(f"\n7. Logistic regression (all features combined):")

    # Build feature matrix
    X = np.column_stack([
        recalls,
        gold_present,
        hops,
        num_passages,
    ])

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    X_b = np.column_stack([X_norm, np.ones(len(X_norm))])

    # Train logistic regression (gradient descent)
    weights = np.zeros(X_b.shape[1])
    lr = 0.01
    reg = 0.01

    for epoch in range(2000):
        z = np.clip(X_b @ weights, -500, 500)
        preds_prob = 1 / (1 + np.exp(-z))
        error = preds_prob - labels
        gradient = (X_b.T @ error) / len(labels) + reg * weights
        weights -= lr * gradient

    preds_prob = 1 / (1 + np.exp(-np.clip(X_b @ weights, -500, 500)))
    preds = (preds_prob >= 0.5).astype(int)
    m = compute_metrics(labels, preds)
    print(f"   Accuracy: {m['accuracy']:.1%}  F1: {m['f1']:.3f}")

    feature_names = ['recall', 'gold_para_present', 'hop_number', 'num_passages']
    print(f"   Feature weights:")
    for name, w in zip(feature_names, weights[:-1]):
        print(f"     {name:25s}: {w:+.4f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<45} {'Accuracy':>10} {'F1':>10}")
    print(f"{'-'*65}")

    results = [
        ('Majority baseline', compute_metrics(labels, majority_preds)),
        ('Retrieval recall (threshold)', compute_metrics(labels, preds)),
        ('Gold paragraph present', compute_metrics(labels, gold_present)),
        ('String match', compute_metrics(labels, string_labels)),
    ]

    # Recompute for summary
    _, _, _, recall_preds = best_threshold(recalls, labels)
    _, _, _, hop_preds = best_threshold(hops, labels)

    summary = [
        ('Majority baseline', compute_metrics(labels, majority_preds)),
        ('Retrieval recall', compute_metrics(labels, recall_preds)),
        ('Hop number', compute_metrics(labels, hop_preds)),
        ('Gold paragraph present (oracle)', compute_metrics(labels, gold_present)),
        ('String match (uses gold answer)', compute_metrics(labels, string_labels)),
        ('Logistic regression', compute_metrics(labels, preds)),
    ]

    for name, m in summary:
        print(f"  {name:<43} {m['accuracy']:>9.1%} {m['f1']:>10.3f}")

    print(f"\n  Note: Gold paragraph and string match use gold annotations")
    print(f"  (not available at inference time). They are upper bounds.")
    print(f"  A trained model needs to beat majority and retrieval recall")
    print(f"  without access to gold information.")

    # Save
    output_path = args.input.replace('.jsonl', '_baselines.json')
    with open(output_path, 'w') as f:
        json.dump({name: m for name, m in summary}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
