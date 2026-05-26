#!/usr/bin/env python3
"""
Baseline Classifiers for Sufficiency Prediction

Tests whether simple, surface-level features can predict whether a
retrieval step is SUFFICIENT or INSUFFICIENT.

This answers the scientific question: "Is sufficiency a shallow property
detectable from simple signals, or does it require deeper semantic
understanding?"

Baselines tested:
1. Majority baseline — always predicts the most common label
2. Logistic regression — linear combination of simple features
3. Random forest — non-linear combination of simple features
4. Individual feature thresholds — each feature alone as a predictor

Features used (computed by extract_training_data.py):
- avg_retrieval_score: BM25 score averaged across retrieved passages
- max_retrieval_score: highest BM25 score among retrieved passages
- lexical_overlap: Jaccard word overlap between sub-question and passages
- query_coverage: fraction of sub-question words found in passages
- passage_word_count: total words across all retrieved passages
- num_passages: number of passages retrieved
- retrieval_recall: fraction of gold passage titles retrieved

Usage:
    python run_baselines.py \
        --train results/sufficiency_data_train.jsonl \
        --dev results/sufficiency_data_dev.jsonl
"""

import json
import argparse
import numpy as np
from collections import Counter


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_data(path: str):
    """Load extracted sufficiency data and return features + labels."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Extract feature vectors and labels
    feature_names = [
        'avg_retrieval_score',
        'max_retrieval_score',
        'lexical_overlap',
        'query_coverage',
        'passage_word_count',
        'num_passages',
    ]

    X = []
    y = []
    extra_features = []  # retrieval_recall (used as a separate baseline)

    for ex in examples:
        features = ex['features']
        row = [features[name] for name in feature_names]
        X.append(row)
        y.append(ex['label_binary'])
        extra_features.append({
            'retrieval_recall': ex['retrieval_recall'],
            'hop_number': ex['hop_number'],
        })

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    return X, y, feature_names, extra_features, examples


# ──────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1 for binary classification."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

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


# ──────────────────────────────────────────────────────────────────────
# Baseline 1: Majority class
# ──────────────────────────────────────────────────────────────────────

def majority_baseline(y_train, y_dev):
    """Always predict the most common class in training data."""
    counts = Counter(y_train)
    majority_class = counts.most_common(1)[0][0]
    y_pred = np.full_like(y_dev, majority_class)
    return y_pred


# ──────────────────────────────────────────────────────────────────────
# Baseline 2: Logistic Regression
# ──────────────────────────────────────────────────────────────────────

def logistic_regression_baseline(X_train, y_train, X_dev, learning_rate=0.01, epochs=1000):
    """
    Simple logistic regression from scratch (no sklearn dependency).
    Uses gradient descent with L2 regularization.
    """
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_dev_norm = (X_dev - mean) / std

    # Add bias term
    X_train_b = np.column_stack([X_train_norm, np.ones(len(X_train_norm))])
    X_dev_b = np.column_stack([X_dev_norm, np.ones(len(X_dev_norm))])

    # Initialize weights
    n_features = X_train_b.shape[1]
    weights = np.zeros(n_features)

    # Train with gradient descent
    reg_lambda = 0.01  # L2 regularization

    for epoch in range(epochs):
        # Forward pass
        z = X_train_b @ weights
        z = np.clip(z, -500, 500)  # prevent overflow
        predictions = 1 / (1 + np.exp(-z))

        # Gradient
        error = predictions - y_train
        gradient = (X_train_b.T @ error) / len(y_train) + reg_lambda * weights
        weights -= learning_rate * gradient

    # Predict on dev
    z_dev = X_dev_b @ weights
    z_dev = np.clip(z_dev, -500, 500)
    probs = 1 / (1 + np.exp(-z_dev))
    y_pred = (probs >= 0.5).astype(int)

    return y_pred, weights[:-1]  # return weights without bias for analysis


# ──────────────────────────────────────────────────────────────────────
# Baseline 3: Random Forest (simple decision tree ensemble)
# ──────────────────────────────────────────────────────────────────────

def simple_decision_stump(X, y, feature_idx):
    """Find the best threshold for a single feature."""
    values = X[:, feature_idx]
    best_acc = 0
    best_threshold = 0
    best_direction = 1

    # Try several thresholds
    percentiles = np.percentile(values, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    for threshold in percentiles:
        for direction in [1, -1]:
            if direction == 1:
                pred = (values >= threshold).astype(int)
            else:
                pred = (values < threshold).astype(int)
            acc = np.mean(pred == y)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_direction = direction

    return best_threshold, best_direction, best_acc


def random_forest_baseline(X_train, y_train, X_dev, n_trees=100):
    """
    Simple random forest using bootstrapping and feature subsampling.
    Each tree is a single decision stump (best single-feature split).
    """
    n_samples, n_features = X_train.shape
    trees = []

    for _ in range(n_trees):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Random feature subset (sqrt of total features)
        n_sub = max(2, int(np.sqrt(n_features)))
        feature_subset = np.random.choice(n_features, size=n_sub, replace=False)

        # Find best stump among subset
        best_feat = None
        best_thresh = 0
        best_dir = 1
        best_acc = 0

        for feat_idx in feature_subset:
            threshold, direction, acc = simple_decision_stump(X_boot, y_boot, feat_idx)
            if acc > best_acc:
                best_acc = acc
                best_feat = feat_idx
                best_thresh = threshold
                best_dir = direction

        trees.append((best_feat, best_thresh, best_dir))

    # Predict on dev using majority vote
    votes = np.zeros((len(X_dev), 2))
    for feat_idx, threshold, direction in trees:
        if direction == 1:
            pred = (X_dev[:, feat_idx] >= threshold).astype(int)
        else:
            pred = (X_dev[:, feat_idx] < threshold).astype(int)
        for i, p in enumerate(pred):
            votes[i, p] += 1

    y_pred = np.argmax(votes, axis=1)
    return y_pred


# ──────────────────────────────────────────────────────────────────────
# Baseline 4: Individual feature thresholds
# ──────────────────────────────────────────────────────────────────────

def individual_feature_baselines(X_train, y_train, X_dev, y_dev, feature_names):
    """Test each feature individually as a sufficiency predictor."""
    results = {}
    for i, name in enumerate(feature_names):
        threshold, direction, train_acc = simple_decision_stump(X_train, y_train, i)

        if direction == 1:
            y_pred = (X_dev[:, i] >= threshold).astype(int)
        else:
            y_pred = (X_dev[:, i] < threshold).astype(int)

        metrics = compute_metrics(y_dev, y_pred)
        results[name] = metrics
        results[name]['threshold'] = round(float(threshold), 4)

    return results


# ──────────────────────────────────────────────────────────────────────
# Baseline 5: Retrieval recall as predictor
# ──────────────────────────────────────────────────────────────────────

def retrieval_recall_baseline(extra_features_train, y_train, extra_features_dev, y_dev):
    """
    Use retrieval recall directly as a sufficiency predictor.
    This is the most natural baseline — "if you retrieved the gold passages,
    the step should be sufficient."
    """
    recalls_train = np.array([ef['retrieval_recall'] for ef in extra_features_train])
    recalls_dev = np.array([ef['retrieval_recall'] for ef in extra_features_dev])

    # Find best threshold on train
    best_acc = 0
    best_thresh = 0.5
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pred = (recalls_train >= thresh).astype(int)
        acc = np.mean(pred == y_train)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Apply to dev
    y_pred = (recalls_dev >= best_thresh).astype(int)
    return y_pred, best_thresh


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baseline classifiers for sufficiency prediction")
    parser.add_argument("--train", required=True, help="Training data JSONL")
    parser.add_argument("--dev", required=True, help="Dev data JSONL")
    parser.add_argument("--output", default="results/baseline_results.json", help="Output path")
    args = parser.parse_args()

    np.random.seed(42)

    # Load data
    print("Loading data...")
    X_train, y_train, feature_names, extra_train, _ = load_data(args.train)
    X_dev, y_dev, _, extra_dev, _ = load_data(args.dev)

    print(f"Train: {len(X_train)} examples ({sum(y_train)}/{len(y_train)} sufficient)")
    print(f"Dev:   {len(X_dev)} examples ({sum(y_dev)}/{len(y_dev)} sufficient)")

    all_results = {}

    # ── Baseline 1: Majority ──
    print("\n1. Majority baseline...")
    y_pred = majority_baseline(y_train, y_dev)
    metrics = compute_metrics(y_dev, y_pred)
    all_results['majority'] = metrics
    print(f"   Accuracy: {metrics['accuracy']:.1%}")

    # ── Baseline 2: Logistic Regression ──
    print("\n2. Logistic regression (all simple features)...")
    y_pred, weights = logistic_regression_baseline(X_train, y_train, X_dev)
    metrics = compute_metrics(y_dev, y_pred)
    all_results['logistic_regression'] = metrics
    print(f"   Accuracy: {metrics['accuracy']:.1%}  F1: {metrics['f1']:.3f}")

    # Show feature weights
    print("   Feature weights (higher = more predictive of SUFFICIENT):")
    weight_pairs = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)
    for name, w in weight_pairs:
        print(f"     {name:25s}: {w:+.4f}")

    # ── Baseline 3: Random Forest ──
    print("\n3. Random forest (100 trees)...")
    y_pred = random_forest_baseline(X_train, y_train, X_dev)
    metrics = compute_metrics(y_dev, y_pred)
    all_results['random_forest'] = metrics
    print(f"   Accuracy: {metrics['accuracy']:.1%}  F1: {metrics['f1']:.3f}")

    # ── Baseline 4: Individual features ──
    print("\n4. Individual feature thresholds:")
    individual_results = individual_feature_baselines(X_train, y_train, X_dev, y_dev, feature_names)
    for name, metrics in individual_results.items():
        print(f"   {name:25s}: Accuracy={metrics['accuracy']:.1%}  F1={metrics['f1']:.3f}  (threshold={metrics['threshold']})")
    all_results['individual_features'] = individual_results

    # ── Baseline 5: Retrieval recall ──
    print("\n5. Retrieval recall as predictor...")
    y_pred, thresh = retrieval_recall_baseline(extra_train, y_train, extra_dev, y_dev)
    metrics = compute_metrics(y_dev, y_pred)
    all_results['retrieval_recall'] = metrics
    all_results['retrieval_recall']['threshold'] = thresh
    print(f"   Accuracy: {metrics['accuracy']:.1%}  F1: {metrics['f1']:.3f}  (threshold={thresh})")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: BASELINE RESULTS FOR SUFFICIENCY PREDICTION")
    print("=" * 70)
    print(f"{'Method':<35} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 70)

    summary_rows = [
        ('Majority baseline', all_results['majority']),
        ('Retrieval recall', all_results['retrieval_recall']),
        ('Logistic regression', all_results['logistic_regression']),
        ('Random forest', all_results['random_forest']),
    ]

    for name, m in summary_rows:
        print(f"{name:<35} {m['accuracy']:>9.1%} {m['f1']:>10.3f} {m['precision']:>10.3f} {m['recall']:>10.3f}")

    print("-" * 70)
    print("  (A trained model needs to significantly beat these baselines")
    print("   to prove sufficiency requires semantic understanding)")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
