"""
Contextual bandit for intervention decisions.
Offline, from existing data. No GPU, no API calls.

Run on VM: cd ~/ircot && python contextual_bandit.py
"""
import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load augmentation results (has per-hop flags and correctness)
with open('results/section4_full_dev.json') as f:
    aug = json.load(f)

examples = aug['examples']
print(f"Loaded {len(examples)} examples")

# Build per-question training data
# For each question, we know:
#   - baseline correct (no intervention)
#   - always correct (intervene everywhere)
#   - deberta correct (intervene where flagged)
#   - per-hop flags
#   - n_hops
#
# We want to learn: given features, should we intervene on this question?
# Reward = correct with policy - correct without policy

# Features per question (not per hop — we decide per question since
# we only observe question-level outcomes)
X = []
y_intervene_helps = []  # 1 if always > baseline, 0 if always <= baseline

for ex in examples:
    n_hops = ex['n_hops']
    flags = ex['deberta']['flags']
    n_flags = sum(flags)
    
    base_correct = int(ex['baseline']['correct'])
    always_correct = int(ex['always']['correct'])
    
    # Features
    feat = [
        n_flags,                          # total flags
        n_flags / n_hops,                 # flag rate
        n_hops,                           # chain length
        flags[0] if len(flags) > 0 else 0,  # hop 1 flagged
        flags[-1] if len(flags) > 0 else 0, # last hop flagged
        1 if n_flags == 0 else 0,         # no flags indicator
        1 if n_flags == n_hops else 0,    # all flagged indicator
    ]
    X.append(feat)
    
    # Label: does intervention help this question?
    y_intervene_helps.append(1 if always_correct > base_correct else 0)

X = np.array(X)
y = np.array(y_intervene_helps)

feature_names = ['n_flags', 'flag_rate', 'n_hops', 'hop1_flag', 
                 'last_hop_flag', 'no_flags', 'all_flagged']

print(f"Features: {X.shape}")
print(f"Intervention helps: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
print(f"Intervention hurts or neutral: {(1-y).sum()}/{len(y)} ({(1-y).mean()*100:.1f}%)")
print()

# ----- Cross-validated evaluation -----
print("=" * 60)
print("CROSS-VALIDATED POLICY EVALUATION")
print("=" * 60)

# Baseline policies (evaluated on full data for reference)
base_correct_all = np.array([int(ex['baseline']['correct']) for ex in examples])
always_correct_all = np.array([int(ex['always']['correct']) for ex in examples])
deberta_correct_all = np.array([int(ex['deberta']['correct']) for ex in examples])

n = len(examples)
print(f"\nFixed policies (full data, n={n}):")
print(f"  Never intervene (baseline): {base_correct_all.sum()}/{n} = {base_correct_all.mean()*100:.1f}%")
print(f"  Always intervene:           {always_correct_all.sum()}/{n} = {always_correct_all.mean()*100:.1f}%")
print(f"  DeBERTa threshold=0.5:      {deberta_correct_all.sum()}/{n} = {deberta_correct_all.mean()*100:.1f}%")

# Learned policy: for each question, predict whether to intervene
# If intervene: use always_correct outcome
# If skip: use baseline outcome
print()
print("Learned policy (5-fold cross-validation):")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []
fold_decisions = []

all_policy_correct = np.zeros(n, dtype=int)
all_policy_intervene = np.zeros(n, dtype=int)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Policy decisions on test set
    policy_intervene = clf.predict(X_test)
    
    # Evaluate: if policy says intervene, use always outcome; else use baseline
    policy_correct = np.where(
        policy_intervene == 1,
        always_correct_all[test_idx],
        base_correct_all[test_idx]
    )
    
    fold_acc = policy_correct.mean() * 100
    fold_accs.append(fold_acc)
    
    n_intervene = policy_intervene.sum()
    fold_decisions.append(n_intervene / len(test_idx))
    
    all_policy_correct[test_idx] = policy_correct
    all_policy_intervene[test_idx] = policy_intervene
    
    print(f"  Fold {fold_idx+1}: {fold_acc:.1f}% accuracy, "
          f"intervene on {n_intervene}/{len(test_idx)} ({n_intervene/len(test_idx)*100:.1f}%)")

overall_acc = all_policy_correct.mean() * 100
overall_intervene_rate = all_policy_intervene.mean() * 100

print()
print(f"  Overall: {overall_acc:.1f}% accuracy, "
      f"intervene on {all_policy_intervene.sum()}/{n} ({overall_intervene_rate:.1f}%)")
print(f"  Std across folds: {np.std(fold_accs):.1f}%")

# ----- Summary comparison -----
print()
print("=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)
print(f"{'Policy':<30} {'Accuracy':>9} {'Intervene%':>11} {'Δ vs Base':>10}")
print("-" * 62)
print(f"{'Never (baseline)':<30} {base_correct_all.mean()*100:>8.1f}% {'0.0%':>11} {'—':>10}")
print(f"{'Always':<30} {always_correct_all.mean()*100:>8.1f}% {'100.0%':>11} {(always_correct_all.mean()-base_correct_all.mean())*100:>+9.1f}%")
print(f"{'DeBERTa threshold=0.5':<30} {deberta_correct_all.mean()*100:>8.1f}% {'50.0%':>11} {(deberta_correct_all.mean()-base_correct_all.mean())*100:>+9.1f}%")
print(f"{'Learned policy (CV)':<30} {overall_acc:>8.1f}% {overall_intervene_rate:>10.1f}% {(overall_acc-base_correct_all.mean()*100):>+9.1f}%")

# ----- Feature importance -----
print()
print("=" * 60)
print("FEATURE IMPORTANCE (final model on all data)")
print("=" * 60)

clf_full = LogisticRegression(random_state=42, max_iter=1000)
clf_full.fit(X, y)

coefs = clf_full.coef_[0]
for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
    direction = "→ intervene" if coef > 0 else "→ skip"
    print(f"  {name:<20} {coef:>+7.3f}  ({direction})")

print(f"  {'intercept':<20} {clf_full.intercept_[0]:>+7.3f}")

# ----- Stratified by hop count -----
print()
print("=" * 60)
print("STRATIFIED BY HOP COUNT")
print("=" * 60)

for target_hops in [2, 3, 4]:
    mask = np.array([ex['n_hops'] == target_hops for ex in examples])
    if mask.sum() == 0:
        continue
    
    b = base_correct_all[mask].mean() * 100
    a = always_correct_all[mask].mean() * 100
    d = deberta_correct_all[mask].mean() * 100
    l = all_policy_correct[mask].mean() * 100
    li = all_policy_intervene[mask].mean() * 100
    
    print(f"\n{target_hops}-hop (n={mask.sum()}):")
    print(f"  Baseline:  {b:.1f}%")
    print(f"  Always:    {a:.1f}%")
    print(f"  DeBERTa:   {d:.1f}%")
    print(f"  Learned:   {l:.1f}% (intervene {li:.1f}%)")

# ----- Also try on reranking data -----
print()
print("=" * 60)
print("TRANSFER TO RERANKING (train on augment, test on rerank)")
print("=" * 60)

try:
    with open('results/section4_rerank_full.json') as f:
        rerank = json.load(f)
    
    rerank_examples = rerank['examples']
    
    X_rerank = []
    for ex in rerank_examples:
        n_hops = ex['n_hops']
        flags = ex['deberta_rerank']['flags']
        n_flags = sum(flags)
        
        feat = [
            n_flags,
            n_flags / n_hops,
            n_hops,
            flags[0] if len(flags) > 0 else 0,
            flags[-1] if len(flags) > 0 else 0,
            1 if n_flags == 0 else 0,
            1 if n_flags == n_hops else 0,
        ]
        X_rerank.append(feat)
    
    X_rerank = np.array(X_rerank)
    
    base_rerank = np.array([int(ex['baseline']['correct']) for ex in rerank_examples])
    always_rerank = np.array([int(ex['always_rerank']['correct']) for ex in rerank_examples])
    deberta_rerank = np.array([int(ex['deberta_rerank']['correct']) for ex in rerank_examples])
    
    # Use model trained on augmentation data
    policy_rerank = clf_full.predict(X_rerank)
    policy_correct_rerank = np.where(
        policy_rerank == 1,
        always_rerank,
        base_rerank
    )
    
    n_r = len(rerank_examples)
    print(f"n = {n_r}")
    print(f"{'Policy':<30} {'Accuracy':>9} {'Intervene%':>11}")
    print("-" * 52)
    print(f"{'Never (baseline)':<30} {base_rerank.mean()*100:>8.1f}% {'0.0%':>11}")
    print(f"{'Always-rerank':<30} {always_rerank.mean()*100:>8.1f}% {'100.0%':>11}")
    print(f"{'DeBERTa threshold=0.5':<30} {deberta_rerank.mean()*100:>8.1f}% {'50.0%':>11}")
    print(f"{'Learned policy (transfer)':<30} {policy_correct_rerank.mean()*100:>8.1f}% {policy_rerank.mean()*100:>10.1f}%")

except Exception as e:
    print(f"Could not load reranking data: {e}")
