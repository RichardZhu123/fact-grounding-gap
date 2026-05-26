"""
Quick HotpotQA diagnostics from existing results.
Run on VM: cd ~/ircot && python hotpotqa_diagnostics.py
"""
import json
from collections import defaultdict

with open('results/hotpotqa_full.json') as f:
    d = json.load(f)

examples = d['examples']
n = len(examples)

# Auto-detect key names
ex0 = examples[0]
if 'deberta' in ex0:
    dkey, akey = 'deberta', 'always'
elif 'deberta_rerank' in ex0:
    dkey, akey = 'deberta_rerank', 'always_rerank'
else:
    dkey = [k for k in ex0 if 'deberta' in k.lower()][0]
    akey = [k for k in ex0 if 'always' in k.lower()][0]

print(f"HotpotQA: n={n}, all 2-hop")
print(f"Detected keys: deberta='{dkey}', always='{akey}'")
print()

# 1. Flag-count degradation (0, 1, 2 flags)
by_flags = defaultdict(lambda: {'n': 0, 'base': 0, 'always': 0, 'deberta': 0})
for ex in examples:
    nf = sum(ex[dkey]['flags'])
    by_flags[nf]['n'] += 1
    if ex['baseline']['correct']:
        by_flags[nf]['base'] += 1
    if ex[akey]['correct']:
        by_flags[nf]['always'] += 1
    if ex[dkey]['correct']:
        by_flags[nf]['deberta'] += 1

print("FLAG-COUNT DEGRADATION (HotpotQA)")
print(f"{'Flags':>5} | {'n':>6} | {'Baseline':>9} | {'Always':>9} | {'DeBERTa':>9} | {'D-Base':>7}")
print("-" * 60)
for k in sorted(by_flags):
    s = by_flags[k]
    ba = s['base'] / s['n'] * 100
    al = s['always'] / s['n'] * 100
    da = s['deberta'] / s['n'] * 100
    print(f"{k:>5} | {s['n']:>6} | {ba:>8.1f}% | {al:>8.1f}% | {da:>8.1f}% | {da-ba:>+6.1f}%")

# 2. Statistical tests
try:
    import numpy as np
    from scipy import stats
    
    b = np.array([1 if ex['baseline']['correct'] else 0 for ex in examples])
    a = np.array([1 if ex[akey]['correct'] else 0 for ex in examples])
    dd = np.array([1 if ex[dkey]['correct'] else 0 for ex in examples])
    
    def mcnemar(x, y):
        n10 = int(np.sum((x == 1) & (y == 0)))
        n01 = int(np.sum((x == 0) & (y == 1)))
        if n10 + n01 == 0:
            return 1.0, n10, n01
        chi2 = (abs(n10 - n01) - 1)**2 / (n10 + n01)
        p = 1 - stats.chi2.cdf(chi2, 1)
        return p, n10, n01
    
    print()
    print("McNEMAR'S TESTS (HotpotQA)")
    p_db, db10, db01 = mcnemar(dd, b)
    print(f"  DeBERTa vs Baseline: D-only={db01}, B-only={db10}, p={p_db:.2e}")
    p_da, da10, da01 = mcnemar(dd, a)
    print(f"  DeBERTa vs Always:   D-only={da01}, A-only={da10}, p={p_da:.2e}")
    p_ab, ab10, ab01 = mcnemar(a, b)
    print(f"  Always vs Baseline:  A-only={ab01}, B-only={ab10}, p={p_ab:.2e}")
    
    # Bootstrap CIs
    print()
    print("BOOTSTRAP 95% CIs (HotpotQA)")
    rng = np.random.RandomState(42)
    for name, arr in [('Baseline', b), ('Always', a), ('DeBERTa', dd)]:
        means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(10000)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        print(f"  {name:>10}: {np.mean(arr)*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")
    
    # Spearman
    flags_arr = np.array([sum(ex[dkey]['flags']) for ex in examples])
    rho, p_s = stats.spearmanr(flags_arr, b)
    print()
    print(f"SPEARMAN (flag count vs baseline): rho={rho:.4f}, p={p_s:.2e}")

except ImportError:
    print("\n(scipy not available - skipping statistical tests)")
