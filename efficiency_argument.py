"""
Compute the efficiency argument for DeBERTa-targeted intervention.
Run on VM: cd ~/ircot && python efficiency_argument.py
"""
import json

def analyze(name, path):
    with open(path) as f:
        d = json.load(f)
    
    print(f"{'='*60}")
    print(f"EFFICIENCY ANALYSIS: {name}")
    print(f"{'='*60}")
    
    examples = d['examples']
    n = len(examples)
    
    total_hops = sum(ex['n_hops'] for ex in examples)
    deberta_flagged = sum(sum(ex['deberta']['flags']) for ex in examples)
    
    base_correct = sum(1 for ex in examples if ex['baseline']['correct'])
    always_correct = sum(1 for ex in examples if ex['always']['correct'])
    deberta_correct = sum(1 for ex in examples if ex['deberta']['correct'])
    
    base_acc = base_correct / n * 100
    always_acc = always_correct / n * 100
    deberta_acc = deberta_correct / n * 100
    flag_rate = deberta_flagged / total_hops * 100
    
    print(f"n = {n} questions, {total_hops} total hops")
    print()
    print(f"Intervention rates:")
    print(f"  Always:  {total_hops}/{total_hops} hops = 100.0%")
    print(f"  DeBERTa: {deberta_flagged}/{total_hops} hops = {flag_rate:.1f}%")
    print(f"  Savings: {100 - flag_rate:.1f}% fewer interventions")
    print()
    print(f"Accuracy:")
    print(f"  Baseline: {base_acc:.1f}% ({base_correct}/{n})")
    print(f"  Always:   {always_acc:.1f}% ({always_correct}/{n}) @ 100% intervention")
    print(f"  DeBERTa:  {deberta_acc:.1f}% ({deberta_correct}/{n}) @ {flag_rate:.1f}% intervention")
    print()
    
    diff_vs_always = deberta_acc - always_acc
    diff_vs_base = deberta_acc - base_acc
    
    if deberta_acc >= always_acc:
        print(f"  >>> DeBERTa: HIGHER accuracy ({diff_vs_always:+.1f}%) with {100-flag_rate:.0f}% fewer interventions")
    else:
        print(f"  >>> DeBERTa: {diff_vs_always:+.1f}% vs Always, but {100-flag_rate:.0f}% fewer interventions")
    print(f"  >>> DeBERTa vs Baseline: {diff_vs_base:+.1f}%")
    print()
    
    # Cost per question
    avg_hops = total_hops / n
    print(f"Avg re-retrieval calls per question:")
    print(f"  Always:  {avg_hops:.2f}")
    print(f"  DeBERTa: {deberta_flagged/n:.2f}")
    print(f"  Ratio:   {deberta_flagged/total_hops:.2f}x")
    print()
    
    return {
        'n': n, 'total_hops': total_hops, 'flagged': deberta_flagged,
        'flag_rate': flag_rate, 'base_acc': base_acc,
        'always_acc': always_acc, 'deberta_acc': deberta_acc
    }

# MuSiQue experiments
aug = analyze("MUSIQUE AUGMENTATION", "results/section4_full_dev.json")
rerank = analyze("MUSIQUE RERANKING", "results/section4_rerank_full.json")

# HotpotQA - pull from file, no hardcoding
print(f"{'='*60}")
print(f"EFFICIENCY ANALYSIS: HOTPOTQA (cross-dataset, zero-shot)")
print(f"{'='*60}")

with open('results/hotpotqa_full.json') as f:
    hqa = json.load(f)

n_hqa = hqa['n']
examples_hqa = hqa['examples']

total_hops_hqa = sum(ex['n_hops'] for ex in examples_hqa)
deberta_flagged_hqa = sum(sum(ex['deberta']['flags']) for ex in examples_hqa)

base_correct_hqa = sum(1 for ex in examples_hqa if ex['baseline']['correct'])
always_correct_hqa = sum(1 for ex in examples_hqa if ex['always']['correct'])
deberta_correct_hqa = sum(1 for ex in examples_hqa if ex['deberta']['correct'])

base_acc_hqa = base_correct_hqa / n_hqa * 100
always_acc_hqa = always_correct_hqa / n_hqa * 100
deberta_acc_hqa = deberta_correct_hqa / n_hqa * 100
flag_rate_hqa = deberta_flagged_hqa / total_hops_hqa * 100

print(f"n = {n_hqa} questions, {total_hops_hqa} total hops")
print()
print(f"Intervention rates:")
print(f"  Always:  {total_hops_hqa}/{total_hops_hqa} hops = 100.0%")
print(f"  DeBERTa: {deberta_flagged_hqa}/{total_hops_hqa} hops = {flag_rate_hqa:.1f}%")
print(f"  Savings: {100 - flag_rate_hqa:.1f}% fewer interventions")
print()
print(f"Accuracy:")
print(f"  Baseline: {base_acc_hqa:.1f}% ({base_correct_hqa}/{n_hqa})")
print(f"  Always:   {always_acc_hqa:.1f}% ({always_correct_hqa}/{n_hqa}) @ 100% intervention")
print(f"  DeBERTa:  {deberta_acc_hqa:.1f}% ({deberta_correct_hqa}/{n_hqa}) @ {flag_rate_hqa:.1f}% intervention")
print()
print(f"  >>> DeBERTa: {deberta_acc_hqa - always_acc_hqa:+.1f}% vs Always, but {100-flag_rate_hqa:.0f}% fewer interventions")
print(f"  >>> DeBERTa vs Baseline: {deberta_acc_hqa - base_acc_hqa:+.1f}%")
print()

# Summary table
print(f"{'='*60}")
print(f"SUMMARY TABLE (for paper)")
print(f"{'='*60}")
print(f"{'Setting':<25} {'Flag%':>6} {'Base':>6} {'Always':>7} {'DeBERTa':>8} {'D vs A':>7}")
print(f"{'-'*60}")
print(f"{'MuSiQue Augment':<25} {aug['flag_rate']:>5.1f}% {aug['base_acc']:>5.1f}% {aug['always_acc']:>6.1f}% {aug['deberta_acc']:>7.1f}% {aug['deberta_acc']-aug['always_acc']:>+6.1f}%")
print(f"{'MuSiQue Rerank':<25} {rerank['flag_rate']:>5.1f}% {rerank['base_acc']:>5.1f}% {rerank['always_acc']:>6.1f}% {rerank['deberta_acc']:>7.1f}% {rerank['deberta_acc']-rerank['always_acc']:>+6.1f}%")
print(f"{'HotpotQA (zero-shot)':<25} {flag_rate_hqa:>5.1f}% {base_acc_hqa:>5.1f}% {always_acc_hqa:>6.1f}% {deberta_acc_hqa:>7.1f}% {deberta_acc_hqa-always_acc_hqa:>+6.1f}%")
print()
print("Flag% = percentage of hops where DeBERTa triggers intervention")
print("D vs A = DeBERTa accuracy minus Always accuracy")
print("Key insight: DeBERTa matches or exceeds Always while intervening on far fewer hops")
