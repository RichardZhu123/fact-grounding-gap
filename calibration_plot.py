"""
DeBERTa calibration + threshold sensitivity analysis.
Memory-optimized for small VMs (4GB RAM).

Run on VM: cd ~/ircot && python calibration_plot.py
"""
import json
import os
import gc
import numpy as np

probs_file = 'results/deberta_dev_probs.json'

if os.path.exists(probs_file):
    print(f"Loading cached probabilities from {probs_file}")
    with open(probs_file) as f:
        probs_list = json.load(f)
else:
    print("No cached probabilities. Running DeBERTa inference (memory-optimized)...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_path = 'models/deberta_fg_v2_best'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    model = model.float().cpu()
    
    dev_examples = []
    with open('results/fact_grounded_final_dev.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('label') not in ('ANSWERABLE', 'NOT-ANSWERABLE'):
                continue
            dev_examples.append(ex)
    
    print(f"Running inference on {len(dev_examples)} examples...")
    
    probs_list = []
    batch_size = 1
    
    for i in range(0, len(dev_examples), batch_size):
        batch = dev_examples[i:i+batch_size]
        texts = []
        labels = []
        for ex in batch:
            sq = ex['sub_question']
            ctx = ex.get('passage_text_combined', '')
            texts.append(f"{sq} [SEP] {ctx[:512]}")
            labels.append(1 if ex['label'] == 'ANSWERABLE' else 0)
        
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        for j in range(len(batch)):
            prob_ans = probs[j][1].item()
            probs_list.append([prob_ans, labels[j]])
        
        del inputs, logits, probs
        gc.collect()
        
        if (i + batch_size) % 500 == 0 or i == 0:
            print(f"  {i+len(batch)}/{len(dev_examples)}")
    
    with open(probs_file, 'w') as f:
        json.dump(probs_list, f)
    print(f"Saved {len(probs_list)} predictions to {probs_file}")
    
    del model, tokenizer
    gc.collect()

# ----- Analysis -----
print()
print("=" * 60)
print("CALIBRATION ANALYSIS (Reliability Diagram)")
print("=" * 60)

probs_arr = np.array([p[0] for p in probs_list])
labels_arr = np.array([p[1] for p in probs_list])

n_total = len(probs_list)
print(f"n = {n_total}")
print(f"Positive rate (ANSWERABLE): {labels_arr.mean():.3f}")
print()

n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)

print(f"{'Bin':>12} | {'n':>6} | {'Mean Pred':>9} | {'True Pos%':>9} | {'|Gap|':>6}")
print("-" * 55)

ece = 0
total_in_bins = 0

for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i+1]
    if i == n_bins - 1:
        mask = (probs_arr >= lo) & (probs_arr <= hi)
    else:
        mask = (probs_arr >= lo) & (probs_arr < hi)
    
    n_in_bin = mask.sum()
    if n_in_bin == 0:
        print(f"[{lo:.1f}, {hi:.1f}) | {0:>6} |       --- |       --- |     ---")
        continue
    
    mean_pred = probs_arr[mask].mean()
    true_pos = labels_arr[mask].mean()
    gap = abs(mean_pred - true_pos)
    ece += gap * n_in_bin
    total_in_bins += n_in_bin
    
    print(f"[{lo:.1f}, {hi:.1f}) | {n_in_bin:>6} | {mean_pred:>8.3f} | {true_pos:>8.3f} | {gap:>5.3f}")

ece /= total_in_bins
print()
print(f"Expected Calibration Error (ECE): {ece:.4f}")

if ece < 0.05:
    print("Interpretation: WELL CALIBRATED (ECE < 0.05)")
elif ece < 0.10:
    print("Interpretation: REASONABLY CALIBRATED (ECE < 0.10)")
elif ece < 0.15:
    print("Interpretation: MODERATELY CALIBRATED (ECE < 0.15)")
else:
    print(f"Interpretation: POORLY CALIBRATED (ECE = {ece:.3f})")

print()
print("=" * 60)
print("THRESHOLD SENSITIVITY (DeBERTa predictor)")
print("=" * 60)
print(f"{'Thresh':>7} | {'Pred+':>6} | {'Pred-':>6} | {'Flag%':>6} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
print("-" * 65)

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (probs_arr >= thresh).astype(int)
    tp = ((preds == 1) & (labels_arr == 1)).sum()
    tn = ((preds == 0) & (labels_arr == 0)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()
    
    acc = (tp + tn) / n_total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    n_pos = (preds == 1).sum()
    n_neg = (preds == 0).sum()
    flag_pct = n_pos / n_total * 100
    
    marker = " <-- operating point" if thresh == 0.5 else ""
    print(f"  {thresh:.1f}  | {n_pos:>6} | {n_neg:>6} | {flag_pct:>5.1f}% | {acc:>5.3f} | {prec:>5.3f} | {rec:>5.3f} | {f1:>5.3f}{marker}")

print()
print("Note: thresh=0.5 is the operating point used in all experiments.")
