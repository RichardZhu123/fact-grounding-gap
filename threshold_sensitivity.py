"""
DeBERTa threshold sensitivity analysis.
Sweeps thresholds [0.3, 0.4, 0.5, 0.6, 0.7] and shows accuracy vs flag rate.
Uses existing DeBERTa predictions on dev set — no new model runs needed.

Run on VM: cd ~/ircot && python threshold_sensitivity.py
"""
import json
import numpy as np
from collections import defaultdict

# Load the augmentation results (has per-hop DeBERTa predictions)
with open('results/section4_full_dev.json') as f:
    aug = json.load(f)

examples = aug['examples']
n = len(examples)
total_hops = sum(ex['n_hops'] for ex in examples)

# We need raw DeBERTa probabilities. Check what's available.
ex0 = examples[0]
has_probs = 'deberta_probs' in ex0 or 'deberta' in ex0 and 'probs' in ex0.get('deberta', {})

if has_probs:
    print("Found DeBERTa probabilities in results file.")
else:
    # Try loading from a separate predictions file
    import os
    pred_paths = [
        'results/deberta_dev_predictions.json',
        'results/deberta_predictions_dev.json',
        'models/deberta_fg_v2_best/dev_predictions.json',
    ]
    pred_data = None
    for p in pred_paths:
        if os.path.exists(p):
            print(f"Loading predictions from {p}")
            with open(p) as f:
                pred_data = json.load(f)
            break
    
    if pred_data is None:
        # Generate predictions from the model directly
        print("No cached predictions found. Running DeBERTa inference on dev set...")
        print("This requires the model and dev data. Attempting...")
        
        # Try to load fact_grounded_final_dev.jsonl and run inference
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_path = 'models/deberta_fg_v2_best'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        dev_examples = []
        with open('results/fact_grounded_final_dev.jsonl') as f:
            for line in f:
                dev_examples.append(json.loads(line))
        
        print(f"Running inference on {len(dev_examples)} examples...")
        
        # Build a lookup: (sub_question) -> probability of ANSWERABLE
        probs_lookup = {}
        batch_size = 32
        
        for i in range(0, len(dev_examples), batch_size):
            batch = dev_examples[i:i+batch_size]
            texts = []
            for ex in batch:
                sq = ex['sub_question']
                ctx = ex.get('passage_text_combined', '')
                texts.append(f"{sq} [SEP] {ctx[:1500]}")
            
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
            
            for j, ex in enumerate(batch):
                # Assuming label 1 = ANSWERABLE
                prob_answerable = probs[j][1].item()
                key = ex['sub_question']
                probs_lookup[key] = prob_answerable
            
            if (i // batch_size) % 20 == 0:
                print(f"  Processed {i+len(batch)}/{len(dev_examples)}")
        
        print(f"Generated {len(probs_lookup)} predictions")
        
        # Save for future use
        with open('results/deberta_dev_probs.json', 'w') as f:
            json.dump(probs_lookup, f)
        print("Saved to results/deberta_dev_probs.json")
        
        pred_data = probs_lookup

# Now do the threshold sweep
# We need per-hop probabilities matched to the intervention examples
# If we have a probs_lookup keyed by sub_question, match against the experiment examples

print()
print("=" * 60)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)

# Try to get per-question, per-hop probabilities
# The section4_full_dev.json should have sub_questions per hop
# Let's check structure
if 'sub_questions' in ex0:
    has_subq = True
elif 'hops' in ex0:
    has_subq = True
else:
    has_subq = False

print(f"Example keys: {list(ex0.keys())}")
if 'deberta' in ex0:
    print(f"DeBERTa keys: {list(ex0['deberta'].keys())}")

# If we have deberta scores/probs in the experiment file
deberta_info = ex0.get('deberta', {})
if 'scores' in deberta_info or 'probs' in deberta_info:
    score_key = 'scores' if 'scores' in deberta_info else 'probs'
    print(f"Found per-hop scores in experiment file under 'deberta.{score_key}'")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print()
    print(f"{'Thresh':>7} | {'Flag%':>6} | {'Flags/Q':>7} | {'Accuracy':>9} | {'Δ vs Base':>9} | {'Δ vs 0.5':>9}")
    print("-" * 65)
    
    base_correct = sum(1 for ex in examples if ex['baseline']['correct'])
    base_acc = base_correct / n * 100
    
    # Get accuracy at 0.5 for comparison
    acc_at_05 = sum(1 for ex in examples if ex['deberta']['correct']) / n * 100
    
    results = []
    for thresh in thresholds:
        total_flags = 0
        correct = 0
        
        for ex in examples:
            scores = ex['deberta'][score_key]
            flags = [1 if s >= thresh else 0 for s in scores]
            total_flags += sum(flags)
            
            # For accuracy, we need to know what would happen with these flags
            # At threshold 0.5, the stored 'correct' applies
            # For other thresholds, we'd need to re-run the intervention
            # BUT we can report flag rate and use the stored result as approximation
            # Actually, we can only exactly compute flag rates; accuracy requires re-running
            
        flag_rate = total_flags / total_hops * 100
        flags_per_q = total_flags / n
        
        # For accuracy at non-0.5 thresholds, we can't compute exactly without re-running
        # But we CAN compute it if the file stores per-hop intervention results
        # For now, report flag rates and note that accuracy at 0.5 is the validated number
        
        results.append({
            'thresh': thresh,
            'flag_rate': flag_rate,
            'flags_per_q': flags_per_q,
            'total_flags': total_flags,
        })
        
        acc_str = f"{acc_at_05:.1f}%" if thresh == 0.5 else "—"
        d_base = f"+{acc_at_05 - base_acc:.1f}%" if thresh == 0.5 else "—"
        
        print(f"  {thresh:.1f}  | {flag_rate:>5.1f}% | {flags_per_q:>6.2f} | {acc_str:>9} | {d_base:>9} | {'  (ref)' if thresh == 0.5 else '—':>9}")
    
    print()
    print("NOTE: Accuracy is only validated at threshold=0.5 (the setting used in experiments).")
    print("Flag rates show the cost-accuracy trade-off curve:")
    print("  Lower threshold → more flags → more interventions → higher cost")
    print("  Higher threshold → fewer flags → fewer interventions → lower cost")
    
else:
    # No per-hop scores, try the lookup approach
    print("No per-hop scores found in experiment file.")
    print(f"Available deberta keys: {list(deberta_info.keys())}")
    print()
    print("To run threshold sensitivity, we need DeBERTa probability scores per hop.")
    print("Option 1: Re-run DeBERTa inference saving probabilities")
    print("Option 2: Add --save-scores flag to your experiment script")
    
    # Still useful: show the flag distribution at 0.5
    flags_per_q = defaultdict(int)
    for ex in examples:
        nf = sum(ex['deberta']['flags'])
        flags_per_q[nf] += 1
    
    print()
    print("FLAG DISTRIBUTION AT THRESHOLD=0.5:")
    for k in sorted(flags_per_q):
        print(f"  {k} flags: {flags_per_q[k]} questions ({flags_per_q[k]/n*100:.1f}%)")
