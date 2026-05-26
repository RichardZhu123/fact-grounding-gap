"""
Evaluate fine-tuned fact-aware GPT-4.1-mini against base model.
4 conditions:
1. Base GPT-4.1-mini (no intervention)
2. Base + DeBERTa intervention
3. Fine-tuned GPT-4.1-mini (no intervention)
4. Fine-tuned + DeBERTa intervention

Run on VM: cd ~/ircot && python sft_evaluate.py --n 50 --verbose
"""
import json
import sys
import argparse
import time
import numpy as np
from collections import defaultdict

# Add project to path
sys.path.insert(0, '.')
from simple_multihop_qa import MultiHopQA
from deberta_inference import DeBERTaPredictor

BASE_MODEL = "gpt-4.1-mini"
FT_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:fact-aware:DdK9vXB1"

def run_condition(qa_engine, predictor, examples, use_deberta=False, verbose=False):
    """Run one experimental condition."""
    results = []
    correct = 0
    
    for i, ex in enumerate(examples):
        qid = ex['id']
        question = ex['question']
        gold = ex['answer']
        
        try:
            # Build interventions dict if using DeBERTa
            interventions = {}
            if use_deberta and predictor:
                # Run Self-Ask to get sub-questions and passages per hop,
                # then use DeBERTa to decide which hops need intervention.
                # For simplicity, run without intervention first to get trajectory,
                # then re-run with interventions on flagged hops.
                
                # First pass: get trajectory without intervention
                traj = qa_engine.answer_with_interventions(question)
                
                # Check each hop with DeBERTa
                for step in traj.steps:
                    hop_num = step.hop_number
                    sub_q = step.query
                    passages = step.retrieved_passages
                    
                    if sub_q and passages:
                        passage_text = ' '.join(p.get('text', p.get('paragraph_text', '')) 
                                               for p in passages)
                        is_sufficient = predictor.predict(sub_q, passage_text)
                        
                        if not is_sufficient:
                            # Re-retrieve with reformulated query
                            extra = qa_engine.retrieve(sub_q)
                            interventions[hop_num] = extra
                
                # Second pass: run with interventions
                if interventions:
                    traj = qa_engine.answer_with_interventions(question, interventions=interventions)
            else:
                traj = qa_engine.answer_with_interventions(question)
            
            pred = traj.final_answer
            
            # Flexible matching (same as your existing experiments)
            pred_clean = pred.lower().strip()
            gold_clean = gold.lower().strip()
            is_correct = (gold_clean in pred_clean) or (pred_clean in gold_clean)
            
            if is_correct:
                correct += 1
            
            results.append({
                'qid': qid,
                'gold': gold,
                'pred': pred,
                'correct': is_correct,
                'n_interventions': len(interventions),
            })
            
            if verbose and not is_correct:
                print(f"  WRONG {qid}: gold='{gold}' pred='{pred[:80]}'")
                
        except Exception as e:
            if verbose:
                print(f"  ERROR {qid}: {e}")
            results.append({
                'qid': qid,
                'gold': gold,
                'pred': '',
                'correct': False,
                'error': str(e),
            })
        
        if (i + 1) % 50 == 0 or (i + 1) == len(examples):
            acc = correct / (i + 1) * 100
            print(f"  {i+1}/{len(examples)}: {acc:.1f}%")
    
    return results, correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None, help='Number of examples (None=all)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--conditions', nargs='+',
                        default=['base', 'base_deberta', 'ft', 'ft_deberta'])
    parser.add_argument('--output', default='results/sft_evaluation.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    # Load dev data
    print("Loading data...")
    examples = []
    with open('raw_data/musique/musique_ans_v1.0_dev.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('answerable', True):
                examples.append(ex)
    
    examples = examples[args.start:args.start + args.n] if args.n else examples
    print(f"Evaluating {len(examples)} examples")
    
    # Load DeBERTa if needed
    predictor = None
    if any('deberta' in c for c in args.conditions):
        print("Loading DeBERTa predictor...")
        predictor = DeBERTaPredictor('models/deberta_fg_v2_best')
    
    # Initialize QA engines
    engines = {}
    if any(c.startswith('base') for c in args.conditions):
        print(f"Initializing base engine: {BASE_MODEL}")
        engines['base'] = MultiHopQA(model=BASE_MODEL)
    if any(c.startswith('ft') for c in args.conditions):
        print(f"Initializing fine-tuned engine: {FT_MODEL}")
        engines['ft'] = MultiHopQA(model=FT_MODEL)
    
    # Run conditions
    all_results = {
        'n': len(examples),
        'base_model': BASE_MODEL,
        'ft_model': FT_MODEL,
        'conditions': {}
    }
    
    configs = {
        'base':         {'engine': 'base', 'deberta': False, 'label': 'Base GPT-4.1-mini'},
        'base_deberta': {'engine': 'base', 'deberta': True,  'label': 'Base + DeBERTa'},
        'ft':           {'engine': 'ft',   'deberta': False, 'label': 'Fine-tuned GPT-4.1-mini'},
        'ft_deberta':   {'engine': 'ft',   'deberta': True,  'label': 'Fine-tuned + DeBERTa'},
    }
    
    for cond in args.conditions:
        if cond not in configs:
            print(f"Unknown condition: {cond}")
            continue
        
        cfg = configs[cond]
        engine = engines[cfg['engine']]
        
        print(f"\n{'='*60}")
        print(f"CONDITION: {cfg['label']}")
        print(f"  DeBERTa: {cfg['deberta']}")
        print(f"{'='*60}")
        
        t0 = time.time()
        results, correct = run_condition(
            engine, predictor, examples,
            use_deberta=cfg['deberta'],
            verbose=args.verbose
        )
        elapsed = time.time() - t0
        
        acc = correct / len(examples) * 100
        print(f"\n  RESULT: {correct}/{len(examples)} = {acc:.1f}% ({elapsed:.0f}s)")
        
        all_results['conditions'][cond] = {
            'label': cfg['label'],
            'correct': correct,
            'accuracy': acc,
            'elapsed': elapsed,
            'results': results,
        }
        
        # Save after each condition
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (n={len(examples)})")
    print(f"{'='*60}")
    print(f"{'Condition':<30} {'Accuracy':>9} {'Correct':>8}")
    print("-" * 50)
    for cond in args.conditions:
        if cond in all_results['conditions']:
            c = all_results['conditions'][cond]
            print(f"{c['label']:<30} {c['accuracy']:>8.1f}% {c['correct']:>7}/{len(examples)}")
    
    print(f"\nReference (section4_full_dev.json, n=2416):")
    print(f"  Baseline:        51.9%")
    print(f"  Always:          54.6%")
    print(f"  DeBERTa t=0.5:   55.6%")

if __name__ == '__main__':
    main()
