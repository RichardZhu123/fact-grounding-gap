#!/usr/bin/env python3
"""
Run Self-Ask baseline + DeBERTa intervention on 2WikiMultihopQA dev.
Mirrors the MuSiQue and HotpotQA experiment pipelines.

Run on VM: cd ~/ircot && python 2wiki_experiment.py --n 50  (pilot)
           cd ~/ircot && python 2wiki_experiment.py          (full)
"""
import json
import sys
import time
import argparse
import numpy as np
from collections import defaultdict
from dataclasses import asdict
from tqdm import tqdm

sys.path.insert(0, '.')
from simple_multihop_qa import SimpleMultiHopQA
from deberta_inference import DebertaPredictor


def load_2wiki_dev(path="raw_data/2wikimultihopqa/dev.json", n=None):
    """Load 2WikiMultihopQA dev examples."""
    with open(path) as f:
        data = json.load(f)
    
    examples = []
    for ex in data:
        # Get sub-questions from evidences (KB triples)
        # evidences format: [['entity1', 'relation', 'entity2'], ...]
        evidences = ex.get('evidences', [])
        n_hops = len(evidences) if evidences else 2  # default 2-hop
        
        examples.append({
            'id': ex['_id'],
            'question': ex['question'],
            'answer': ex['answer'],
            'type': ex.get('type', ''),
            'n_hops': n_hops,
            'supporting_facts': ex.get('supporting_facts', []),
            'evidences': evidences,
        })
    
    if n:
        examples = examples[:n]
    
    return examples


def run_experiment(examples, qa_base, qa_always, predictor, verbose=False):
    """Run all 4 conditions on the examples."""
    results = []
    
    for i, ex in enumerate(tqdm(examples, desc="Processing")):
        qid = ex['id']
        question = ex['question']
        gold = ex['answer']
        n_hops = ex['n_hops']
        
        # Condition 1: Baseline (no intervention)
        try:
            base_traj = qa_base.answer(question, gold_answer=gold)
            base_pred = base_traj.final_answer
        except Exception as e:
            if verbose:
                print(f"  ERROR base {qid}: {e}")
            base_pred = ""
        
        # Condition 2: Always intervene
        try:
            # Build interventions for every hop
            always_interventions = {}
            # Run baseline first to get queries, then re-retrieve at each hop
            temp_traj = qa_base.answer(question)
            for step in temp_traj.steps:
                extra = qa_base.retrieve(step.query)
                always_interventions[step.hop_number] = extra
            
            always_traj = qa_base.answer_with_interventions(
                question, interventions=always_interventions, gold_answer=gold
            )
            always_pred = always_traj.final_answer
        except Exception as e:
            if verbose:
                print(f"  ERROR always {qid}: {e}")
            always_pred = ""
        
        # Condition 3: DeBERTa-targeted intervention
        try:
            # First pass: get trajectory
            deberta_traj = qa_base.answer(question)
            
            # Check each hop with DeBERTa
            deberta_interventions = {}
            deberta_flags = []
            for step in deberta_traj.steps:
                passage_text = ' '.join(
                    p.get('text', p.get('paragraph_text', ''))
                    for p in step.retrieved_passages
                )
                is_sufficient = predictor.predict(step.query, passage_text)
                flag = not is_sufficient
                deberta_flags.append(flag)
                
                if flag:
                    extra = qa_base.retrieve(step.query)
                    deberta_interventions[step.hop_number] = extra
            
            # Second pass with interventions if any
            if deberta_interventions:
                deberta_traj = qa_base.answer_with_interventions(
                    question, interventions=deberta_interventions, gold_answer=gold
                )
            
            deberta_pred = deberta_traj.final_answer
        except Exception as e:
            if verbose:
                print(f"  ERROR deberta {qid}: {e}")
            deberta_pred = ""
            deberta_flags = []
        
        # Store result (correctness will be evaluated by LLM judge later)
        results.append({
            'qid': qid,
            'question': question,
            'gold': gold,
            'n_hops': n_hops,
            'type': ex.get('type', ''),
            'baseline': {
                'answer': base_pred,
            },
            'always': {
                'answer': always_pred,
            },
            'deberta': {
                'answer': deberta_pred,
                'flags': deberta_flags,
            },
        })
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(examples)} done")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None, help='Number of examples (None=all)')
    parser.add_argument('--output', default='results/2wiki_experiment.json')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip-deberta', action='store_true', help='Skip DeBERTa conditions (faster)')
    args = parser.parse_args()
    
    print("Loading 2WikiMultihopQA dev data...")
    examples = load_2wiki_dev(n=args.n)
    print(f"Loaded {len(examples)} examples")
    
    # Stats
    type_counts = defaultdict(int)
    hop_counts = defaultdict(int)
    for ex in examples:
        type_counts[ex['type']] += 1
        hop_counts[ex['n_hops']] += 1
    print(f"Types: {dict(type_counts)}")
    print(f"Hops: {dict(hop_counts)}")
    
    print("\nInitializing QA engine...")
    qa = SimpleMultiHopQA(
        corpus_name="2wikimultihopqa",
        model="gpt-4.1-mini",
        max_hops=4,
    )
    
    predictor = None
    if not args.skip_deberta:
        print("Loading DeBERTa predictor...")
        predictor = DebertaPredictor('models/deberta_fg_v2_best')
    
    print(f"\nRunning experiment on {len(examples)} examples...")
    t0 = time.time()
    
    results = run_experiment(
        examples, qa, qa, predictor, verbose=args.verbose
    )
    
    elapsed = time.time() - t0
    
    # Save
    output = {
        'dataset': '2wikimultihopqa',
        'n': len(results),
        'elapsed_seconds': elapsed,
        'results': results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(results)} results to {args.output}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    # Quick stats (substring match, not LLM judge)
    for cond in ['baseline', 'always', 'deberta']:
        correct = 0
        for r in results:
            pred = r[cond]['answer'].lower().strip()
            gold = r['gold'].lower().strip()
            if gold in pred or pred in gold:
                correct += 1
        print(f"  {cond}: {correct}/{len(results)} = {correct/len(results)*100:.1f}% (substring match)")


if __name__ == '__main__':
    main()
