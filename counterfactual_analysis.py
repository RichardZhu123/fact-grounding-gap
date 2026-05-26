#!/usr/bin/env python3
"""
Counterfactual Hop Attribution Script

For each failed multi-hop QA example, at each hop k:
  - Replace retrieved passages with ALL gold supporting passages
  - Re-run the LLM reasoning from that point
  - Check if the final answer becomes correct

Produces:
  1. Per-hop causal responsibility scores (for paper Section 2)
  2. SUFFICIENT/INSUFFICIENT labels per step (training data for Section 3)
  3. Retrieval recall vs sufficiency correlation data (for paper Section 2)
"""

import json
import re
import os
import argparse
from tqdm import tqdm
from simple_multihop_qa import SimpleMultiHopQA


def normalize(s):
    """Normalize answer string for comparison."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def answer_is_correct(pred, gold):
    """Check if prediction contains the gold answer."""
    return normalize(gold) in normalize(pred)


def get_gold_passages(example):
    """Get all gold supporting passages from a MuSiQue example."""
    gold = []
    for ctx in example.get('contexts', []):
        if ctx.get('is_supporting', False):
            gold.append({
                'title': ctx.get('title', ''),
                'text': ctx.get('paragraph_text', ''),
                'score': 99.0
            })
    return gold


def get_retrieval_recall(retrieved_passages, gold_passages):
    """What fraction of gold passage titles were retrieved?"""
    if not gold_passages:
        return 0.0
    gold_titles = set(g['title'].lower().strip() for g in gold_passages)
    retrieved_titles = set(p['title'].lower().strip() for p in retrieved_passages)
    if not gold_titles:
        return 0.0
    return len(gold_titles & retrieved_titles) / len(gold_titles)


def run_with_gold_at_hop(qa, question, original_trajectory, gold_passages, target_hop):
    """
    Re-run QA but at target_hop, use gold passages instead of retrieved ones.
    At all other hops, use the original retrieved passages.
    
    This answers: "If retrieval had been perfect at hop k, would the answer be correct?"
    """
    all_passages = []
    previous_reasoning = []
    original_steps = original_trajectory['steps']
    num_hops = len(original_steps)
    
    for hop in range(num_hops):
        # Get passages for this hop
        if hop == target_hop:
            # USE GOLD PASSAGES at this hop
            passages = gold_passages
        else:
            # Use original retrieved passages at other hops
            passages = original_steps[hop].get('retrieved_passages', [])
        
        # Accumulate passages (same as the original system does)
        all_passages.extend(passages)
        seen = set()
        all_passages = [p for p in all_passages if p.get('text', '') not in seen and not seen.add(p.get('text', ''))]
        
        # Format context and ask LLM to reason
        context = qa.format_context(all_passages)
        reasoning, next_query, is_done = qa.reason_and_decide(
            question, context, previous_reasoning
        )
        previous_reasoning.append(reasoning)
        
        if is_done:
            if "ANSWER:" in reasoning:
                return reasoning.split("ANSWER:")[1].strip()
            return reasoning
    
    # Force answer at end
    context = qa.format_context(all_passages)
    response = qa.client.chat.completions.create(
        model=qa.model,
        messages=[{"role": "user", "content": f"Question: {question}\n\nInfo:\n{context}\n\nAnswer concisely:"}],
        temperature=0, max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results JSON from run_multihop_eval.py")
    parser.add_argument("--data", required=True, help="MuSiQue data file (jsonl)")
    parser.add_argument("--output", default="results/counterfactual.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    # Load results
    with open(args.results) as f:
        results_data = json.load(f)
    
    # Load MuSiQue data (for gold passages)
    musique = {}
    with open(args.data) as f:
        for line in f:
            item = json.loads(line)
            musique[item.get('question_id', '')] = item
    
    # Initialize QA system
    qa = SimpleMultiHopQA()
    
    # Get genuine failures only
    failures = []
    for r in results_data['results']:
        is_correct = r.get('em', 0) == 1.0 or answer_is_correct(r['predicted_answer'], r['gold_answer'])
        if not is_correct:
            failures.append(r)
    
    if args.limit:
        failures = failures[:args.limit]
    
    print(f"Processing {len(failures)} genuine failures")
    
    # Run counterfactual analysis
    all_results = []
    
    # Counters for summary
    hop_fixed = {}   # hop_num -> count of times gold swap fixed the answer
    hop_total = {}   # hop_num -> total interventions
    correlation_data = []  # (retrieval_recall, was_sufficient) pairs
    
    for r in tqdm(failures, desc="Counterfactual"):
        qid = r.get('qid', '')
        question = r['question']
        gold_answer = r['gold_answer']
        trajectory = r['trajectory']
        num_hops = len(trajectory['steps'])
        
        # Get gold passages from MuSiQue
        musique_item = musique.get(qid, {})
        gold_passages = get_gold_passages(musique_item)
        
        if not gold_passages:
            continue
        
        # For each hop, do the counterfactual swap
        hop_results = []
        for hop_k in range(num_hops):
            # Run QA with gold passages at hop k
            try:
                new_answer = run_with_gold_at_hop(
                    qa, question, trajectory, gold_passages, hop_k
                )
                fixed = answer_is_correct(new_answer, gold_answer)
            except Exception as e:
                new_answer = f"ERROR: {e}"
                fixed = False
            
            # Compute retrieval recall at this hop
            retrieved = trajectory['steps'][hop_k].get('retrieved_passages', [])
            recall = get_retrieval_recall(retrieved, gold_passages)
            
            # Label
            label = "INSUFFICIENT" if fixed else "SUFFICIENT"
            
            # Track counts
            h = hop_k + 1
            hop_total[h] = hop_total.get(h, 0) + 1
            if fixed:
                hop_fixed[h] = hop_fixed.get(h, 0) + 1
            
            # Track correlation data
            correlation_data.append({
                'retrieval_recall': recall,
                'sufficient': not fixed,
                'hop': h
            })
            
            hop_results.append({
                'hop': h,
                'label': label,
                'retrieval_recall': recall,
                'gold_swap_fixed': fixed,
                'new_answer': new_answer[:150],
                # Training data fields
                'sub_question': trajectory['steps'][hop_k].get('query', ''),
                'retrieved_passages': retrieved,
                'gold_passages_used': fixed
            })
        
        all_results.append({
            'qid': qid,
            'question': question,
            'gold': gold_answer,
            'original_pred': r['predicted_answer'][:150],
            'num_hops': num_hops,
            'hop_results': hop_results
        })
    
    # === PRINT SUMMARY ===
    print("\n" + "="*60)
    print("COUNTERFACTUAL ATTRIBUTION RESULTS")
    print("="*60)
    print(f"Total failures analyzed: {len(all_results)}")
    
    # Per-hop causal responsibility
    print("\nPer-hop causal responsibility:")
    print("(% of failures fixed by gold swap at this hop)")
    for h in sorted(hop_total.keys()):
        total = hop_total[h]
        fixed = hop_fixed.get(h, 0)
        pct = fixed / total * 100 if total > 0 else 0
        print(f"  Hop {h}: {fixed}/{total} fixed ({pct:.1f}%)")
    
    # Retrieval recall vs sufficiency
    suf_recalls = [d['retrieval_recall'] for d in correlation_data if d['sufficient']]
    insuf_recalls = [d['retrieval_recall'] for d in correlation_data if not d['sufficient']]
    
    print(f"\nRetrieval recall comparison:")
    if suf_recalls:
        print(f"  SUFFICIENT steps avg recall:   {sum(suf_recalls)/len(suf_recalls):.3f} (n={len(suf_recalls)})")
    if insuf_recalls:
        print(f"  INSUFFICIENT steps avg recall: {sum(insuf_recalls)/len(insuf_recalls):.3f} (n={len(insuf_recalls)})")
    
    if suf_recalls and insuf_recalls:
        diff = abs(sum(suf_recalls)/len(suf_recalls) - sum(insuf_recalls)/len(insuf_recalls))
        if diff < 0.15:
            print(f"  >>> GAP IS SMALL ({diff:.3f}) — retrieval recall does NOT predict sufficiency well!")
            print(f"  >>> This supports your paper's thesis.")
        else:
            print(f"  >>> GAP IS LARGE ({diff:.3f}) — retrieval recall does predict sufficiency.")
            print(f"  >>> Paper thesis is weaker.")
    
    # Label distribution
    all_labels = [hr['label'] for r in all_results for hr in r['hop_results']]
    suf = all_labels.count('SUFFICIENT')
    insuf = all_labels.count('INSUFFICIENT')
    print(f"\nLabel distribution (training data):")
    print(f"  SUFFICIENT:   {suf} ({suf/len(all_labels)*100:.1f}%)")
    print(f"  INSUFFICIENT: {insuf} ({insuf/len(all_labels)*100:.1f}%)")
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    output = {
        'summary': {
            'total_failures': len(all_results),
            'hop_fixed': {str(k): v for k, v in hop_fixed.items()},
            'hop_total': {str(k): v for k, v in hop_total.items()},
            'labels': {'SUFFICIENT': suf, 'INSUFFICIENT': insuf}
        },
        'correlation_data': correlation_data,
        'results': all_results
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
