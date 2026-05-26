#!/usr/bin/env python3
"""
2WikiMultihopQA: LLM-flagged intervention (single pass).

For each hop:
1. Retrieve passages
2. LLM checks if passages contain the needed fact
3. If flagged: re-retrieve with reformulated query, add passages
4. Reason with accumulated passages

Uses the gold evidences from 2WikiMQA to know what fact each hop needs.
Baseline already exists from 2wiki_experiment.py (78.8%).

Run: python 2wiki_llm_intervention.py --n 50  (pilot)
     python 2wiki_llm_intervention.py          (full)
"""
import json
import sys
import time
import argparse
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, '.')
from simple_multihop_qa import SimpleMultiHopQA
from query_reformulator import QueryReformulator


def llm_fact_check(client, sub_question, passages_text, model="gpt-4.1-mini"):
    """Quick LLM check: does the passage contain enough to answer the sub-question?"""
    prompt = f"""Do the following passages contain enough information to answer this question?

Question: {sub_question}
Passages:
{passages_text[:2000]}

Answer YES or NO. One word only."""
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return "NO" in answer  # True = flagged (insufficient)
    except:
        return False  # Don't flag on error


def run_llm_intervention(qa, client, reformulator, question, gold_answer, max_hops=4):
    """Single-pass Self-Ask with LLM fact-checking at each hop."""
    steps = []
    all_passages = []
    previous_reasoning = []
    current_query = question
    flags = []
    
    for hop in range(max_hops):
        hop_num = hop + 1
        
        # Normal retrieval
        passages = qa.retrieve(current_query)
        all_passages.extend(passages)
        
        # Format passages for LLM check
        passages_text = "\n".join(
            f"[{i+1}] {p['title']}: {p['text'][:500]}"
            for i, p in enumerate(all_passages)
        )
        
        # LLM fact-grounding check
        is_flagged = llm_fact_check(client, current_query, passages_text)
        flags.append(is_flagged)
        
        # If flagged, reformulate query and re-retrieve
        if is_flagged:
            passages_for_reform = "\n".join(
                f"[{i+1}] {p['title']}: {p['text'][:300]}"
                for i, p in enumerate(passages)
            )
            new_query = reformulator.reformulate(question, current_query)
            extra = qa.retrieve(new_query)
            all_passages.extend(extra)
        
        # Dedupe
        seen_texts = set()
        unique_passages = []
        for p in all_passages:
            if p["text"] not in seen_texts:
                seen_texts.add(p["text"])
                unique_passages.append(p)
        all_passages = unique_passages
        
        # Reason
        context = qa.format_context(all_passages)
        reasoning, next_query, is_done = qa.reason_and_decide(
            question, context, previous_reasoning
        )
        
        previous_reasoning.append(reasoning)
        
        if is_done:
            if "ANSWER:" in reasoning:
                final_answer = reasoning.split("ANSWER:")[1].strip()
            else:
                final_answer = reasoning
            return final_answer, flags
        
        current_query = next_query
    
    # Max hops - force answer
    context = qa.format_context(all_passages)
    prompt = f"""Question: {question}
Information:
{context}
Based on all the information above, provide your best answer. Be concise.
Answer:"""
    response = qa.client.chat.completions.create(
        model=qa.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    final_answer = response.choices[0].message.content.strip()
    return final_answer, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--output', default='results/2wiki_llm_intervention.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    # Load data
    print("Loading 2WikiMultihopQA dev...")
    with open('raw_data/2wikimultihopqa/dev.json') as f:
        data = json.load(f)
    
    if args.n:
        data = data[:args.n]
    print(f"Processing {len(data)} examples")
    
    # Load existing baseline results for comparison
    print("Loading baseline results...")
    with open('results/2wiki_full.json') as f:
        baseline_data = json.load(f)
    baseline_lookup = {r['qid']: r for r in baseline_data['results']}
    
    # Init
    client = OpenAI()
    qa = SimpleMultiHopQA(corpus_name="2wikimultihopqa", model="gpt-4.1-mini")
    reformulator = QueryReformulator()
    
    results = []
    total_flags = 0
    total_hops = 0
    
    for ex in tqdm(data, desc="LLM-flagged intervention"):
        qid = ex['_id']
        question = ex['question']
        gold = ex['answer']
        
        try:
            pred, flags = run_llm_intervention(qa, client, reformulator, question, gold)
            total_flags += sum(flags)
            total_hops += len(flags)
        except Exception as e:
            if args.verbose:
                print(f"  ERROR {qid}: {e}")
            pred = ""
            flags = []
        
        results.append({
            'qid': qid,
            'question': question,
            'gold': gold,
            'llm_intervention': {
                'answer': pred,
                'flags': flags,
            },
            'baseline_answer': baseline_lookup.get(qid, {}).get('baseline', {}).get('answer', ''),
        })
        
        if args.verbose and (len(results)) % 100 == 0:
            print(f"  {len(results)}/{len(data)}, flags so far: {total_flags}/{total_hops} ({total_flags/max(total_hops,1)*100:.1f}%)")
    
    # Save
    output = {
        'n': len(results),
        'total_hops': total_hops,
        'total_flags': total_flags,
        'flag_rate': total_flags / max(total_hops, 1),
        'results': results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Quick stats (substring match)
    base_correct = 0
    llm_correct = 0
    for r in results:
        gold = r['gold'].lower().strip()
        bp = r['baseline_answer'].lower().strip()
        lp = r['llm_intervention']['answer'].lower().strip()
        if gold in bp or bp in gold:
            base_correct += 1
        if gold in lp or lp in gold:
            llm_correct += 1
    
    n = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS (n={n})")
    print(f"{'='*60}")
    print(f"Flag rate: {total_flags}/{total_hops} ({total_flags/max(total_hops,1)*100:.1f}%)")
    print(f"Baseline:         {base_correct}/{n} = {base_correct/n*100:.1f}%")
    print(f"LLM-intervention: {llm_correct}/{n} = {llm_correct/n*100:.1f}%")
    print(f"Delta:            {(llm_correct-base_correct)/n*100:+.1f}%")
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
