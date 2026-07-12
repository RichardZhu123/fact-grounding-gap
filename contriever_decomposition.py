#!/usr/bin/env python3
"""Compute three-way decomposition for Contriever vs BM25."""
import json

def check_gold_in_passages(gold_para, passages):
    """Check if gold paragraph text appears in retrieved passages."""
    if not gold_para:
        return False
    gold_text = gold_para['paragraph_text'].strip().lower()
    for p in passages:
        p_text = p.get('text', p.get('paragraph_text', '')).strip().lower()
        # Check substantial overlap
        if gold_text[:200] in p_text or p_text[:200] in gold_text:
            return True
    return False

def decompose(base_run_path, raw_data_path, fg_labels_path):
    # Load base run trajectories
    with open(base_run_path) as f:
        base_run = json.load(f)
    
    # Load raw MuSiQue data
    raw = {}
    with open(raw_data_path) as f:
        for line in f:
            d = json.loads(line)
            raw[d['id']] = d
    
    # Load fact-grounding labels
    fg = {}
    with open(fg_labels_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d['qid'], d['hop_number'])
            fg[key] = d['label_binary']
    
    retrieval_fail = 0
    extraction_fail = 0
    fact_present = 0
    total = 0
    
    for result in base_run['results']:
        qid = result['qid']
        if qid not in raw:
            continue
        
        musique = raw[qid]
        paragraphs = musique['paragraphs']
        decomposition = musique['question_decomposition']
        steps = result['trajectory']['steps']
        
        # Build accumulated passages per hop
        accumulated = []
        for step_idx, step in enumerate(steps):
            accumulated.extend(step['retrieved_passages'])
            
            if step_idx >= len(decomposition):
                continue
            
            subq = decomposition[step_idx]
            gold_idx = subq.get('paragraph_support_idx')
            
            if gold_idx is None or gold_idx >= len(paragraphs):
                continue
            
            gold_para = paragraphs[gold_idx]
            hop_num = step_idx + 1
            
            # Check gold paragraph in accumulated passages
            gold_present = check_gold_in_passages(gold_para, accumulated)
            
            # Get fact-grounding label
            label = fg.get((qid, hop_num))
            if label is None:
                continue
            
            total += 1
            if label == 1:
                fact_present += 1
            elif gold_present:
                extraction_fail += 1
            else:
                retrieval_fail += 1
    
    print(f"Total hops: {total}")
    print(f"Retrieval failure: {retrieval_fail} ({retrieval_fail/total*100:.1f}%)")
    print(f"Extraction failure: {extraction_fail} ({extraction_fail/total*100:.1f}%)")
    print(f"Fact present: {fact_present} ({fact_present/total*100:.1f}%)")
    
    return {'total': total, 'retrieval_failure': retrieval_fail,
            'extraction_failure': extraction_fail, 'fact_present': fact_present}

print("=== BM25 ===")
decompose('results/full_run_all_4.1.json', 
          'raw_data/musique/musique_ans_v1.0_dev.jsonl',
          'results/fact_grounded_final_dev.jsonl')

print("\n=== CONTRIEVER ===")
decompose('results/contriever_base_run.json',
          'raw_data/musique/musique_ans_v1.0_dev.jsonl', 
          'results/contriever_fact_grounded.jsonl')
