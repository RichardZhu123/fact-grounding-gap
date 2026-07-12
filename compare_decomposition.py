#!/usr/bin/env python3
"""Compare three-way decomposition: BM25 vs Contriever using text-content matching."""
import json

def load_trajectories(path):
    with open(path) as f:
        return json.load(f)

def load_fg_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels[(d['qid'], d['hop_number'])] = d['label_binary']
    return labels

def check_gold_in_passages(gold_text, passages):
    gold_clean = gold_text.strip().lower()[:300]
    for p in passages:
        p_text = p.get('text', p.get('paragraph_text', '')).strip().lower()
        if gold_clean in p_text or p_text[:300] in gold_clean:
            return True
    return False

def decompose(traj_path, fg_path, raw_path, label):
    traj = load_trajectories(traj_path)
    fg = load_fg_labels(fg_path)
    
    raw = {}
    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            raw[d['id']] = d
    
    ret_fail = 0
    ext_fail = 0
    fact_present = 0
    total = 0
    
    for result in traj['results']:
        qid = result['qid']
        if qid not in raw:
            continue
        musique = raw[qid]
        paragraphs = musique['paragraphs']
        decomp = musique['question_decomposition']
        steps = result['trajectory']['steps']
        
        accumulated = []
        for step_idx, step in enumerate(steps):
            accumulated.extend(step['retrieved_passages'])
            
            if step_idx >= len(decomp):
                continue
            
            gold_idx = decomp[step_idx].get('paragraph_support_idx')
            if gold_idx is None or gold_idx >= len(paragraphs):
                continue
            
            gold_text = paragraphs[gold_idx]['paragraph_text']
            hop_num = step_idx + 1
            lab = fg.get((qid, hop_num))
            if lab is None:
                continue
            
            total += 1
            gold_present = check_gold_in_passages(gold_text, accumulated)
            
            if lab == 1:
                fact_present += 1
            elif gold_present:
                ext_fail += 1
            else:
                ret_fail += 1
    
    print(f"\n=== {label} ===")
    print(f"Total hops: {total}")
    print(f"Retrieval failure: {ret_fail} ({ret_fail/total*100:.1f}%)")
    print(f"Extraction failure: {ext_fail} ({ext_fail/total*100:.1f}%)")
    print(f"Fact present: {fact_present} ({fact_present/total*100:.1f}%)")

raw_path = 'raw_data/musique/musique_ans_v1.0_dev.jsonl'

decompose('results/full_run_all_4.1.json',
          'results/fact_grounded_final_dev.jsonl',
          raw_path, 'BM25')

decompose('results/contriever_base_run.json',
          'results/contriever_fact_grounded.jsonl',
          raw_path, 'CONTRIEVER')
