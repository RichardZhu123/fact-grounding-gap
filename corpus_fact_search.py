#!/usr/bin/env python3
"""For each extraction failure, check if the needed fact exists anywhere in the corpus."""
import json
from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'], timeout=30)

# Load fact-grounding labels
data = []
with open('results/fact_grounded_final_dev.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

# Load raw data for gold paragraph check
raw = {}
with open('raw_data/musique/musique_ans_v1.0_dev.jsonl') as f:
    for line in f:
        d = json.loads(line)
        raw[d['id']] = d

# Load gold passage analysis to get true extraction failures
gold = json.load(open('results/gold_passage_analysis.json'))

# For each hop that's an extraction failure, search the full corpus
# for the intermediate answer
extraction_failures = []
for d in data:
    qid = d['qid']
    if qid not in raw:
        continue
    musique = raw[qid]
    paragraphs = musique['paragraphs']
    decomp = musique['question_decomposition']
    hop = d['hop_number'] - 1
    if hop >= len(decomp):
        continue
    
    gold_idx = decomp[hop].get('paragraph_support_idx')
    if gold_idx is None:
        continue
    
    # Check if gold para was in retrieved passages (by text match)
    gold_text = paragraphs[gold_idx]['paragraph_text'].strip().lower()
    passages_text = d.get('passage_text_combined', '').lower()
    gold_present = gold_text[:200] in passages_text if passages_text else False
    
    # Extraction failure = gold present but not answerable
    if gold_present and d['label_binary'] == 0:
        extraction_failures.append(d)

print(f"Extraction failures to check: {len(extraction_failures)}")

found_elsewhere = 0
not_found = 0
for d in extraction_failures:
    answer = d['intermediate_answer'].strip()
    if not answer:
        not_found += 1
        continue
    
    # Search corpus for the answer
    result = es.search(index='musique', body={
        "size": 20,
        "_source": ["title", "paragraph_text"],
        "query": {"match_phrase": {"paragraph_text": answer}}
    })
    
    hits = result['hits']['hits']
    if len(hits) > 0:
        found_elsewhere += 1
    else:
        not_found += 1

total = found_elsewhere + not_found
print(f"Answer string found elsewhere in corpus: {found_elsewhere}/{total} ({found_elsewhere/total*100:.1f}%)")
print(f"Answer string NOT found in corpus: {not_found}/{total} ({not_found/total*100:.1f}%)")
print("(Note: string presence != fact presence, so this is an upper bound)")
