#!/usr/bin/env python3
"""Judged corpus search: for each extraction failure, run the LLM judge on
the top-10 corpus passages containing the answer string (excluding the gold
passage itself), to test whether the needed fact is actually stated anywhere
in the corpus.

Usage:
  nohup python3 corpus_fact_search_judged.py > judged_search.log 2>&1 &
"""
import json
from elasticsearch import Elasticsearch
from openai import OpenAI
from tqdm import tqdm

JUDGE_PROMPT = """You are evaluating whether a sub-question can be answered using ONLY the provided passages.

Sub-question: {sub_question}

Passages:
{passages}

Can the sub-question be answered using only the information in the passages above? Provide brief reasoning, then answer YES or NO."""

es = Elasticsearch(['http://localhost:9200'], timeout=30)
client = OpenAI()

data = []
with open('results/fact_grounded_final_dev.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

raw = {}
with open('raw_data/musique/musique_ans_v1.0_dev.jsonl') as f:
    for line in f:
        d = json.loads(line)
        raw[d['id']] = d

# Same extraction-failure filter as corpus_fact_search.py,
# plus: carry the gold-passage prefix for exclusion during judging
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
    gold_text = paragraphs[gold_idx]['paragraph_text'].strip().lower()
    passages_text = d.get('passage_text_combined', '').lower()
    gold_present = gold_text[:200] in passages_text if passages_text else False
    if gold_present and d['label_binary'] == 0:
        d['_gold_prefix'] = gold_text[:200]
        extraction_failures.append(d)

print(f"Extraction failures to check: {len(extraction_failures)}")

results = []
fact_found = 0
fact_absent = 0
for d in tqdm(extraction_failures, desc="Judged corpus search"):
    answer = d['intermediate_answer'].strip()
    sub_q = d.get('sub_question_natural', d.get('sub_question', ''))
    if not answer or not sub_q:
        fact_absent += 1
        results.append({'qid': d['qid'], 'hop': d['hop_number'],
                        'status': 'skipped_empty'})
        continue

    r = es.search(index='musique', body={
        "size": 10,
        "_source": ["title", "paragraph_text"],
        "query": {"match_phrase": {"paragraph_text": answer}}
    })
    hits = r['hits']['hits']

    found = False
    judged_any = False
    for h in hits:
        p_text = h['_source']['paragraph_text']
        # Skip the gold passage itself (already judged NO in the pipeline)
        if p_text.strip().lower()[:200] == d['_gold_prefix']:
            continue
        judged_any = True
        try:
            resp = client.chat.completions.create(
                model='gpt-4.1-mini',
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    sub_question=sub_q, passages=p_text[:2000])}],
                temperature=0.0, max_tokens=150)
            verdict = resp.choices[0].message.content.strip().upper()
            if verdict.endswith('YES') or ' YES' in verdict[-20:]:
                found = True
                results.append({'qid': d['qid'], 'hop': d['hop_number'],
                                'status': 'fact_found',
                                'passage_title': h['_source'].get('title', '')})
                break
        except Exception as e:
            print(f"\n[warn] {d['qid']}: {e}")
    if found:
        fact_found += 1
    else:
        fact_absent += 1
        status = 'fact_absent_judged' if judged_any else 'no_nongold_hits'
        results.append({'qid': d['qid'], 'hop': d['hop_number'],
                        'status': status})

    if len(results) % 25 == 0:
        json.dump({'checked': len(results), 'fact_found': fact_found,
                   'fact_absent': fact_absent, 'results': results},
                  open('results/corpus_fact_search_judged.json', 'w'))

total = fact_found + fact_absent
print(f"\nJudged corpus search results:")
print(f"  Fact found elsewhere : {fact_found}/{total} ({fact_found/total*100:.1f}%)")
print(f"  Fact absent (judged) : {fact_absent}/{total} ({fact_absent/total*100:.1f}%)")
json.dump({'checked': len(results), 'fact_found': fact_found,
           'fact_absent': fact_absent, 'results': results},
          open('results/corpus_fact_search_judged.json', 'w'))
print("Saved to results/corpus_fact_search_judged.json")
