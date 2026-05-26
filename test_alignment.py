#!/usr/bin/env python3
"""Check semantic alignment between MuSiQue natural sub-questions and system queries."""

import json, random
from openai import OpenAI

# Same setup as the test script
with open("results/full_run_all_4.1.json") as f:
    qa = json.load(f)
with open("results/musique_dev_subquestions.json") as f:
    subq_lookup = json.load(f)
with open("results/fact_grounded_final_dev.jsonl") as f:
    oracle = {}
    for line in f:
        e = json.loads(line)
        if e['label'] == 'UNKNOWN':
            continue
        oracle.setdefault(e['qid'], {})[e['sub_question']] = e['label_binary']

candidates = []
for r in qa['results']:
    qid = r.get('qid', '')
    if qid not in subq_lookup or qid not in oracle:
        continue
    natural_subqs = subq_lookup[qid].get('natural_sub_questions', [])
    steps = r.get('trajectory', {}).get('steps', [])
    for i, sub_q in enumerate(natural_subqs):
        step_num = i + 1
        if step_num > len(steps):
            continue
        if sub_q not in oracle[qid]:
            continue
        candidates.append({
            'qid': qid,
            'hop_num': step_num,
            'system_query': steps[step_num - 1].get('query', ''),
            'natural_sub_question': sub_q,
        })

random.seed(42)
random.shuffle(candidates)
sample = candidates[:50]  # SAME SEED as the deberta test → same 50 examples

client = OpenAI()
ALIGN_PROMPT = """Two questions are below. Are they asking about the same fact?

Question A: {a}
Question B: {b}

Answer YES if they are asking for substantially the same piece of information (even if phrased differently). Answer NO if they are asking about different topics or different facts.

Answer ONLY "YES" or "NO"."""

aligned = 0
breakdown = []
for i, c in enumerate(sample):
    prompt = ALIGN_PROMPT.format(a=c['natural_sub_question'], b=c['system_query'])
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    verdict = "YES" in resp.choices[0].message.content.strip().upper()
    if verdict:
        aligned += 1
    breakdown.append({'hop': c['hop_num'], 'aligned': verdict, 'natural': c['natural_sub_question'][:80], 'system': c['system_query'][:80]})

print(f"\nALIGNMENT: {aligned}/{len(sample)} = {aligned/len(sample)*100:.1f}%")
print(f"\nPer-hop breakdown:")
hop_counts = {}
for b in breakdown:
    h = b['hop']
    hop_counts.setdefault(h, [0, 0])
    hop_counts[h][1] += 1
    if b['aligned']:
        hop_counts[h][0] += 1
for h in sorted(hop_counts):
    a, t = hop_counts[h]
    print(f"  Hop {h}: {a}/{t} = {a/t*100:.0f}% aligned")

print(f"\nMisaligned examples:")
for b in breakdown:
    if not b['aligned']:
        print(f"  hop {b['hop']}")
        print(f"    natural: {b['natural']}")
        print(f"    system:  {b['system']}")
