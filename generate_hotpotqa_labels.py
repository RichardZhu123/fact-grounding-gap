#!/usr/bin/env python3
"""
Generate DeBERTa training labels for HotpotQA.
Decompose (hop-2 conditioned on gold bridge entity) -> BM25 k=3 with
accumulation -> LLM judge on retrieved passages -> JSONL triples.
Resumable; atomic per-qid writes; memory-safe on 4GB RAM.

Usage:
  nohup python3 generate_hotpotqa_labels.py --limit 6000 > hotpotqa_labelgen.log 2>&1 &
"""
import json
import argparse
import gc
import os
from tqdm import tqdm
from openai import OpenAI
from es_retriever import ESRetriever

DECOMPOSE_PROMPT = """Decompose this multi-hop question into exactly 2 sequential sub-questions.
The answer to sub-question 1 is: {intermediate}
Sub-question 2 should use that answer and lead to the final answer.

Question: {question}

Reply with exactly two lines:
1: <sub-question 1>
2: <sub-question 2>"""

JUDGE_PROMPT = """Given the following passages, can the question below be answered using ONLY information in the passages?

Passages:
{passages}

Question: {question}

Reply with ONLY "YES" or "NO"."""


def format_passages(passages, max_chars=6000):
    parts, total = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"[{i}] {p.get('title','')}: {p.get('text','')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def get_intermediate_answer(item):
    titles = list(dict.fromkeys(t for t, _ in item['supporting_facts']))
    if len(titles) < 2:
        return None
    q = item['question'].lower()
    not_in_q = [t for t in titles if t.lower() not in q]
    return (not_in_q[0] if not_in_q else titles[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_file', default='raw_data/hotpotqa/hotpot_train_v1.1.json')
    ap.add_argument('--output', default='results/hotpotqa_train_labels.jsonl')
    ap.add_argument('--limit', type=int, default=6000)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--model', default='gpt-4.1-mini')
    args = ap.parse_args()

    client = OpenAI()
    retriever = ESRetriever(index='hotpotqa')

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['qid'])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} qids already labeled")

    print("Loading train set (large file, ~1 min)...")
    with open(args.train_file) as f:
        data = json.load(f)
    bridge = [x for x in data if x.get('type') == 'bridge'][:args.limit]
    del data
    gc.collect()
    print(f"{len(bridge)} bridge questions selected")

    out = open(args.output, 'a')
    n_done = 0
    for item in tqdm(bridge, desc="Labeling"):
        qid = item['_id']
        if qid in done:
            continue
        try:
            intermediate = get_intermediate_answer(item)
            if intermediate is None:
                continue

            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(
                    question=item['question'], intermediate=intermediate)}],
                temperature=0.0, max_tokens=150)
            lines = [l.strip() for l in resp.choices[0].message.content.strip().split('\n') if l.strip()]
            subs = []
            for l in lines:
                if l[:2] in ('1:', '2:'):
                    subs.append(l[2:].strip())
            if len(subs) != 2:
                continue

            accumulated = []
            hop_records = []
            for hop_num, sub_q in enumerate(subs, 1):
                passages = retriever.retrieve(sub_q, k=args.k)
                accumulated.extend(passages)
                seen, uniq = set(), []
                for p in accumulated:
                    if p['text'] not in seen:
                        seen.add(p['text'])
                        uniq.append(p)
                accumulated = uniq

                jr = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        passages=format_passages(accumulated), question=sub_q)}],
                    temperature=0.0, max_tokens=5)
                verdict = jr.choices[0].message.content.strip().upper()
                label = 1 if verdict.startswith('Y') else 0

                hop_records.append({
                    'qid': qid,
                    'hop_number': hop_num,
                    'sub_question': sub_q,
                    'passages': [{'title': p.get('title',''), 'text': p.get('text','')} for p in accumulated],
                    'label_binary': label,
                })

            for rec in hop_records:
                out.write(json.dumps(rec) + '\n')
            n_done += 1
            if n_done % 50 == 0:
                out.flush()
        except Exception as e:
            print(f"\n[warn] {qid}: {e}")
            continue
    out.close()
    print("Done.")


if __name__ == '__main__':
    main()
