#!/usr/bin/env python3
"""Judged corpus search for HotpotQA / 2WikiMQA / IIRC extraction failures.

For each extraction failure (gold present, not answerable), search the ES
corpus for the gold answer string, exclude gold passages by title, and run
the LLM judge on top-10 non-gold hits against the full question.

Usage:
  nohup python3 corpus_fact_search_judged_multi.py --dataset hotpotqa > judged_hotpotqa.log 2>&1 &
  nohup python3 corpus_fact_search_judged_multi.py --dataset 2wiki    > judged_2wiki.log 2>&1 &
  nohup python3 corpus_fact_search_judged_multi.py --dataset iirc     > judged_iirc.log 2>&1 &
"""
import json
import argparse
from elasticsearch import Elasticsearch
from openai import OpenAI
from tqdm import tqdm

CONFIG = {
    'hotpotqa': {'file': 'results/hotpotqa_factgrounding.json', 'index': 'hotpotqa'},
    '2wiki':    {'file': 'results/2wiki_factgrounding.json',    'index': '2wikimultihopqa'},
    'iirc':     {'file': 'results/iirc_factgrounding.json',     'index': 'iirc'},
}

JUDGE_PROMPT = """You are evaluating whether a sub-question can be answered using ONLY the provided passages.

Sub-question: {sub_question}

Passages:
{passages}

Can the sub-question be answered using only the information in the passages above? Provide brief reasoning, then answer YES or NO."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(CONFIG))
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()
    cfg = CONFIG[args.dataset]

    es = Elasticsearch(['http://localhost:9200'], timeout=30)
    client = OpenAI()

    d = json.load(open(cfg['file']))
    records = d['results']
    failures = [x for x in records
                if (x.get('all_gold_present') or x.get('gold_present'))
                and not x.get('answerable')]
    print(f"[{args.dataset}] extraction failures: {len(failures)}")

    out_path = f"results/corpus_fact_search_judged_{args.dataset}.json"
    results, fact_found, fact_absent = [], 0, 0

    for x in tqdm(failures, desc=f"judged search {args.dataset}"):
        answer = (x.get('gold_answer') or '').strip()
        question = (x.get('question') or '').strip()
        gold_titles = set()
        if 'gold_titles' in x:
            gold_titles = {t.strip().lower() for t in x['gold_titles']}
        elif 'gold_passage_title' in x:
            gold_titles = {x['gold_passage_title'].strip().lower()}

        if not answer or not question:
            fact_absent += 1
            results.append({'qid': x['qid'], 'status': 'skipped_empty'})
            continue

        try:
            r = es.search(index=cfg['index'], body={
                "size": args.topk,
                "_source": ["title", "paragraph_text"],
                "query": {"match_phrase": {"paragraph_text": answer}}
            })
            hits = r['hits']['hits']
        except Exception as e:
            print(f"\n[warn-es] {x['qid']}: {e}")
            fact_absent += 1
            results.append({'qid': x['qid'], 'status': 'es_error'})
            continue

        found = False
        judged_any = False
        for h in hits:
            title = (h['_source'].get('title') or '').strip().lower()
            if title in gold_titles:
                continue  # skip gold passages (already judged NO in pipeline)
            judged_any = True
            try:
                resp = client.chat.completions.create(
                    model='gpt-4.1-mini',
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        sub_question=question,
                        passages=h['_source']['paragraph_text'][:2000])}],
                    temperature=0.0, max_tokens=150)
                verdict = resp.choices[0].message.content.strip().upper()
                if verdict.endswith('YES') or ' YES' in verdict[-20:]:
                    found = True
                    results.append({'qid': x['qid'], 'status': 'fact_found',
                                    'passage_title': h['_source'].get('title', '')})
                    break
            except Exception as e:
                print(f"\n[warn-llm] {x['qid']}: {e}")
        if found:
            fact_found += 1
        else:
            fact_absent += 1
            status = 'fact_absent_judged' if judged_any else 'no_nongold_hits'
            results.append({'qid': x['qid'], 'status': status})

        if len(results) % 25 == 0:
            json.dump({'dataset': args.dataset, 'checked': len(results),
                       'fact_found': fact_found, 'fact_absent': fact_absent,
                       'results': results}, open(out_path, 'w'))

    total = fact_found + fact_absent
    print(f"\n[{args.dataset}] Judged corpus search results:")
    print(f"  Fact found elsewhere : {fact_found}/{total} ({fact_found/total*100:.1f}%)")
    print(f"  Fact absent (judged) : {fact_absent}/{total} ({fact_absent/total*100:.1f}%)")
    json.dump({'dataset': args.dataset, 'checked': len(results),
               'fact_found': fact_found, 'fact_absent': fact_absent,
               'results': results}, open(out_path, 'w'))
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
