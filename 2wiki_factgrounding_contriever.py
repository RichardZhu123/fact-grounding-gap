#!/usr/bin/env python3
"""2WikiMQA fact-grounding: question-level decomposition with per-type breakdown."""
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
from contriever_retriever import ContrieverRetriever

JUDGE_PROMPT = """You are evaluating whether a question can be answered using ONLY the provided passages.
Question: {question}
Passages: {passages}
Can the question be answered using only the information in the passages above? Provide brief reasoning, then answer YES or NO."""

def format_passages(passages, max_chars=6000):
    parts, total = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"[{i}] {p.get('title','')}: {p.get('text','')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_data/2wikimultihopqa/dev.json")
    parser.add_argument("--output", default="results/2wiki_factgrounding_contriever.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", default="2wikimultihopqa")
    args = parser.parse_args()

    client = OpenAI()
    retriever = ContrieverRetriever(index_path="models/contriever_2wiki.index", meta_path="models/contriever_2wiki_meta.json")

    with open(args.input) as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    print(f"Running on {len(data)} questions (all types)")

    overall = {'fact_present': 0, 'extraction_failure': 0, 'retrieval_failure': 0, 'total': 0}
    by_type = defaultdict(lambda: {'fact_present': 0, 'extraction_failure': 0, 'retrieval_failure': 0, 'total': 0})
    results = []

    for item in tqdm(data, desc="2WikiMQA"):
        question = item['question']
        qtype = item.get('type', 'unknown')
        gold_titles = set(t.strip().lower() for t, _ in item['supporting_facts'])

        # Hop 1: retrieve for original question
        hop1 = retriever.retrieve(question, k=3)

        # Hop 2: generate follow-up, retrieve again
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": f"""Given these passages and the question, what follow-up question would help find missing information?

Question: {question}
Passages: {format_passages(hop1, max_chars=2000)}

Write just the follow-up question, nothing else."""}],
                temperature=0.0, max_tokens=100
            )
            followup = resp.choices[0].message.content.strip()
        except:
            followup = question

        hop2 = retriever.retrieve(followup, k=3)

        # Accumulate and dedupe
        accumulated = []
        seen = set()
        for p in hop1 + hop2:
            t = p.get('text', '')
            if t not in seen:
                seen.add(t)
                accumulated.append(p)

        # Check: are ALL gold paragraphs in accumulated passages?
        retrieved_titles = set(p.get('title', '').strip().lower() for p in accumulated)
        all_gold_present = gold_titles.issubset(retrieved_titles)

        # Judge: can the full question be answered?
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, passages=format_passages(accumulated))}],
                temperature=0.0, max_tokens=200
            )
            answer_text = resp.choices[0].message.content.strip()
            last_lines = [l for l in answer_text.strip().split('\n') if l.strip()][-3:]
            answerable = any("YES" in l.upper() for l in last_lines)
        except:
            answerable = False

        # Classify
        if answerable:
            category = 'fact_present'
        elif all_gold_present:
            category = 'extraction_failure'
        else:
            category = 'retrieval_failure'

        overall[category] += 1
        overall['total'] += 1
        by_type[qtype][category] += 1
        by_type[qtype]['total'] += 1

        results.append({
            'qid': item['_id'],
            'question': question,
            'type': qtype,
            'gold_answer': item['answer'],
            'all_gold_present': all_gold_present,
            'answerable': answerable,
            'category': category,
        })

    # Print results
    t = overall['total']
    print(f"\n{'='*60}")
    print(f"2WikiMQA Question-Level Decomposition (n={t})")
    print(f"{'='*60}")
    print(f"Fact present:       {overall['fact_present']} ({overall['fact_present']/t*100:.1f}%)")
    print(f"Extraction failure: {overall['extraction_failure']} ({overall['extraction_failure']/t*100:.1f}%)")
    print(f"Retrieval failure:  {overall['retrieval_failure']} ({overall['retrieval_failure']/t*100:.1f}%)")

    print(f"\nPer-type breakdown:")
    print(f"{'Type':20s} {'N':>6s} {'Fact present':>14s} {'Ext failure':>14s} {'Ret failure':>14s}")
    for qtype, s in sorted(by_type.items()):
        n = s['total']
        print(f"{qtype:20s} {n:6d} {s['fact_present']/n*100:12.1f}% {s['extraction_failure']/n*100:12.1f}% {s['retrieval_failure']/n*100:12.1f}%")

    summary = {
        'overall': overall,
        'by_type': dict(by_type),
        'results': results
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
