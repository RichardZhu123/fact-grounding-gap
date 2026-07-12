#!/usr/bin/env python3
"""HotpotQA fact-grounding: question-level decomposition."""
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from es_retriever import ESRetriever

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
    parser.add_argument("--input", default="raw_data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output", default="results/hotpotqa_factgrounding.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", default="hotpotqa")
    args = parser.parse_args()

    client = OpenAI()
    retriever = ESRetriever(index=args.index)

    with open(args.input) as f:
        data = json.load(f)

    data = [d for d in data if d.get('type') == 'bridge']
    if args.limit:
        data = data[:args.limit]

    print(f"Running on {len(data)} bridge questions")

    fact_present = 0
    extraction_failure = 0
    retrieval_failure = 0
    total = 0
    results = []

    for item in tqdm(data, desc="HotpotQA"):
        question = item['question']
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

        total += 1
        if answerable:
            fact_present += 1
        elif all_gold_present:
            extraction_failure += 1
        else:
            retrieval_failure += 1

        results.append({
            'qid': item['_id'],
            'question': question,
            'gold_answer': item['answer'],
            'all_gold_present': all_gold_present,
            'answerable': answerable,
            'gold_titles': list(gold_titles),
            'retrieved_titles': list(retrieved_titles),
        })

    print(f"\n{'='*60}")
    print(f"HotpotQA Question-Level Decomposition (n={total})")
    print(f"{'='*60}")
    print(f"Fact present:       {fact_present} ({fact_present/total*100:.1f}%)")
    print(f"Extraction failure: {extraction_failure} ({extraction_failure/total*100:.1f}%)")
    print(f"Retrieval failure:  {retrieval_failure} ({retrieval_failure/total*100:.1f}%)")

    summary = {
        'total': total,
        'fact_present': fact_present,
        'extraction_failure': extraction_failure,
        'retrieval_failure': retrieval_failure,
        'results': results
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
