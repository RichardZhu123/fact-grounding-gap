#!/usr/bin/env python3
"""IIRC fact-grounding: question-level decomposition."""
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
    parser.add_argument("--input", default="raw_data/iirc/dev.json")
    parser.add_argument("--output", default="results/iirc_factgrounding.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", default="iirc")
    args = parser.parse_args()

    client = OpenAI()
    retriever = ESRetriever(index=args.index)

    with open(args.input) as f:
        data = json.load(f)

    questions = []
    for d in data:
        main_title = d['title']
        main_text = d['text']
        for q in d['questions']:
            if q['answer']['type'] != 'span':
                continue
            if q['answer']['answer_spans'][0]['passage'] == 'main':
                continue
            gold_passage = q['answer']['answer_spans'][0]['passage']
            gold_answer = q['answer']['answer_spans'][0]['text']
            questions.append({
                'qid': q['qid'],
                'question': q['question'],
                'gold_answer': gold_answer,
                'gold_passage_title': gold_passage,
                'main_title': main_title,
                'main_text': main_text,
            })

    if args.limit:
        questions = questions[:args.limit]

    print(f"Running on {len(questions)} span questions")

    fact_present = 0
    extraction_failure = 0
    retrieval_failure = 0
    total = 0
    results = []

    for item in tqdm(questions, desc="IIRC"):
        question = item['question']
        gold_title = item['gold_passage_title'].strip().lower()

        hop1 = retriever.retrieve(question, k=3)

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

        accumulated = []
        seen = set()
        for p in hop1 + hop2:
            t = p.get('text', '')
            if t not in seen:
                seen.add(t)
                accumulated.append(p)

        retrieved_titles = set(p.get('title', '').strip().lower() for p in accumulated)
        gold_present = gold_title in retrieved_titles

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

        if answerable:
            category = 'fact_present'
        elif gold_present:
            category = 'extraction_failure'
        else:
            category = 'retrieval_failure'

        total += 1
        if category == 'fact_present':
            fact_present += 1
        elif category == 'extraction_failure':
            extraction_failure += 1
        else:
            retrieval_failure += 1

        results.append({
            'qid': item['qid'],
            'question': question,
            'gold_answer': item['gold_answer'],
            'gold_passage_title': item['gold_passage_title'],
            'gold_present': gold_present,
            'answerable': answerable,
            'category': category,
        })

    t = total
    print(f"\n{'='*60}")
    print(f"IIRC Question-Level Decomposition (n={t})")
    print(f"{'='*60}")
    print(f"Fact present:       {fact_present} ({fact_present/t*100:.1f}%)")
    print(f"Extraction failure: {extraction_failure} ({extraction_failure/t*100:.1f}%)")
    print(f"Retrieval failure:  {retrieval_failure} ({retrieval_failure/t*100:.1f}%)")

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
