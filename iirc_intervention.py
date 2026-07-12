#!/usr/bin/env python3
"""Simple IIRC intervention: baseline vs always-reretrieval."""
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from es_retriever import ESRetriever

QA_PROMPT = """Answer the question using ONLY the passages below.
Give a SHORT, direct answer (just the entity/fact, no explanation).

Passages:
{passages}

Question: {question}
Answer:"""

JUDGE_PROMPT = """Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
Is the predicted answer semantically equivalent to the gold answer?
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_data/iirc/dev.json")
    parser.add_argument("--output", default="results/iirc_intervention.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", default="iirc")
    args = parser.parse_args()

    client = OpenAI()
    retriever = ESRetriever(index=args.index)

    with open(args.input) as f:
        data = json.load(f)

    questions = []
    for d in data:
        for q in d['questions']:
            if q['answer']['type'] != 'span':
                continue
            if q['answer']['answer_spans'][0]['passage'] == 'main':
                continue
            questions.append({
                'qid': q['qid'],
                'question': q['question'],
                'gold_answer': q['answer']['answer_spans'][0]['text'],
                'main_text': d['text'],
            })

    if args.limit:
        questions = questions[:args.limit]

    print(f"Running on {len(questions)} questions")

    baseline_correct = 0
    always_correct = 0
    total = 0
    results = []

    for item in tqdm(questions, desc="IIRC intervention"):
        question = item['question']
        gold = item['gold_answer'].strip().lower()

        # Baseline: 2 hops of retrieval
        hop1 = retriever.retrieve(question, k=3)
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": f"Given this question and passages, what follow-up question would help?\n\nQuestion: {question}\nPassages: {format_passages(hop1, max_chars=2000)}\n\nWrite just the follow-up question."}],
                temperature=0.0, max_tokens=100
            )
            followup = resp.choices[0].message.content.strip()
        except:
            followup = question
        hop2 = retriever.retrieve(followup, k=3)

        # Dedupe baseline passages
        seen = set()
        baseline_passages = []
        for p in hop1 + hop2:
            if p.get('text', '') not in seen:
                seen.add(p['text'])
                baseline_passages.append(p)

        # Baseline answer
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": QA_PROMPT.format(
                    passages=format_passages(baseline_passages), question=question)}],
                temperature=0.0, max_tokens=100
            )
            base_ans = resp.choices[0].message.content.strip().lower()
        except:
            base_ans = ""

        # Always: reformulate and retrieve again
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": f"Rephrase this question to search for different information:\n\n{question}\n\nWrite just the rephrased question."}],
                temperature=0.0, max_tokens=100
            )
            reformulated = resp.choices[0].message.content.strip()
        except:
            reformulated = question
        extra = retriever.retrieve(reformulated, k=3)

        always_passages = list(baseline_passages)
        for p in extra:
            if p.get('text', '') not in seen:
                seen.add(p['text'])
                always_passages.append(p)

        # Always answer
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": QA_PROMPT.format(
                    passages=format_passages(always_passages), question=question)}],
                temperature=0.0, max_tokens=100
            )
            always_ans = resp.choices[0].message.content.strip().lower()
        except:
            always_ans = ""

        # Judge
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, gold=gold, pred=base_ans)}],
                temperature=0.0, max_tokens=5
            )
            base_ok = "YES" in resp.choices[0].message.content.strip().upper()
        except:
            base_ok = False

        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, gold=gold, pred=always_ans)}],
                temperature=0.0, max_tokens=5
            )
            always_ok = "YES" in resp.choices[0].message.content.strip().upper()
        except:
            always_ok = False

        total += 1
        if base_ok:
            baseline_correct += 1
        if always_ok:
            always_correct += 1

        results.append({
            'qid': item['qid'],
            'baseline_correct': base_ok,
            'always_correct': always_ok,
        })

    base_acc = baseline_correct / total * 100
    always_acc = always_correct / total * 100
    print(f"\nBaseline: {baseline_correct}/{total} = {base_acc:.1f}%")
    print(f"Always:   {always_correct}/{total} = {always_acc:.1f}%")
    print(f"Delta:    {always_acc - base_acc:+.1f}%")

    with open(args.output, 'w') as f:
        json.dump({'total': total, 'baseline': baseline_correct, 'always': always_correct,
                   'baseline_acc': base_acc, 'always_acc': always_acc, 'results': results}, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
