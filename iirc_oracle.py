#!/usr/bin/env python3
"""
IIRC oracle experiment: answer each question using its annotated gold
context spans (the evidence IIRC marks as required). Mirrors the 2WikiMQA
oracle: if the diagnostic is right that IIRC is retrieval-bottlenecked,
accuracy with gold context should jump far above the retrieved-passage
baseline.

Usage:
  nohup python3 iirc_oracle.py > iirc_oracle.log 2>&1 &
"""
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

ANSWER_PROMPT = """Answer the question using ONLY the provided context. Reply with just the answer, as briefly as possible. If the context does not contain the answer, reply "Not stated".

Context:
{context}

Question: {question}

Answer:"""

JUDGE_PROMPT = """Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer semantically equivalent to the gold answer? Reply with ONLY "YES" or "NO"."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='raw_data/iirc/dev.json')
    ap.add_argument('--output', default='results/iirc_oracle.json')
    ap.add_argument('--model', default='gpt-4.1-mini')
    args = ap.parse_args()

    client = OpenAI()

    with open(args.input) as f:
        data = json.load(f)

    questions = []
    for para in data:
        for q in para.get('questions', []):
            ans = q.get('answer', {})
            if ans.get('type') != 'span' or not ans.get('answer_spans'):
                continue  # match the intervention script's span-only filter
            gold = ans['answer_spans'][0]['text']
            ctx_texts = [c['text'] for c in q.get('context', []) if c.get('text')]
            if not ctx_texts:
                continue
            # dedupe (overlapping spans are common)
            seen, uniq = set(), []
            for t in ctx_texts:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            questions.append({
                'qid': q['qid'],
                'question': q['question'],
                'gold': gold,
                'context': '\n'.join(f'- {t}' for t in uniq),
            })
    print(f"{len(questions)} span-answer questions with gold context")

    results, correct = [], 0
    for item in tqdm(questions, desc="IIRC oracle"):
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                    context=item['context'], question=item['question'])}],
                temperature=0.0, max_tokens=50)
            pred = r.choices[0].message.content.strip()

            j = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=item['question'], gold=item['gold'], pred=pred)}],
                temperature=0.0, max_tokens=5)
            ok = j.choices[0].message.content.strip().upper().startswith('Y')
            correct += ok
            results.append({'qid': item['qid'], 'pred': pred,
                            'gold': item['gold'], 'correct': ok})
        except Exception as e:
            print(f"\n[warn] {item['qid']}: {e}")
            results.append({'qid': item['qid'], 'error': str(e)})

    n = len([r for r in results if 'correct' in r])
    acc = correct / n if n else 0.0
    print(f"\nOracle accuracy: {correct}/{n} = {acc*100:.1f}%")
    json.dump({'n': n, 'correct': correct, 'accuracy': acc,
               'results': results}, open(args.output, 'w'))
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
