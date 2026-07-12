#!/usr/bin/env python3
"""Oracle passage experiment: feed gold passages to LLM on 2Wiki compositional questions."""
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

QA_PROMPT = """Answer the question using ONLY the passages below.
Give a SHORT, direct answer (just the entity/fact, no explanation).

Passages:
{passages}

Question: {question}
Answer:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_data/2wikimultihopqa/dev.json")
    parser.add_argument("--output", default="results/2wiki_oracle_compositional.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = OpenAI()

    with open(args.input) as f:
        data = json.load(f)

    # Filter to compositional only
    data = [d for d in data if d.get('type') == 'compositional']
    if args.limit:
        data = data[:args.limit]

    print(f"Running oracle on {len(data)} compositional questions")

    correct = 0
    total = 0
    results = []

    for item in tqdm(data, desc="Oracle"):
        question = item['question']
        gold = item['answer'].strip().lower()

        # Build gold passages from supporting_facts
        sf_titles = set(t for t, _ in item['supporting_facts'])
        gold_passages = []
        for title, sentences in item['context']:
            if title in sf_titles:
                gold_passages.append(f"{title}: {' '.join(sentences)}")

        if not gold_passages:
            continue

        passages_text = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(gold_passages))

        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": QA_PROMPT.format(
                    passages=passages_text, question=question)}],
                temperature=0.0, max_tokens=100
            )
            pred = resp.choices[0].message.content.strip().lower()
        except:
            pred = ""

        is_correct = gold in pred or pred in gold
        total += 1
        if is_correct:
            correct += 1

        results.append({
            'qid': item['_id'],
            'question': question,
            'gold': item['answer'],
            'pred': pred,
            'correct': is_correct,
        })

    acc = correct / total * 100 if total else 0
    print(f"\nOracle accuracy on compositional: {correct}/{total} = {acc:.1f}%")
    print(f"(Baseline was 66.4%)")
    print(f"If oracle ~66% → reasoning bottleneck, not retrieval")
    print(f"If oracle ~90% → retrieval is the issue")

    with open(args.output, 'w') as f:
        json.dump({'total': total, 'correct': correct, 'accuracy': acc, 'results': results}, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
