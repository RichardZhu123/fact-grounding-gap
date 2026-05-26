#!/usr/bin/env python3
"""
Generate 2-hop sub-question decompositions for HotpotQA bridge questions.
Uses GPT-4.1-mini to decompose each question into 2 sequential sub-questions.
Processes ALL bridge questions by default (one-time preprocessing).
"""

import json
import argparse
from tqdm import tqdm
from openai import OpenAI


DECOMPOSE_PROMPT = """Decompose this multi-hop question into exactly 2 sequential sub-questions.
The first sub-question should find a bridge entity.
The second sub-question should use that bridge entity to find the final answer.

Question: {question}

Output ONLY two lines:
Step 1: [first sub-question]
Step 2: [second sub-question]"""


def decompose(client, question, model="gpt-4.1-mini"):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
        temperature=0.0,
        max_tokens=150,
    )
    output = resp.choices[0].message.content.strip()

    sub_questions = []
    for line in output.split("\n"):
        line = line.strip()
        if line.lower().startswith("step 1:"):
            sub_questions.append(line[7:].strip())
        elif line.lower().startswith("step 2:"):
            sub_questions.append(line[7:].strip())

    if len(sub_questions) != 2:
        # Fallback: take first two non-empty lines
        lines = [l.strip() for l in output.split("\n") if l.strip()]
        sub_questions = lines[:2] if len(lines) >= 2 else [question, question]

    return sub_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_data/hotpotqa/hotpot_dev_distractor_v1.json")
    parser.add_argument("--output", default="results/hotpotqa_subquestions.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (default: all bridge)")
    args = parser.parse_args()

    print("Loading HotpotQA dev set...")
    with open(args.input) as f:
        data = json.load(f)

    # Filter to bridge questions only
    bridge = [q for q in data if q['type'] == 'bridge']
    print(f"Total bridge questions: {len(bridge)}")

    if args.limit:
        bridge = bridge[:args.limit]
        print(f"Limited to {len(bridge)} questions")

    client = OpenAI()
    results = {}

    for q in tqdm(bridge, desc="Decomposing"):
        qid = q['_id']
        try:
            subs = decompose(client, q['question'])
            results[qid] = {
                'question': q['question'],
                'answer': q['answer'],
                'type': q['type'],
                'supporting_facts': q['supporting_facts'],
                'natural_sub_questions': subs,
                'context_titles': [title for title, _ in q['context']],
            }
        except Exception as e:
            print(f"Failed for {qid}: {e}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDecomposed {len(results)} questions")
    print(f"Saved to {args.output}")

    # Show a few examples
    print("\nExamples:")
    for qid in list(results.keys())[:5]:
        r = results[qid]
        print(f"  Q: {r['question'][:80]}")
        print(f"    Step 1: {r['natural_sub_questions'][0]}")
        print(f"    Step 2: {r['natural_sub_questions'][1]}")
        print()


if __name__ == "__main__":
    main()
