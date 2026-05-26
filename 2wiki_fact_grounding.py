#!/usr/bin/env python3
"""
LLM fact-grounding labels for 2WikiMQA train trajectories.
Same methodology as MuSiQue: for each hop, ask LLM if accumulated
retrieved passages contain enough info to answer the sub-question.
"""
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI


def build_prompt(sub_question, expected_answer, passages_text):
    return f"""You are an expert evaluator checking whether retrieved passages contain enough information to answer a specific sub-question.

Given a sub-question, the expected answer, and retrieved passages, determine:
- FACT-GROUNDED: The passages contain the expected answer OR enough information to determine/infer it.
- NOT-FACT-GROUNDED: The passages do NOT contain the expected answer and there is no way to determine it.

The answer does NOT need to appear as an exact string. Paraphrases and implications count.

Sub-question: {sub_question}
Expected answer: {expected_answer}
Passages: {passages_text[:2000]}

Verdict (FACT-GROUNDED or NOT-FACT-GROUNDED):"""


def judge(client, sub_question, expected_answer, passages_text, model="gpt-4.1-mini"):
    prompt = build_prompt(sub_question, expected_answer, passages_text)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )
        text = resp.choices[0].message.content.strip().upper()
        if "NOT" in text[:30]:
            return 0
        elif "FACT" in text[:30]:
            return 1
        return 0
    except Exception as e:
        time.sleep(5)
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', default='results/2wiki_train_trajectories.json')
    parser.add_argument('--raw_data', default='raw_data/2wikimultihopqa/train_sample_20k.json')
    parser.add_argument('--output', default='results/2wiki_fact_grounded_train.jsonl')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    client = OpenAI()

    print("Loading trajectories...")
    with open(args.trajectories) as f:
        traj_data = json.load(f)
    results = traj_data['results']

    print("Loading raw data for evidences...")
    with open(args.raw_data) as f:
        raw = json.load(f)
    raw_lookup = {ex['_id']: ex for ex in raw}
    del raw

    if args.n:
        results = results[:args.n]
    print(f"Processing {len(results)} examples")

    out_f = open(args.output, 'w')
    total = 0
    grounded = 0

    for r in tqdm(results, desc="Fact-grounding"):
        qid = r['qid']
        traj = r.get('trajectory')
        if not traj:
            continue

        raw_ex = raw_lookup.get(qid)
        if not raw_ex:
            continue

        evidences = raw_ex.get('evidences', [])
        steps = traj.get('steps', [])

        # Accumulate passages across hops
        all_passages = []
        for step_idx, step in enumerate(steps):
            passages = step.get('retrieved_passages', [])
            all_passages.extend(passages)

            # Dedupe
            seen = set()
            unique = []
            for p in all_passages:
                if p['text'] not in seen:
                    seen.add(p['text'])
                    unique.append(p)
            all_passages = unique

            # Match to evidence if available
            if step_idx < len(evidences):
                ev = evidences[step_idx]
                entity, relation, answer = ev[0], ev[1], ev[2]
                sub_q = f"What is the {relation} of {entity}?"
            else:
                sub_q = step.get('query', '')
                answer = raw_ex.get('answer', '')

            # Format passages
            passages_text = "\n".join(
                f"[{i+1}] {p['title']}: {p['text'][:500]}"
                for i, p in enumerate(all_passages)
            )

            label = judge(client, sub_q, answer, passages_text)
            total += 1
            grounded += label

            example = {
                'qid': qid,
                'hop_number': step_idx + 1,
                'sub_question': sub_q,
                'target_answer': answer,
                'label': label,
                'num_passages': len(all_passages),
            }
            out_f.write(json.dumps(example) + '\n')

        if args.verbose and total % 1000 == 0:
            print(f"  {total} hops, {grounded/total*100:.1f}% grounded")

    out_f.close()
    print(f"\nTotal hops: {total}")
    print(f"Grounded: {grounded} ({grounded/total*100:.1f}%)")
    print(f"Not grounded: {total-grounded} ({(total-grounded)/total*100:.1f}%)")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
