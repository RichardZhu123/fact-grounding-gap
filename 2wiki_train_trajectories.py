#!/usr/bin/env python3
import json
import time
import argparse
from dataclasses import asdict
from tqdm import tqdm
from simple_multihop_qa import SimpleMultiHopQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='raw_data/2wikimultihopqa/train_sample_20k.json')
    parser.add_argument('--output', default='results/2wiki_train_trajectories.json')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    with open(args.input) as f:
        data = json.load(f)
    if args.n:
        data = data[:args.n]
    print(f"Processing {len(data)} examples")

    qa = SimpleMultiHopQA(corpus_name="2wikimultihopqa", model="gpt-4.1-mini", max_hops=4)

    results = []
    t0 = time.time()

    for ex in tqdm(data, desc="Self-Ask trajectories"):
        qid = ex['_id']
        question = ex['question']
        gold = ex['answer']

        try:
            traj = qa.answer(question, gold_answer=gold)
            result = {
                'qid': qid,
                'question': question,
                'gold': gold,
                'trajectory': asdict(traj),
            }
        except Exception as e:
            if args.verbose:
                print(f"  ERROR {qid}: {e}")
            result = {
                'qid': qid,
                'question': question,
                'gold': gold,
                'trajectory': None,
            }

        results.append(result)

        if args.verbose and len(results) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {len(results)}/{len(data)} done ({elapsed/60:.0f} min)")

        if len(results) % 1000 == 0:
            with open(args.output, 'w') as f:
                json.dump({'n': len(results), 'results': results}, f)

    output = {'n': len(results), 'results': results}
    with open(args.output, 'w') as f:
        json.dump(output, f)

    elapsed = time.time() - t0
    print(f"\nSaved {len(results)} trajectories to {args.output}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.0f} min)")


if __name__ == '__main__':
    main()
