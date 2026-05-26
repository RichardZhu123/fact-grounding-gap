#!/usr/bin/env python3
"""2WikiMQA intervention with retrained DeBERTa (domain-adapted)."""
import json
import sys
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, '.')
from simple_multihop_qa import SimpleMultiHopQA
from deberta_inference import DebertaPredictor
from query_reformulator import QueryReformulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--output', default='results/2wiki_retrained_deberta.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    with open('raw_data/2wikimultihopqa/dev.json') as f:
        data = json.load(f)
    if args.n:
        data = data[:args.n]
    print(f"Processing {len(data)} examples")

    print("Loading baseline results...")
    with open('results/2wiki_full.json') as f:
        baseline_data = json.load(f)
    baseline_lookup = {r['qid']: r for r in baseline_data['results']}

    qa = SimpleMultiHopQA(corpus_name="2wikimultihopqa", model="gpt-4.1-mini")
    predictor = DebertaPredictor('models/deberta_2wiki_fg/best_model')
    reformulator = QueryReformulator()

    results = []
    total_flags = 0
    total_hops = 0
    t0 = time.time()

    for ex in tqdm(data, desc="DeBERTa-retrained intervention"):
        qid = ex['_id']
        question = ex['question']
        gold = ex['answer']

        try:
            traj = qa.answer(question)

            flags = []
            interventions = {}
            for step in traj.steps:
                passage_text = ' '.join(p.get('text', '') for p in step.retrieved_passages)
                is_sufficient = predictor.predict(step.query, passage_text)
                flag = not (is_sufficient[0] == 'ANSWERABLE')
                flags.append(flag)
                total_hops += 1

                if flag:
                    total_flags += 1
                    new_query = reformulator.reformulate(question, step.query)
                    extra = qa.retrieve(new_query)
                    interventions[step.hop_number] = extra

            if interventions:
                traj = qa.answer_with_interventions(question, interventions=interventions, gold_answer=gold)

            pred = traj.final_answer
            if "ANSWER:" in pred:
                pred = pred.split("ANSWER:")[1].strip()
        except Exception as e:
            if args.verbose:
                print(f"  ERROR {qid}: {e}")
            pred = ""
            flags = []

        results.append({
            'qid': qid,
            'question': question,
            'gold': gold,
            'deberta_retrained': {'answer': pred, 'flags': flags},
            'baseline_answer': baseline_lookup.get(qid, {}).get('baseline', {}).get('answer', ''),
        })

        if args.verbose and len(results) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {len(results)}/{len(data)}, flags: {total_flags}/{total_hops} ({total_flags/max(total_hops,1)*100:.1f}%), {elapsed/60:.0f} min")

        if len(results) % 1000 == 0:
            with open(args.output, 'w') as f:
                json.dump({'n': len(results), 'results': results, 'total_flags': total_flags, 'total_hops': total_hops}, f)

    output = {
        'n': len(results),
        'total_flags': total_flags,
        'total_hops': total_hops,
        'flag_rate': total_flags / max(total_hops, 1),
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f)

    base_correct = 0
    deb_correct = 0
    for r in results:
        gold = r['gold'].lower().strip()
        bp = r['baseline_answer'].lower().strip()
        dp = r['deberta_retrained']['answer'].lower().strip()
        if gold in bp or bp in gold:
            base_correct += 1
        if gold in dp or dp in gold:
            deb_correct += 1

    n = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS (n={n})")
    print(f"{'='*60}")
    print(f"Flag rate: {total_flags}/{total_hops} ({total_flags/max(total_hops,1)*100:.1f}%)")
    print(f"Baseline:              {base_correct}/{n} = {base_correct/n*100:.1f}%")
    print(f"DeBERTa-retrained:     {deb_correct}/{n} = {deb_correct/n*100:.1f}%")
    print(f"Delta:                 {(deb_correct-base_correct)/n*100:+.1f}%")
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
