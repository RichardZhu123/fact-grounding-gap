#!/usr/bin/env python3
"""
Part 4: Re-retrieval Intervention Experiment
Runs 4 conditions: Baseline / Always / DeBERTa-triggered / Oracle-triggered
Reports LLM judge accuracy for each.
"""

import json
import argparse
from tqdm import tqdm
from openai import OpenAI

from deberta_inference import DebertaPredictor
from query_reformulator import QueryReformulator
from es_retriever import ESRetriever


QA_PROMPT = """Answer the following question using ONLY the information in the passages below.
Give a SHORT, direct answer (just the entity/fact, no explanation).

Passages:
{passages}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are evaluating a question-answering system.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer semantically equivalent to the gold answer? Accept paraphrases,
aliases, and partial matches that capture the key entity/fact. Reject answers that name
a different entity or miss the key fact.

Reply with ONLY "YES" or "NO"."""


def format_passages(passages, max_chars=4000):
    parts, total = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"[{i}] {p.get('title','')}: {p.get('text','')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)


def dedupe_passages(passages):
    seen, out = set(), []
    for p in passages:
        t = p.get('text', '')
        if t and t not in seen:
            seen.add(t)
            out.append(p)
    return out


def run_qa(client, question, passages, model="gpt-4.1-mini"):
    prompt = QA_PROMPT.format(passages=format_passages(passages), question=question)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def llm_judge(client, question, gold, pred, model="gpt-4.1-mini"):
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    return "YES" in resp.choices[0].message.content.strip().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_results", default="results/full_run_all_4.1.json")
    parser.add_argument("--fact_grounded", default="results/fact_grounded_final_dev.jsonl")
    parser.add_argument("--subquestions", default="results/musique_dev_subquestions.json")
    parser.add_argument("--deberta_dir", default="models/deberta_fg_v2_best")
    parser.add_argument("--output", default="results/re_retrieval_results.json")
    parser.add_argument("--limit", type=int, default=None, help="Test on N questions only")
    parser.add_argument("--deberta_threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading components...")
    client = OpenAI()
    deberta = DebertaPredictor(args.deberta_dir)
    reformulator = QueryReformulator()
    retriever = ESRetriever()

    print("Loading data...")
    with open(args.qa_results) as f:
        qa = json.load(f)

    with open(args.subquestions) as f:
        subq_lookup = json.load(f)

    # Load oracle labels: qid -> {sub_question -> label_binary}
    oracle = {}
    with open(args.fact_grounded) as f:
        for line in f:
            e = json.loads(line)
            if e['label'] == 'UNKNOWN':
                continue
            oracle.setdefault(e['qid'], {})[e['sub_question']] = e['label_binary']

    results_list = qa['results']
    if args.limit:
        results_list = results_list[:args.limit]

    print(f"Running 4-condition experiment on {len(results_list)} questions\n")

    out = []
    correct = {'baseline': 0, 'always': 0, 'deberta': 0, 'oracle': 0}
    deberta_flags = 0
    oracle_flags = 0
    total_hops = 0

    for r in tqdm(results_list, desc="Re-retrieval"):
        qid = r.get('qid', '')
        question = r.get('question', '')
        gold = r.get('gold_answer', '')
        if not qid or qid not in subq_lookup:
            continue

        # Original accumulated passages
        seen = set()
        orig_passages = []
        for step in r['trajectory']['steps']:
            for p in step.get('retrieved_passages', []):
                t = p.get('text', '')
                if t and t not in seen:
                    seen.add(t)
                    orig_passages.append(p)

        subq_data = subq_lookup[qid]
        sub_questions = subq_data.get('natural_sub_questions', [])

        # For each condition, build augmented passage set
        always_passages = list(orig_passages)
        deberta_passages = list(orig_passages)
        oracle_passages = list(orig_passages)

        # Format orig_passages once for DeBERTa input
        orig_passages_str = format_passages(orig_passages, max_chars=4000)

        for sub_q in sub_questions:
            total_hops += 1

            # Always: reformulate every hop
            try:
                new_q = reformulator.reformulate(question, sub_q, orig_passages)
                new_p = retriever.retrieve(new_q, k=3)
                always_passages.extend(new_p)
            except Exception:
                pass

            # DeBERTa: only if model says NOT-ANSWERABLE
            try:
                label, prob = deberta.predict(sub_q, orig_passages_str)
                if prob < args.deberta_threshold:
                    deberta_flags += 1
                    new_q = reformulator.reformulate(question, sub_q, orig_passages)
                    new_p = retriever.retrieve(new_q, k=3)
                    deberta_passages.extend(new_p)
            except Exception:
                pass

            # Oracle: only if LLM judge says NOT-ANSWERABLE
            if oracle.get(qid, {}).get(sub_q) == 0:
                oracle_flags += 1
                try:
                    new_q = reformulator.reformulate(question, sub_q, orig_passages)
                    new_p = retriever.retrieve(new_q, k=3)
                    oracle_passages.extend(new_p)
                except Exception:
                    pass

        # Dedupe
        always_passages = dedupe_passages(always_passages)
        deberta_passages = dedupe_passages(deberta_passages)
        oracle_passages = dedupe_passages(oracle_passages)

        # Run QA for each condition
        try:
            ans_baseline = r.get('predicted_answer', '')
            ans_always = run_qa(client, question, always_passages)
            ans_deberta = run_qa(client, question, deberta_passages)
            ans_oracle = run_qa(client, question, oracle_passages)
        except Exception as e:
            print(f"QA failed for {qid}: {e}")
            continue

        # Judge each
        try:
            j_base = llm_judge(client, question, gold, ans_baseline)
            j_alw = llm_judge(client, question, gold, ans_always)
            j_deb = llm_judge(client, question, gold, ans_deberta)
            j_ora = llm_judge(client, question, gold, ans_oracle)
        except Exception as e:
            print(f"Judge failed for {qid}: {e}")
            continue

        if j_base: correct['baseline'] += 1
        if j_alw:  correct['always'] += 1
        if j_deb:  correct['deberta'] += 1
        if j_ora:  correct['oracle'] += 1

        out.append({
            'qid': qid,
            'question': question,
            'gold': gold,
            'baseline': {'answer': ans_baseline, 'correct': j_base},
            'always': {'answer': ans_always, 'correct': j_alw},
            'deberta': {'answer': ans_deberta, 'correct': j_deb},
            'oracle': {'answer': ans_oracle, 'correct': j_ora},
        })

    # Final report
    n = len(out)
    print(f"\n{'='*60}")
    print(f"4-CONDITION RE-RETRIEVAL RESULTS (n={n})")
    print(f"{'='*60}")
    print(f"Total hops evaluated: {total_hops}")
    if total_hops > 0:
        print(f"DeBERTa flagged: {deberta_flags} ({deberta_flags/total_hops*100:.1f}%)")
        print(f"Oracle flagged:  {oracle_flags} ({oracle_flags/total_hops*100:.1f}%)")
    print()
    for cond in ['baseline', 'always', 'deberta', 'oracle']:
        acc = correct[cond] / n if n else 0
        print(f"  {cond:12s}: {correct[cond]}/{n} = {acc*100:.1f}%")
    print()

    summary = {
        'n': n,
        'total_hops': total_hops,
        'deberta_flags': deberta_flags,
        'oracle_flags': oracle_flags,
        'accuracy': {k: correct[k]/n for k in correct} if n else {},
        'correct_counts': correct,
        'examples': out,
    }
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
