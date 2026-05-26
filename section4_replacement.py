#!/usr/bin/env python3
"""
Section 4 ablation: Zero-sum passage REPLACEMENT strategy.
v2: With checkpointing and resume support.

Each hop contributes exactly 3 passages. Flagged hops swap originals for reformulated.
Same budget across all conditions — only hop selection differs.
"""

import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI

from deberta_inference import DebertaPredictor
from query_reformulator import QueryReformulator
from es_retriever import ESRetriever


COT_PROMPT = """Answer the question step by step using ONLY the passages below.

Passages:
{passages}

{steps}

Original question: {original_question}

Reason through each step, then write your final answer as a SHORT entity or fact (no full sentences).
Final answer:"""


JUDGE_PROMPT = """You are evaluating a question-answering system.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer semantically equivalent to the gold answer? Accept paraphrases,
aliases, and partial matches that capture the key entity/fact. Reject answers that name
a different entity or miss the key fact.

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


def deberta_format_input(passages, max_chars=4000):
    return format_passages(passages, max_chars=max_chars)


def clean_answer(s):
    if not s:
        return ""
    s = s.strip()
    while s and s[0] in ('*', '_', '"', "'", '`', '#', '-', '>'):
        s = s[1:].strip()
    while s and s[-1] in ('*', '_', '"', "'", '`', '.', ',', ';', ':'):
        s = s[:-1].strip()
    return s


def run_cot_qa(client, original_question, sub_questions, passages, model="gpt-4.1-mini"):
    steps_text = "\n".join(f"Step {i+1}: {sq}" for i, sq in enumerate(sub_questions))
    prompt = COT_PROMPT.format(
        passages=format_passages(passages),
        steps=steps_text,
        original_question=original_question,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    output = resp.choices[0].message.content.strip()

    if "Final answer:" in output:
        tail = output.split("Final answer:", 1)[1].strip()
        for line in tail.split("\n"):
            cleaned = clean_answer(line)
            if cleaned and len(cleaned) > 1:
                return cleaned

    for line in reversed([l for l in output.split("\n") if l.strip()]):
        cleaned = clean_answer(line)
        if cleaned and len(cleaned) > 1:
            return cleaned
    return clean_answer(output) or output


def llm_judge(client, question, gold, pred, model="gpt-4.1-mini"):
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    return "YES" in resp.choices[0].message.content.strip().upper()


def build_passages_replace(retriever, sub_questions, intervention_flags, reformulator,
                           original_question):
    """
    Zero-sum replacement: each hop contributes exactly 3 passages.
    Flagged hops: swap originals with reformulated passages.
    Unflagged hops: keep originals.
    """
    accumulated = []
    for sq, flag in zip(sub_questions, intervention_flags):
        if flag:
            try:
                new_q = reformulator.reformulate(original_question, sq)
                hop_passages = retriever.retrieve(new_q, k=3)
            except Exception:
                hop_passages = retriever.retrieve(sq, k=3)
        else:
            hop_passages = retriever.retrieve(sq, k=3)
        accumulated.extend(hop_passages)

    # Dedupe by text
    seen, deduped = set(), []
    for p in accumulated:
        t = p.get('text', '')
        if t and t not in seen:
            seen.add(t)
            deduped.append(p)
    return deduped


def save_checkpoint(out_path, summary):
    tmp = out_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, out_path)


def load_checkpoint(out_path):
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path) as f:
            return json.load(f)
    except Exception:
        return None


def compute_summary(examples, total_steps, flag_counts, passage_counts):
    n = len(examples)
    correct = {'baseline': 0, 'always_replace': 0, 'deberta_replace': 0, 'oracle_replace': 0}
    by_hopcount = {}
    for ex in examples:
        for c in correct:
            if ex[c]['correct']:
                correct[c] += 1
        h = ex['n_hops']
        s = by_hopcount.setdefault(h, {'n': 0, 'baseline': 0, 'always_replace': 0, 'deberta_replace': 0, 'oracle_replace': 0})
        s['n'] += 1
        for c in correct:
            if ex[c]['correct']:
                s[c] += 1

    return {
        'n': n,
        'total_steps': total_steps,
        'flag_counts': flag_counts,
        'correct_counts': correct,
        'accuracy': {k: correct[k]/n for k in correct} if n else {},
        'avg_passage_counts': {k: sum(v)/len(v) if v else 0 for k, v in passage_counts.items()},
        'stratified': by_hopcount,
        'examples': examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="raw_data/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--subquestions", default="results/musique_dev_subquestions.json")
    parser.add_argument("--fact_grounded", default="results/fact_grounded_final_dev.jsonl")
    parser.add_argument("--deberta_dir", default="models/deberta_fg_v2_best")
    parser.add_argument("--output", default="results/section4_replacement.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deberta_threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("Loading components...")
    client = OpenAI()
    deberta = DebertaPredictor(args.deberta_dir)
    reformulator = QueryReformulator()
    retriever = ESRetriever()

    print("Loading data...")
    raw = {}
    with open(args.raw_data) as f:
        for line in f:
            d = json.loads(line)
            raw[d['id']] = d

    with open(args.subquestions) as f:
        subq_lookup = json.load(f)

    oracle = {}
    with open(args.fact_grounded) as f:
        for line in f:
            e = json.loads(line)
            if e['label'] == 'UNKNOWN':
                continue
            oracle.setdefault(e['qid'], {})[e['sub_question']] = e['label_binary']

    qids = [q for q in raw if q in subq_lookup]
    if args.limit:
        qids = qids[:args.limit]

    # Resume logic
    out = []
    flag_counts = {'always': 0, 'deberta': 0, 'oracle': 0}
    passage_counts = {'baseline': [], 'always_replace': [], 'deberta_replace': [], 'oracle_replace': []}
    total_steps = 0
    completed_qids = set()

    if args.resume:
        existing = load_checkpoint(args.output)
        if existing:
            out = existing.get('examples', [])
            completed_qids = {ex['qid'] for ex in out}
            flag_counts = existing.get('flag_counts', flag_counts)
            total_steps = existing.get('total_steps', 0)
            for ex in out:
                for c in passage_counts:
                    passage_counts[c].append(ex[c].get('n_passages', 0))
            print(f"Resumed: {len(completed_qids)} already done, skipping those.")

    qids_to_run = [q for q in qids if q not in completed_qids]
    print(f"Running replacement experiment on {len(qids_to_run)} questions\n")

    for i, qid in enumerate(tqdm(qids_to_run, desc="Replacement")):
        d = raw[qid]
        original_question = d['question']
        gold = d['answer']
        sub_questions = subq_lookup[qid].get('natural_sub_questions', [])
        if not sub_questions:
            continue
        n_hops = len(sub_questions)
        oracle_for_q = oracle.get(qid, {})

        # Baseline: no replacement
        baseline_passages = build_passages_replace(
            retriever, sub_questions, [False] * n_hops, reformulator, original_question
        )

        # Build flags
        always_flags = [True] * n_hops
        oracle_flags = []
        deberta_flags = []
        passages_str = deberta_format_input(baseline_passages)

        for sq in sub_questions:
            total_steps += 1

            oracle_label = oracle_for_q.get(sq)
            ofg = 1 if oracle_label == 0 else 0
            oracle_flags.append(bool(ofg))
            if ofg:
                flag_counts['oracle'] += 1

            try:
                _, prob = deberta.predict(sq, passages_str)
                dfg = 1 if prob < args.deberta_threshold else 0
            except Exception:
                dfg = 0
            deberta_flags.append(bool(dfg))
            if dfg:
                flag_counts['deberta'] += 1

            flag_counts['always'] += 1

        # Build replacement passage sets
        always_passages = build_passages_replace(
            retriever, sub_questions, always_flags, reformulator, original_question
        )
        deberta_passages = build_passages_replace(
            retriever, sub_questions, deberta_flags, reformulator, original_question
        )
        oracle_passages = build_passages_replace(
            retriever, sub_questions, oracle_flags, reformulator, original_question
        )

        passage_counts['baseline'].append(len(baseline_passages))
        passage_counts['always_replace'].append(len(always_passages))
        passage_counts['deberta_replace'].append(len(deberta_passages))
        passage_counts['oracle_replace'].append(len(oracle_passages))

        try:
            ans_baseline = run_cot_qa(client, original_question, sub_questions, baseline_passages)
            ans_always = run_cot_qa(client, original_question, sub_questions, always_passages)
            ans_deberta = run_cot_qa(client, original_question, sub_questions, deberta_passages)
            ans_oracle = run_cot_qa(client, original_question, sub_questions, oracle_passages)
        except Exception as e:
            print(f"QA failed for {qid}: {e}")
            continue

        try:
            j_base = llm_judge(client, original_question, gold, ans_baseline)
            j_alw = llm_judge(client, original_question, gold, ans_always)
            j_deb = llm_judge(client, original_question, gold, ans_deberta)
            j_ora = llm_judge(client, original_question, gold, ans_oracle)
        except Exception as e:
            print(f"Judge failed for {qid}: {e}")
            continue

        out.append({
            'qid': qid,
            'question': original_question,
            'gold': gold,
            'n_hops': n_hops,
            'baseline': {'answer': ans_baseline, 'correct': j_base, 'n_passages': len(baseline_passages)},
            'always_replace': {'answer': ans_always, 'correct': j_alw, 'n_passages': len(always_passages)},
            'deberta_replace': {'answer': ans_deberta, 'correct': j_deb, 'flags': deberta_flags, 'n_passages': len(deberta_passages)},
            'oracle_replace': {'answer': ans_oracle, 'correct': j_ora, 'flags': oracle_flags, 'n_passages': len(oracle_passages)},
        })

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0:
            summary = compute_summary(out, total_steps, flag_counts, passage_counts)
            save_checkpoint(args.output, summary)

    # Final save + report
    summary = compute_summary(out, total_steps, flag_counts, passage_counts)
    save_checkpoint(args.output, summary)

    n = len(out)
    correct = summary['correct_counts']
    print(f"\n{'='*60}")
    print(f"REPLACEMENT EXPERIMENT RESULTS (n={n})")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    if total_steps > 0:
        for k in ['always', 'deberta', 'oracle']:
            print(f"  {k:8s} flag rate: {flag_counts[k]}/{total_steps} = {flag_counts[k]/total_steps*100:.1f}%")
    print()
    print("Average passage counts:")
    for k, v in summary['avg_passage_counts'].items():
        print(f"  {k:16s}: {v:.1f}")
    print()
    print("Overall accuracy:")
    for cond in ['baseline', 'always_replace', 'deberta_replace', 'oracle_replace']:
        if n > 0:
            print(f"  {cond:16s}: {correct[cond]}/{n} = {correct[cond]/n*100:.1f}%")
    print()
    print("Stratified by hop count:")
    for h in sorted(summary['stratified']):
        s = summary['stratified'][h]
        print(f"  {h}-hop (n={s['n']}):")
        for cond in ['baseline', 'always_replace', 'deberta_replace', 'oracle_replace']:
            print(f"    {cond:16s}: {s[cond]}/{s['n']} = {s[cond]/s['n']*100:.1f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
