#!/usr/bin/env python3
"""
Section 4: Decomposition-guided QA with fact-presence-triggered re-retrieval.
v4: Entity-only prompt + passage capping + checkpointing/resume.

For each MuSiQue dev question, runs 4 conditions:
  - baseline:  per-step retrieval, no fact-checking
  - always:    reformulate + re-retrieve at every step
  - deberta:   reformulate when DeBERTa flags NOT-ANSWERABLE
  - oracle:    reformulate when existing LLM judge labels say NOT-ANSWERABLE

Passage capping: total accumulated passages capped at --max_passages.
Lowest BM25-scored dropped when over limit.
Checkpoints written every --checkpoint_every questions; --resume picks up where it left off.
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


def dedupe_and_cap(passages, max_passages=None):
    """Dedupe by text, then cap to max_passages keeping highest BM25 score."""
    seen, out = set(), []
    for p in passages:
        t = p.get('text', '')
        if t and t not in seen:
            seen.add(t)
            out.append(p)
    if max_passages and len(out) > max_passages:
        out.sort(key=lambda p: p.get('score', 0), reverse=True)
        out = out[:max_passages]
    return out


def deberta_format_input(passages, max_chars=4000):
    return format_passages(passages, max_chars=max_chars)


def clean_answer(s):
    """Strip markdown, quotes, trailing punctuation."""
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


def run_one_condition(retriever, sub_questions, intervention_flags, reformulator,
                      original_question, max_passages=None):
    accumulated = []
    for sq, flag in zip(sub_questions, intervention_flags):
        base_passages = retriever.retrieve(sq, k=3)
        accumulated.extend(base_passages)
        if flag:
            try:
                new_q = reformulator.reformulate(original_question, sq)
                extra = retriever.retrieve(new_q, k=3)
                accumulated.extend(extra)
            except Exception:
                pass
    return dedupe_and_cap(accumulated, max_passages=max_passages)


def save_checkpoint(out_path, summary):
    """Atomic write: tmp then rename."""
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


def compute_summary(examples, total_steps, flag_counts, passage_counts, max_passages):
    n = len(examples)
    correct = {'baseline': 0, 'always': 0, 'deberta': 0, 'oracle': 0}
    by_hopcount = {}
    for ex in examples:
        for c in correct:
            if ex[c]['correct']:
                correct[c] += 1
        h = ex['n_hops']
        s = by_hopcount.setdefault(h, {'n': 0, 'baseline': 0, 'always': 0, 'deberta': 0, 'oracle': 0})
        s['n'] += 1
        for c in correct:
            if ex[c]['correct']:
                s[c] += 1

    return {
        'n': n,
        'total_steps': total_steps,
        'max_passages': max_passages,
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
    parser.add_argument("--output", default="results/section4_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deberta_threshold", type=float, default=0.5)
    parser.add_argument("--max_passages", type=int, default=12)
    parser.add_argument("--checkpoint_every", type=int, default=200,
                        help="Save partial results every N questions")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping completed qids")
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

    out = []
    flag_counts = {'always': 0, 'deberta': 0, 'oracle': 0}
    passage_counts = {'baseline': [], 'always': [], 'deberta': [], 'oracle': []}
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
    print(f"Running on {len(qids_to_run)} remaining questions (max_passages={args.max_passages})\n")

    for i, qid in enumerate(tqdm(qids_to_run, desc="Section 4")):
        d = raw[qid]
        original_question = d['question']
        gold = d['answer']
        sub_questions = subq_lookup[qid].get('natural_sub_questions', [])
        if not sub_questions:
            continue
        n_hops = len(sub_questions)
        oracle_for_q = oracle.get(qid, {})

        baseline_passages = run_one_condition(
            retriever, sub_questions, [False] * n_hops, reformulator,
            original_question, max_passages=args.max_passages
        )

        always_flags = [True] * n_hops
        oracle_flags = []
        deberta_flags = []
        for sq in sub_questions:
            total_steps += 1

            oracle_label = oracle_for_q.get(sq)
            ofg = 1 if oracle_label == 0 else 0
            oracle_flags.append(bool(ofg))
            if ofg:
                flag_counts['oracle'] += 1

            try:
                _, prob = deberta.predict(sq, deberta_format_input(baseline_passages))
                dfg = 1 if prob < args.deberta_threshold else 0
            except Exception:
                dfg = 0
            deberta_flags.append(bool(dfg))
            if dfg:
                flag_counts['deberta'] += 1

            flag_counts['always'] += 1

        always_passages = run_one_condition(retriever, sub_questions, always_flags, reformulator, original_question, max_passages=args.max_passages)
        deberta_passages = run_one_condition(retriever, sub_questions, deberta_flags, reformulator, original_question, max_passages=args.max_passages)
        oracle_passages = run_one_condition(retriever, sub_questions, oracle_flags, reformulator, original_question, max_passages=args.max_passages)

        passage_counts['baseline'].append(len(baseline_passages))
        passage_counts['always'].append(len(always_passages))
        passage_counts['deberta'].append(len(deberta_passages))
        passage_counts['oracle'].append(len(oracle_passages))

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
            'always': {'answer': ans_always, 'correct': j_alw, 'n_passages': len(always_passages)},
            'deberta': {'answer': ans_deberta, 'correct': j_deb, 'flags': deberta_flags, 'n_passages': len(deberta_passages)},
            'oracle': {'answer': ans_oracle, 'correct': j_ora, 'flags': oracle_flags, 'n_passages': len(oracle_passages)},
        })

        if (i + 1) % args.checkpoint_every == 0:
            summary = compute_summary(out, total_steps, flag_counts, passage_counts, args.max_passages)
            save_checkpoint(args.output, summary)

    # Final save + report
    summary = compute_summary(out, total_steps, flag_counts, passage_counts, args.max_passages)
    save_checkpoint(args.output, summary)

    n = len(out)
    correct = summary['correct_counts']
    print(f"\n{'='*60}")
    print(f"SECTION 4 RESULTS (n={n}, max_passages={args.max_passages})")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    if total_steps > 0:
        for k in ['always', 'deberta', 'oracle']:
            print(f"  {k:8s} flag rate: {flag_counts[k]}/{total_steps} = {flag_counts[k]/total_steps*100:.1f}%")
    print()
    print("Average passage counts:")
    for k, v in summary['avg_passage_counts'].items():
        print(f"  {k:10s}: {v:.1f}")
    print()
    print("Overall accuracy:")
    for cond in ['baseline', 'always', 'deberta', 'oracle']:
        if n > 0:
            print(f"  {cond:10s}: {correct[cond]}/{n} = {correct[cond]/n*100:.1f}%")
    print()
    print("Stratified by hop count:")
    for h in sorted(summary['stratified']):
        s = summary['stratified'][h]
        print(f"  {h}-hop (n={s['n']}):")
        for cond in ['baseline', 'always', 'deberta', 'oracle']:
            print(f"    {cond:10s}: {s[cond]}/{s['n']} = {s[cond]/s['n']*100:.1f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
