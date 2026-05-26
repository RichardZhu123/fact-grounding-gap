#!/usr/bin/env python3
"""
Section 4: Iterative per-hop extraction with DeBERTa verification gate.
Each hop: retrieve → extract intermediate answer → DeBERTa verify → re-retrieve if needed.
Final synthesis from verified intermediate answers.
"""

import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import CrossEncoder

from deberta_inference import DebertaPredictor
from query_reformulator import QueryReformulator
from es_retriever import ESRetriever


HOP_PROMPT = """Answer ONLY this specific question using the passages below.
Give a SHORT answer (just the entity or fact, no explanation).

Passages:
{passages}

Question: {question}

Answer:"""


SYNTHESIS_PROMPT = """Given the following intermediate answers to sub-questions, answer the original question.

{intermediate_qa}

Original question: {original_question}

Give a SHORT answer (just the entity or fact, no explanation).
Final answer:"""


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


def rerank_passages(reranker, query, passages, top_k=3):
    if not passages:
        return passages
    pairs = [(query, p.get('text', '')) for p in passages]
    scores = reranker.predict(pairs)
    for p, s in zip(passages, scores):
        p['rerank_score'] = float(s)
    ranked = sorted(passages, key=lambda p: p.get('rerank_score', 0), reverse=True)
    return ranked[:top_k]


def answer_hop(client, question, passages, model="gpt-4.1-mini"):
    """Answer a single sub-question from focused passages."""
    prompt = HOP_PROMPT.format(
        passages=format_passages(passages),
        question=question,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    return clean_answer(resp.choices[0].message.content.strip())


def synthesize_answer(client, original_question, sub_questions, intermediate_answers, model="gpt-4.1-mini"):
    """Final synthesis from verified intermediate answers."""
    qa_pairs = "\n".join(
        f"Q{i+1}: {sq}\nA{i+1}: {ans}"
        for i, (sq, ans) in enumerate(zip(sub_questions, intermediate_answers))
    )
    prompt = SYNTHESIS_PROMPT.format(
        intermediate_qa=qa_pairs,
        original_question=original_question,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    output = resp.choices[0].message.content.strip()

    if "Final answer:" in output:
        tail = output.split("Final answer:", 1)[1].strip()
        for line in tail.split("\n"):
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


def run_iterative_baseline(client, retriever, reranker, sub_questions):
    """Baseline: per-hop extraction with light reranking, no verification gate."""
    intermediate_answers = []
    accumulated_passages = []

    for sq in sub_questions:
        candidates = retriever.retrieve(sq, k=10)
        hop_passages = rerank_passages(reranker, sq, candidates, top_k=3)
        accumulated_passages.extend(hop_passages)
        answer = answer_hop(client, sq, dedupe_passages(accumulated_passages))
        intermediate_answers.append(answer)

    return intermediate_answers, dedupe_passages(accumulated_passages)


def run_iterative_always(client, retriever, reranker, reformulator, sub_questions,
                         original_question):
    """Always: per-hop extraction with deep reranking at every hop."""
    intermediate_answers = []
    accumulated_passages = []

    for sq in sub_questions:
        # Deep retrieval: k=20 original + k=10 reformulated
        candidates = retriever.retrieve(sq, k=20)
        try:
            new_q = reformulator.reformulate(original_question, sq)
            extra = retriever.retrieve(new_q, k=10)
            seen = {p['text'] for p in candidates}
            for p in extra:
                if p['text'] not in seen:
                    seen.add(p['text'])
                    candidates.append(p)
        except Exception:
            pass
        hop_passages = rerank_passages(reranker, sq, candidates, top_k=3)
        accumulated_passages.extend(hop_passages)
        answer = answer_hop(client, sq, dedupe_passages(accumulated_passages))
        intermediate_answers.append(answer)

    return intermediate_answers, dedupe_passages(accumulated_passages)


def run_iterative_deberta(client, retriever, reranker, reformulator, deberta,
                          sub_questions, original_question, threshold=0.5):
    """
    DeBERTa-gated: per-hop extraction with verification.
    Light rerank initially. If DeBERTa flags NOT-ANSWERABLE, deep rerank + re-answer.
    """
    intermediate_answers = []
    accumulated_passages = []
    flags = []

    for sq in sub_questions:
        # Light retrieval first
        candidates = retriever.retrieve(sq, k=10)
        hop_passages = rerank_passages(reranker, sq, candidates, top_k=3)
        accumulated_passages.extend(hop_passages)
        current_passages = dedupe_passages(accumulated_passages)

        # DeBERTa gate: check fact presence
        passages_str = deberta_format_input(current_passages)
        try:
            _, prob = deberta.predict(sq, passages_str)
            flagged = prob < threshold
        except Exception:
            flagged = False

        flags.append(flagged)

        if flagged:
            # Deep retrieval: k=20 + reformulated k=10, rerank
            deep_candidates = retriever.retrieve(sq, k=20)
            try:
                new_q = reformulator.reformulate(original_question, sq)
                extra = retriever.retrieve(new_q, k=10)
                seen = {p['text'] for p in deep_candidates}
                for p in extra:
                    if p['text'] not in seen:
                        seen.add(p['text'])
                        deep_candidates.append(p)
            except Exception:
                pass
            extra_passages = rerank_passages(reranker, sq, deep_candidates, top_k=3)
            accumulated_passages.extend(extra_passages)
            current_passages = dedupe_passages(accumulated_passages)

        # Answer this hop with focused extraction
        answer = answer_hop(client, sq, current_passages)
        intermediate_answers.append(answer)

    return intermediate_answers, dedupe_passages(accumulated_passages), flags


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
    correct = {'baseline_iter': 0, 'always_iter': 0, 'deberta_iter': 0}
    by_hopcount = {}
    for ex in examples:
        for c in correct:
            if ex[c]['correct']:
                correct[c] += 1
        h = ex['n_hops']
        s = by_hopcount.setdefault(h, {'n': 0})
        for c in correct:
            s.setdefault(c, 0)
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
    parser.add_argument("--deberta_dir", default="models/deberta_fg_v2_best")
    parser.add_argument("--output", default="results/section4_iterative.json")
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
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("  All components loaded")

    print("Loading data...")
    raw = {}
    with open(args.raw_data) as f:
        for line in f:
            d = json.loads(line)
            raw[d['id']] = d

    with open(args.subquestions) as f:
        subq_lookup = json.load(f)

    qids = [q for q in raw if q in subq_lookup]
    if args.limit:
        qids = qids[:args.limit]

    # Resume logic
    out = []
    flag_counts = {'deberta': 0, 'total_hops': 0}
    passage_counts = {'baseline_iter': [], 'always_iter': [], 'deberta_iter': []}
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
            print(f"Resumed: {len(completed_qids)} already done.")

    qids_to_run = [q for q in qids if q not in completed_qids]
    print(f"Running iterative extraction on {len(qids_to_run)} questions\n")

    for i, qid in enumerate(tqdm(qids_to_run, desc="Iterative")):
        d_raw = raw[qid]
        original_question = d_raw['question']
        gold = d_raw['answer']
        sub_questions = subq_lookup[qid].get('natural_sub_questions', [])
        if not sub_questions:
            continue
        n_hops = len(sub_questions)
        total_steps += n_hops

        try:
            # Baseline: per-hop extraction, light rerank, no gate
            base_answers, base_passages = run_iterative_baseline(
                client, retriever, reranker, sub_questions
            )
            base_final = synthesize_answer(client, original_question, sub_questions, base_answers)

            # Always: per-hop extraction, deep rerank every hop
            always_answers, always_passages = run_iterative_always(
                client, retriever, reranker, reformulator, sub_questions, original_question
            )
            always_final = synthesize_answer(client, original_question, sub_questions, always_answers)

            # DeBERTa: per-hop extraction with verification gate
            deb_answers, deb_passages, deb_flags = run_iterative_deberta(
                client, retriever, reranker, reformulator, deberta,
                sub_questions, original_question, threshold=args.deberta_threshold
            )
            deb_final = synthesize_answer(client, original_question, sub_questions, deb_answers)

        except Exception as e:
            print(f"QA failed for {qid}: {e}")
            continue

        flag_counts['deberta'] += sum(deb_flags)
        flag_counts['total_hops'] += n_hops

        try:
            j_base = llm_judge(client, original_question, gold, base_final)
            j_always = llm_judge(client, original_question, gold, always_final)
            j_deb = llm_judge(client, original_question, gold, deb_final)
        except Exception as e:
            print(f"Judge failed for {qid}: {e}")
            continue

        passage_counts['baseline_iter'].append(len(base_passages))
        passage_counts['always_iter'].append(len(always_passages))
        passage_counts['deberta_iter'].append(len(deb_passages))

        out.append({
            'qid': qid,
            'question': original_question,
            'gold': gold,
            'n_hops': n_hops,
            'baseline_iter': {
                'answer': base_final, 'correct': j_base,
                'intermediate': base_answers, 'n_passages': len(base_passages),
            },
            'always_iter': {
                'answer': always_final, 'correct': j_always,
                'intermediate': always_answers, 'n_passages': len(always_passages),
            },
            'deberta_iter': {
                'answer': deb_final, 'correct': j_deb,
                'intermediate': deb_answers, 'flags': deb_flags, 'n_passages': len(deb_passages),
            },
        })

        if (i + 1) % args.checkpoint_every == 0:
            summary = compute_summary(out, total_steps, flag_counts, passage_counts)
            save_checkpoint(args.output, summary)

    # Final save + report
    summary = compute_summary(out, total_steps, flag_counts, passage_counts)
    save_checkpoint(args.output, summary)

    n = len(out)
    correct = summary['correct_counts']
    print(f"\n{'='*60}")
    print(f"ITERATIVE EXTRACTION RESULTS (n={n})")
    print(f"{'='*60}")
    if flag_counts['total_hops'] > 0:
        print(f"DeBERTa flag rate: {flag_counts['deberta']}/{flag_counts['total_hops']} = {flag_counts['deberta']/flag_counts['total_hops']*100:.1f}%")
    print()
    print("Average passage counts:")
    for k, v in summary['avg_passage_counts'].items():
        print(f"  {k:16s}: {v:.1f}")
    print()
    print("Overall accuracy:")
    for cond in ['baseline_iter', 'always_iter', 'deberta_iter']:
        if n > 0:
            print(f"  {cond:16s}: {correct[cond]}/{n} = {correct[cond]/n*100:.1f}%")
    print()
    print("Stratified by hop count:")
    for h in sorted(summary['stratified']):
        s = summary['stratified'][h]
        print(f"  {h}-hop (n={s['n']}):")
        for cond in ['baseline_iter', 'always_iter', 'deberta_iter']:
            print(f"    {cond:16s}: {s.get(cond, 0)}/{s['n']} = {s.get(cond, 0)/s['n']*100:.1f}%")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
