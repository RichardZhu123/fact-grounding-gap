#!/usr/bin/env python3
"""
Iterative re-retrieval experiment.
For each flagged hop, re-retrieve up to N rounds, replacing passages each time.
Track accuracy per round, split by retrieval failure vs extraction failure.
"""
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from deberta_inference import DebertaPredictor
from query_reformulator import QueryReformulator
from es_retriever import ESRetriever

JUDGE_PROMPT = """You are evaluating a question-answering system.
Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
Is the predicted answer semantically equivalent to the gold answer? Accept paraphrases,
aliases, and partial matches that capture the key entity/fact. Reject answers that name
a different entity or miss the key fact.
Reply with ONLY "YES" or "NO"."""

COT_PROMPT = """Answer the question step by step using ONLY the passages below.
Passages:
{passages}
{steps}
Original question: {original_question}
Reason through each step, then write your final answer as a SHORT entity or fact (no full sentences).
Final answer:"""

def format_passages(passages, max_chars=6000):
    parts, total = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"[{i}] {p.get('title','')}: {p.get('text','')}"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n".join(parts)

def clean_answer(s):
    if not s:
        return ""
    s = s.strip()
    while s and s[0] in ('*', '_', '"', "'", '`', '#', '-', '>'):
        s = s[1:].strip()
    while s and s[-1] in ('*', '_', '"', "'", '`', '.', ',', ';', ':'):
        s = s[:-1].strip()
    return s

def run_qa(client, original_question, sub_questions, passages):
    steps_text = "\n".join(f"Step {i+1}: {sq}" for i, sq in enumerate(sub_questions))
    prompt = COT_PROMPT.format(
        passages=format_passages(passages),
        steps=steps_text,
        original_question=original_question,
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=400,
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

def llm_judge(client, question, gold, pred):
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=5,
    )
    return "YES" in resp.choices[0].message.content.strip().upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="raw_data/musique/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--subquestions", default="results/musique_dev_subquestions.json")
    parser.add_argument("--fact_grounded", default="results/fact_grounded_final_dev.jsonl")
    parser.add_argument("--deberta_dir", default="models/deberta_fg_v2_best")
    parser.add_argument("--output", default="results/iterative_reretrieval.json")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = OpenAI()
    deberta = DebertaPredictor(args.deberta_dir)
    reformulator = QueryReformulator()
    retriever = ESRetriever()

    # Load data
    raw = {}
    with open(args.raw_data) as f:
        for line in f:
            d = json.loads(line)
            raw[d['id']] = d

    with open(args.subquestions) as f:
        subq_lookup = json.load(f)

    # Load fact-grounding labels to classify failure type
    fg_labels = {}
    with open(args.fact_grounded) as f:
        for line in f:
            e = json.loads(line)
            fg_labels.setdefault(e['qid'], {})[e['hop_number']] = {
                'label': e['label_binary'],
                'gold_present': e.get('gold_para_present', False)
            }

    qids = [q for q in raw if q in subq_lookup]
    if args.limit:
        qids = qids[:args.limit]

    # Classify each question by its dominant failure type
    def get_failure_type(qid):
        hops = fg_labels.get(qid, {})
        has_retrieval_failure = False
        has_extraction_failure = False
        for hop_num, info in hops.items():
            if info['label'] == 0:
                if info['gold_present']:
                    has_extraction_failure = True
                else:
                    has_retrieval_failure = True
        if has_retrieval_failure and not has_extraction_failure:
            return 'retrieval_only'
        elif has_extraction_failure and not has_retrieval_failure:
            return 'extraction_only'
        elif has_retrieval_failure and has_extraction_failure:
            return 'both'
        else:
            return 'no_failure'

    # Results per round
    results_by_round = {r: {'retrieval_only': {'n': 0, 'correct': 0},
                            'extraction_only': {'n': 0, 'correct': 0},
                            'both': {'n': 0, 'correct': 0},
                            'no_failure': {'n': 0, 'correct': 0}}
                        for r in range(args.max_rounds + 1)}

    print(f"Running iterative re-retrieval (max {args.max_rounds} rounds) on {len(qids)} questions")

    for qid in tqdm(qids, desc="Iterative"):
        d = raw[qid]
        question = d['question']
        gold = d['answer']
        sub_questions = subq_lookup[qid].get('natural_sub_questions', [])
        if not sub_questions:
            continue

        failure_type = get_failure_type(qid)

        # Cache per-hop baseline passages
        hop_passages = {}
        for sq in sub_questions:
            hop_passages[sq] = retriever.retrieve(sq, k=3)

        # Round 0: baseline retrieval
        all_passages = []
        for sq in sub_questions:
            all_passages.extend(hop_passages[sq])

        # Dedupe
        seen = set()
        unique = []
        for p in all_passages:
            if p['text'] not in seen:
                seen.add(p['text'])
                unique.append(p)

        # Evaluate round 0
        try:
            ans = run_qa(client, question, sub_questions, unique)
            correct = llm_judge(client, question, gold, ans)
        except:
            correct = False

        results_by_round[0][failure_type]['n'] += 1
        if correct:
            results_by_round[0][failure_type]['correct'] += 1

        # Rounds 1-N: iterative re-retrieval with replacement
        current_passages = list(unique)
        for round_num in range(1, args.max_rounds + 1):
            new_passages = []
            for sq in sub_questions:
                # Check DeBERTa
                passage_text = format_passages(current_passages, max_chars=4000)
                try:
                    _, prob = deberta.predict(sq, passage_text)
                    flagged = prob < 0.5
                except:
                    flagged = False

                if flagged:
                    # Reformulate and REPLACE
                    try:
                        new_q = reformulator.reformulate(question, sq)
                        replaced = retriever.retrieve(new_q, k=3)
                        new_passages.extend(replaced)
                    except:
                        new_passages.extend(hop_passages[sq])
                else:
                    new_passages.extend(hop_passages[sq])

            # Dedupe new passages
            seen = set()
            current_passages = []
            for p in new_passages:
                if p['text'] not in seen:
                    seen.add(p['text'])
                    current_passages.append(p)

            # Evaluate this round
            try:
                ans = run_qa(client, question, sub_questions, current_passages)
                correct = llm_judge(client, question, gold, ans)
            except:
                correct = False

            results_by_round[round_num][failure_type]['n'] += 1
            if correct:
                results_by_round[round_num][failure_type]['correct'] += 1

    # Print results
    print(f"\n{'='*70}")
    print(f"ITERATIVE RE-RETRIEVAL RESULTS (max {args.max_rounds} rounds)")
    print(f"{'='*70}")
    print(f"{'Round':<8} {'Type':<20} {'N':<8} {'Correct':<10} {'Acc%':<8}")
    print("-" * 70)
    for r in range(args.max_rounds + 1):
        for ftype in ['retrieval_only', 'extraction_only', 'both', 'no_failure']:
            s = results_by_round[r][ftype]
            if s['n'] > 0:
                acc = s['correct'] / s['n'] * 100
                print(f"{r:<8} {ftype:<20} {s['n']:<8} {s['correct']:<10} {acc:<8.1f}")
        print()

    # Save
    summary = {'max_rounds': args.max_rounds, 'results_by_round': results_by_round}
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
