#!/usr/bin/env python3
"""
Fact-Grounded Sufficiency Relabeling

For each hop in a QA trajectory, checks whether the gold intermediate
answer (from MuSiQue's reasoning_steps) appears in the retrieved passages.

Labels:
  FACT-GROUNDED: The intermediate answer IS present in the retrieved passages.
                 The needed fact was successfully retrieved.
  NOT-FACT-GROUNDED: The intermediate answer is NOT present in the retrieved
                     passages. The retriever found a relevant document but
                     missed the specific fact needed for this reasoning step.

This is an objective, locally verifiable label — no chain-level dependencies,
no counterfactual intervention needed.

Usage:
    python relabel_fact_grounded.py \
        --results results/full_run_all_4.1.json \
        --data processed_data/musique/dev.jsonl \
        --output results/fact_grounded_dev.jsonl \
        --stats

    python relabel_fact_grounded.py \
        --results results/train_run_4.1.json \
        --data processed_data/musique/train.jsonl \
        --output results/fact_grounded_train.jsonl \
        --stats
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter


def normalize(s: str) -> str:
    """Normalize text for substring matching."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def parse_reasoning_step(step_str: str) -> dict:
    """
    Parse MuSiQue reasoning step format.
    Format: "entity >> relation >>>> intermediate_answer"
    Returns: {'entity': ..., 'relation': ..., 'answer': ...}
    """
    # Split on >>>> first (separates answer)
    if '>>>>' in step_str:
        before_answer, answer = step_str.rsplit('>>>>', 1)
        answer = answer.strip()
    else:
        return {'entity': '', 'relation': '', 'answer': step_str.strip()}

    # Split the before part on >> (separates entity and relation)
    if '>>' in before_answer:
        entity, relation = before_answer.split('>>', 1)
        entity = entity.strip()
        relation = relation.strip()
    else:
        entity = before_answer.strip()
        relation = ''

    return {
        'entity': entity,
        'relation': relation,
        'answer': answer,
    }


def answer_in_passages(answer: str, passages: list) -> bool:
    """
    Check if the intermediate answer appears in any of the retrieved passages.
    Uses normalized substring matching.
    """
    if not answer.strip():
        return False

    norm_answer = normalize(answer)
    if not norm_answer:
        return False

    for p in passages:
        text = p.get('text', '') + ' ' + p.get('title', '')
        if norm_answer in normalize(text):
            return True

    return False


def get_retrieval_recall(retrieved_passages: list, gold_passages: list) -> float:
    """What fraction of gold passage titles were retrieved?"""
    if not gold_passages:
        return 0.0
    gold_titles = set(g['title'].lower().strip() for g in gold_passages)
    retrieved_titles = set(p['title'].lower().strip() for p in retrieved_passages)
    if not gold_titles:
        return 0.0
    return len(gold_titles & retrieved_titles) / len(gold_titles)


def get_gold_passages(example: dict) -> list:
    """Get all gold supporting passages from a MuSiQue example."""
    gold = []
    for ctx in example.get('contexts', []):
        if ctx.get('is_supporting', False):
            gold.append({
                'title': ctx.get('title', ''),
                'text': ctx.get('paragraph_text', ''),
            })
    return gold


def main():
    parser = argparse.ArgumentParser(description="Relabel with fact-grounded sufficiency")
    parser.add_argument("--results", required=True, help="QA results JSON from run_multihop_eval.py")
    parser.add_argument("--data", required=True, help="MuSiQue data file (jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL with relabeled examples")
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics")
    args = parser.parse_args()

    # Load QA results
    with open(args.results) as f:
        results_data = json.load(f)

    # Load MuSiQue data
    musique = {}
    with open(args.data) as f:
        for line in f:
            item = json.loads(line)
            musique[item.get('question_id', '')] = item

    print(f"Loaded {len(results_data['results'])} QA results")
    print(f"Loaded {len(musique)} MuSiQue examples")

    # Process each example
    all_examples = []
    stats = {
        'total_hops': 0,
        'matched_hops': 0,  # hops where we could find a reasoning step
        'fact_grounded': 0,
        'not_fact_grounded': 0,
        'no_reasoning_step': 0,
        'recall_when_grounded': [],
        'recall_when_not_grounded': [],
        'per_hop': {},  # hop_num -> {grounded, not_grounded}
    }

    for r in results_data['results']:
        qid = r.get('qid', '')
        question = r['question']
        trajectory = r.get('trajectory', {})
        steps = trajectory.get('steps', [])

        # Get MuSiQue data for this question
        musique_item = musique.get(qid, {})
        reasoning_steps_raw = musique_item.get('reasoning_steps', [])
        gold_passages = get_gold_passages(musique_item)

        # Parse reasoning steps to get intermediate answers
        reasoning_steps = [parse_reasoning_step(s) for s in reasoning_steps_raw]
        intermediate_answers = [rs['answer'] for rs in reasoning_steps]

        if not reasoning_steps or not steps:
            continue

        # For each hop in the trajectory, check if any intermediate answer
        # is present in the retrieved passages
        for hop_idx, step in enumerate(steps):
            retrieved = step.get('retrieved_passages', [])
            sub_question = step.get('query', '')
            hop_num = hop_idx + 1

            stats['total_hops'] += 1

            # Try to match this hop to a reasoning step
            # Strategy: check each intermediate answer against passages
            # Use the hop index if it aligns, otherwise check all unmatched
            found_answer = None
            is_grounded = False

            if hop_idx < len(intermediate_answers):
                # Direct alignment: check the answer for this hop index
                target_answer = intermediate_answers[hop_idx]
                is_grounded = answer_in_passages(target_answer, retrieved)
                found_answer = target_answer
                stats['matched_hops'] += 1
            else:
                # More hops than reasoning steps — check all answers
                for ans in intermediate_answers:
                    if answer_in_passages(ans, retrieved):
                        is_grounded = True
                        found_answer = ans
                        break
                if found_answer is None and intermediate_answers:
                    found_answer = intermediate_answers[-1]  # use last as default
                stats['matched_hops'] += 1

            # Compute retrieval recall for this hop
            recall = get_retrieval_recall(retrieved, gold_passages)

            # Label
            label = "FACT-GROUNDED" if is_grounded else "NOT-FACT-GROUNDED"
            label_binary = 1 if is_grounded else 0

            # Update stats
            if is_grounded:
                stats['fact_grounded'] += 1
                stats['recall_when_grounded'].append(recall)
            else:
                stats['not_fact_grounded'] += 1
                stats['recall_when_not_grounded'].append(recall)

            # Per-hop stats
            if hop_num not in stats['per_hop']:
                stats['per_hop'][hop_num] = {'grounded': 0, 'not_grounded': 0}
            if is_grounded:
                stats['per_hop'][hop_num]['grounded'] += 1
            else:
                stats['per_hop'][hop_num]['not_grounded'] += 1

            # Compute simple features for baseline comparison
            all_text = " ".join(p.get('text', '') for p in retrieved)
            query_words = set(normalize(sub_question).split())
            passage_words = set(normalize(all_text).split())

            example = {
                'qid': qid,
                'original_question': question,
                'sub_question': sub_question,
                'hop_number': hop_num,
                'target_intermediate_answer': found_answer,
                'passages': retrieved,
                'passage_text_combined': "\n".join(
                    f"[{i+1}] {p.get('title','')}: {p.get('text','')}"
                    for i, p in enumerate(retrieved)
                ),
                'retrieval_recall': round(recall, 4),
                'label': label,
                'label_binary': label_binary,
                'features': {
                    'avg_retrieval_score': round(
                        sum(p.get('score', 0) for p in retrieved) / max(len(retrieved), 1), 4
                    ),
                    'lexical_overlap': round(
                        len(query_words & passage_words) / max(len(query_words | passage_words), 1), 4
                    ),
                    'query_coverage': round(
                        len(query_words & passage_words) / max(len(query_words), 1), 4
                    ),
                    'passage_word_count': len(all_text.split()),
                    'num_passages': len(retrieved),
                },
            }
            all_examples.append(example)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved {len(all_examples)} examples to {args.output}")

    # Print statistics
    total = stats['fact_grounded'] + stats['not_fact_grounded']
    print(f"\n{'='*60}")
    print(f"FACT-GROUNDED SUFFICIENCY STATISTICS")
    print(f"{'='*60}")
    print(f"Total hop-level examples: {total}")
    print(f"  FACT-GROUNDED:     {stats['fact_grounded']} ({stats['fact_grounded']/total*100:.1f}%)")
    print(f"  NOT-FACT-GROUNDED: {stats['not_fact_grounded']} ({stats['not_fact_grounded']/total*100:.1f}%)")

    # The key finding: retrieval recall vs fact-groundedness
    if stats['recall_when_grounded'] and stats['recall_when_not_grounded']:
        avg_recall_grounded = sum(stats['recall_when_grounded']) / len(stats['recall_when_grounded'])
        avg_recall_not_grounded = sum(stats['recall_when_not_grounded']) / len(stats['recall_when_not_grounded'])
        gap = abs(avg_recall_grounded - avg_recall_not_grounded)

        print(f"\n  KEY FINDING: Document recall vs fact presence")
        print(f"  Avg recall when FACT-GROUNDED:     {avg_recall_grounded:.3f}")
        print(f"  Avg recall when NOT-FACT-GROUNDED: {avg_recall_not_grounded:.3f}")
        print(f"  Gap: {gap:.3f}")

        if gap > 0.05:
            print(f"  >>> Retrieval recall partially distinguishes fact presence")
        else:
            print(f"  >>> Retrieval recall does NOT distinguish fact presence")
            print(f"  >>> Right document ≠ right fact (paper's key finding)")

    # Per-hop breakdown
    print(f"\n  Per-hop breakdown:")
    for h in sorted(stats['per_hop'].keys()):
        g = stats['per_hop'][h]['grounded']
        ng = stats['per_hop'][h]['not_grounded']
        total_h = g + ng
        print(f"    Hop {h}: {g}/{total_h} fact-grounded ({g/total_h*100:.1f}%)")

    # Cross-tabulation: recall vs fact-groundedness
    if args.stats:
        print(f"\n  Cross-tabulation (recall > 0 vs fact-grounded):")
        high_recall_grounded = sum(1 for r, ex in zip(
            stats['recall_when_grounded'], [e for e in all_examples if e['label'] == 'FACT-GROUNDED']
        ) if r > 0)
        high_recall_not_grounded = sum(1 for r in stats['recall_when_not_grounded'] if r > 0)

        print(f"    Recall > 0 AND fact-grounded: {high_recall_grounded}")
        print(f"    Recall > 0 AND NOT fact-grounded: {high_recall_not_grounded}")
        print(f"    Recall = 0 AND fact-grounded: {stats['fact_grounded'] - high_recall_grounded}")
        print(f"    Recall = 0 AND NOT fact-grounded: {stats['not_fact_grounded'] - high_recall_not_grounded}")

        # Check: among examples where recall > 0 (right document retrieved),
        # how often is the fact actually present?
        all_recall = [ex['retrieval_recall'] for ex in all_examples]
        all_labels = [ex['label_binary'] for ex in all_examples]

        high_recall_examples = [(r, l) for r, l in zip(all_recall, all_labels) if r > 0]
        if high_recall_examples:
            fact_rate = sum(l for _, l in high_recall_examples) / len(high_recall_examples)
            print(f"\n  CRITICAL FINDING:")
            print(f"    Among hops with recall > 0 (right document retrieved):")
            print(f"    Only {fact_rate:.1%} actually contain the needed intermediate answer")
            print(f"    This means {1-fact_rate:.1%} of 'successful' retrievals are missing the key fact")


if __name__ == "__main__":
    main()
