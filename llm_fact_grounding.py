#!/usr/bin/env python3
"""
LLM-Based Fact-Grounding Evaluation

For each hop in a QA trajectory, uses GPT-4.1-mini to judge whether
the retrieved passages contain sufficient information to answer the
sub-question, given the expected intermediate answer.

This is more robust than string matching because it handles:
- Paraphrased answers ("the guitarist from Gong" = "Steve Hillage")
- Implied facts (passage implies the answer without stating it directly)
- Indirect references ("he" referring to the correct entity)

Labels:
  FACT-GROUNDED: The passages contain enough information to determine
                 the intermediate answer (even if not stated verbatim).
  NOT-FACT-GROUNDED: The passages do NOT contain enough information.
                     The needed fact is genuinely missing.

Usage:
    python llm_fact_grounding.py \
        --results results/full_run_all_4.1.json \
        --data processed_data/musique/dev.jsonl \
        --output results/fact_grounded_llm_dev.jsonl \
        --stats

    python llm_fact_grounding.py \
        --results results/train_run_4.1.json \
        --data processed_data/musique/train.jsonl \
        --output results/fact_grounded_llm_train.jsonl
"""

import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter
from openai import OpenAI


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def parse_reasoning_step(step_str: str) -> dict:
    if '>>>>' in step_str:
        before_answer, answer = step_str.rsplit('>>>>', 1)
        answer = answer.strip()
    else:
        return {'entity': '', 'relation': '', 'answer': step_str.strip()}

    if '>>' in before_answer:
        entity, relation = before_answer.split('>>', 1)
        entity = entity.strip()
        relation = relation.strip()
    else:
        entity = before_answer.strip()
        relation = ''

    return {'entity': entity, 'relation': relation, 'answer': answer}


def get_gold_passages(example: dict) -> list:
    gold = []
    for ctx in example.get('contexts', []):
        if ctx.get('is_supporting', False):
            gold.append({
                'title': ctx.get('title', ''),
                'text': ctx.get('paragraph_text', ''),
            })
    return gold


def get_retrieval_recall(retrieved_passages: list, gold_passages: list) -> float:
    if not gold_passages:
        return 0.0
    gold_titles = set(g['title'].lower().strip() for g in gold_passages)
    retrieved_titles = set(p['title'].lower().strip() for p in retrieved_passages)
    if not gold_titles:
        return 0.0
    return len(gold_titles & retrieved_titles) / len(gold_titles)


# ──────────────────────────────────────────────────────────────────────
# LLM Judge for fact-grounding
# ──────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "sub_question": "Who performed the album Green?",
        "expected_answer": "Steve Hillage",
        "passages": "[1] Green (album): Green is a 1978 album by Steve Hillage, recorded at Rockfield Studios.",
        "reasoning": "The passage directly states that Green is 'by Steve Hillage', which matches the expected answer.",
        "verdict": "FACT-GROUNDED"
    },
    {
        "sub_question": "Who performed the album Green?",
        "expected_answer": "Steve Hillage",
        "passages": "[1] Green (R.E.M. album): Green is the sixth studio album by R.E.M., released in 1988 on Warner Bros.",
        "reasoning": "This passage is about a different album called Green by R.E.M., not by Steve Hillage. The expected answer is not present or inferrable.",
        "verdict": "NOT-FACT-GROUNDED"
    },
    {
        "sub_question": "When was the employer of John Smith founded?",
        "expected_answer": "1842",
        "passages": "[1] John Smith (professor): John Smith is a professor at the University of Edinburgh, specializing in computational linguistics.",
        "reasoning": "The passage identifies John Smith's employer as the University of Edinburgh, but does not mention when it was founded. The expected answer '1842' cannot be determined from this passage.",
        "verdict": "NOT-FACT-GROUNDED"
    },
    {
        "sub_question": "Who is the director of Inception?",
        "expected_answer": "Christopher Nolan",
        "passages": "[1] Inception: Inception is a 2010 science fiction film written, produced, and directed by the British-American filmmaker who also made The Dark Knight trilogy.",
        "reasoning": "The passage describes the director as 'the British-American filmmaker who also made The Dark Knight trilogy'. While 'Christopher Nolan' is not stated verbatim, this description uniquely identifies him. The fact is inferrable from the passage.",
        "verdict": "FACT-GROUNDED"
    },
    {
        "sub_question": "What country is the University of Southampton in?",
        "expected_answer": "United Kingdom",
        "passages": "[1] University of Southampton: The University of Southampton is a public research university in Southampton, England.",
        "reasoning": "The passage says the university is in 'Southampton, England'. England is part of the United Kingdom, so the expected answer is inferrable from the passage.",
        "verdict": "FACT-GROUNDED"
    },
    {
        "sub_question": "Who is the spouse of Steve Hillage?",
        "expected_answer": "Miquette Giraudy",
        "passages": "[1] Steve Hillage: Steve Hillage is an English musician best known as a guitarist with the band Gong. He has released several solo albums and is known for his psychedelic rock style.",
        "reasoning": "The passage describes Steve Hillage's career but does not mention his spouse or Miquette Giraudy. The needed fact is missing.",
        "verdict": "NOT-FACT-GROUNDED"
    },
]


def build_judge_prompt(sub_question: str, expected_answer: str, passages: str) -> str:
    prompt = """You are an expert evaluator checking whether retrieved passages contain enough information to answer a specific sub-question.

Given a sub-question, the expected answer, and retrieved passages, determine:
- FACT-GROUNDED: The passages contain the expected answer OR enough information to determine/infer it (even if not stated verbatim — paraphrases, descriptions, and implications count).
- NOT-FACT-GROUNDED: The passages do NOT contain the expected answer and there is no way to determine it from the given text.

IMPORTANT: The answer does NOT need to appear as an exact string. If the passage clearly implies or describes the answer, that counts as FACT-GROUNDED.

Examples:

"""
    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"""Sub-question: {ex['sub_question']}
Expected answer: {ex['expected_answer']}
Passages: {ex['passages']}
Reasoning: {ex['reasoning']}
Verdict: {ex['verdict']}

"""

    prompt += f"""Now evaluate:

Sub-question: {sub_question}
Expected answer: {expected_answer}
Passages: {passages}
Reasoning:"""

    return prompt


def parse_verdict(response_text: str) -> str:
    text = response_text.strip().upper()
    if "VERDICT:" in text:
        after = text.split("VERDICT:")[-1].strip()
        if after.startswith("NOT-FACT-GROUNDED") or after.startswith("NOT FACT"):
            return "NOT-FACT-GROUNDED"
        if after.startswith("FACT-GROUNDED") or after.startswith("FACT GROUNDED"):
            return "FACT-GROUNDED"
    lines = text.strip().split("\n")
    last_line = lines[-1].strip()
    if "NOT-FACT-GROUNDED" in last_line or "NOT FACT" in last_line:
        return "NOT-FACT-GROUNDED"
    if "FACT-GROUNDED" in last_line or "FACT GROUNDED" in last_line:
        return "FACT-GROUNDED"
    if "NOT-FACT-GROUNDED" in text or "NOT FACT" in text:
        return "NOT-FACT-GROUNDED"
    if "FACT-GROUNDED" in text or "FACT GROUNDED" in text:
        return "FACT-GROUNDED"
    return "UNKNOWN"


def judge_fact_grounding(
    client: OpenAI,
    sub_question: str,
    expected_answer: str,
    passages_text: str,
    model: str = "gpt-4.1-mini",
    max_retries: int = 5,
) -> tuple:
    """Returns (verdict, reasoning)."""
    prompt = build_judge_prompt(sub_question, expected_answer, passages_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            response_text = response.choices[0].message.content.strip()
            verdict = parse_verdict(response_text)

            if verdict != "UNKNOWN":
                return verdict, response_text
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return "UNKNOWN", f"ERROR: {e}"

    return "UNKNOWN", "Could not parse verdict"


def main():
    parser = argparse.ArgumentParser(description="LLM-based fact-grounding evaluation")
    parser.add_argument("--results", required=True, help="QA results JSON")
    parser.add_argument("--data", required=True, help="MuSiQue data file (jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Judge model")
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA results to process")
    args = parser.parse_args()

    client = OpenAI()

    # Load data
    with open(args.results) as f:
        results_data = json.load(f)

    musique = {}
    with open(args.data) as f:
        for line in f:
            item = json.loads(line)
            musique[item.get('question_id', '')] = item

    results_list = results_data['results']
    if args.limit:
        results_list = results_list[:args.limit]

    print(f"Loaded {len(results_list)} QA results")
    print(f"Loaded {len(musique)} MuSiQue examples")
    print(f"Judge model: {args.model}")

    # Process
    all_examples = []
    total_judged = 0
    unknown_count = 0

    # Also track string match for comparison
    string_match_agree = 0
    string_match_disagree = 0

    from tqdm import tqdm

    for r in tqdm(results_list, desc="Fact-grounding"):
        qid = r.get('qid', '')
        question = r['question']
        trajectory = r.get('trajectory', {})
        steps = trajectory.get('steps', [])

        musique_item = musique.get(qid, {})
        reasoning_steps_raw = musique_item.get('reasoning_steps', [])
        gold_passages = get_gold_passages(musique_item)

        reasoning_steps = [parse_reasoning_step(s) for s in reasoning_steps_raw]
        intermediate_answers = [rs['answer'] for rs in reasoning_steps]

        if not reasoning_steps or not steps:
            continue

        for hop_idx, step in enumerate(steps):
            retrieved = step.get('retrieved_passages', [])
            sub_question = step.get('query', '')
            hop_num = hop_idx + 1

            # Get target intermediate answer
            if hop_idx < len(intermediate_answers):
                target_answer = intermediate_answers[hop_idx]
            elif intermediate_answers:
                target_answer = intermediate_answers[-1]
            else:
                continue

            # Format passages for judge
            passage_parts = []
            for i, p in enumerate(retrieved, 1):
                title = p.get('title', '')
                text = p.get('text', '')
                passage_parts.append(f"[{i}] {title}: {text[:500]}")
            passages_text = "\n".join(passage_parts) if passage_parts else "No passages retrieved."

            # LLM judge
            verdict, reasoning = judge_fact_grounding(
                client, sub_question, target_answer, passages_text, args.model
            )

            total_judged += 1
            if verdict == "UNKNOWN":
                unknown_count += 1

            label_binary = 1 if verdict == "FACT-GROUNDED" else 0

            # String match comparison
            norm_answer = normalize(target_answer)
            all_text = " ".join(p.get('text', '') + ' ' + p.get('title', '') for p in retrieved)
            string_match = norm_answer in normalize(all_text) if norm_answer else False
            string_label = 1 if string_match else 0

            if label_binary == string_label:
                string_match_agree += 1
            else:
                string_match_disagree += 1

            # Retrieval recall
            recall = get_retrieval_recall(retrieved, gold_passages)

            # Features
            query_words = set(normalize(sub_question).split())
            passage_words = set(normalize(all_text).split())

            example = {
                'qid': qid,
                'original_question': question,
                'sub_question': sub_question,
                'hop_number': hop_num,
                'target_intermediate_answer': target_answer,
                'passages': retrieved,
                'passage_text_combined': "\n".join(
                    f"[{i+1}] {p.get('title','')}: {p.get('text','')}"
                    for i, p in enumerate(retrieved)
                ),
                'retrieval_recall': round(recall, 4),
                'label': verdict,
                'label_binary': label_binary,
                'string_match_label': string_label,
                'llm_reasoning': reasoning[:300],
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

    # Statistics
    total = len(all_examples)
    grounded = sum(1 for e in all_examples if e['label'] == 'FACT-GROUNDED')
    not_grounded = sum(1 for e in all_examples if e['label'] == 'NOT-FACT-GROUNDED')

    print(f"\n{'='*60}")
    print(f"FACT-GROUNDED SUFFICIENCY (LLM-JUDGED)")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"  FACT-GROUNDED:     {grounded} ({grounded/total*100:.1f}%)")
    print(f"  NOT-FACT-GROUNDED: {not_grounded} ({not_grounded/total*100:.1f}%)")
    if unknown_count > 0:
        print(f"  UNKNOWN:           {unknown_count}")

    # String match vs LLM comparison
    print(f"\n  String match vs LLM judge agreement:")
    print(f"    Agree: {string_match_agree} ({string_match_agree/total*100:.1f}%)")
    print(f"    Disagree: {string_match_disagree} ({string_match_disagree/total*100:.1f}%)")

    # Key finding: recall vs fact-groundedness
    recall_grounded = [e['retrieval_recall'] for e in all_examples if e['label'] == 'FACT-GROUNDED']
    recall_not_grounded = [e['retrieval_recall'] for e in all_examples if e['label'] == 'NOT-FACT-GROUNDED']

    if recall_grounded and recall_not_grounded:
        avg_g = sum(recall_grounded) / len(recall_grounded)
        avg_ng = sum(recall_not_grounded) / len(recall_not_grounded)
        gap = abs(avg_g - avg_ng)
        print(f"\n  Document recall vs fact presence:")
        print(f"    Avg recall when FACT-GROUNDED:     {avg_g:.3f}")
        print(f"    Avg recall when NOT-FACT-GROUNDED: {avg_ng:.3f}")
        print(f"    Gap: {gap:.3f}")

    # Critical finding
    high_recall = [e for e in all_examples if e['retrieval_recall'] > 0]
    if high_recall:
        fact_rate = sum(1 for e in high_recall if e['label'] == 'FACT-GROUNDED') / len(high_recall)
        print(f"\n  CRITICAL FINDING:")
        print(f"    Among hops with recall > 0 (right document retrieved):")
        print(f"    {fact_rate:.1%} contain the needed fact (LLM-judged)")
        print(f"    {1-fact_rate:.1%} are missing the key fact despite 'successful' retrieval")

    # Per-hop
    print(f"\n  Per-hop breakdown:")
    hops = sorted(set(e['hop_number'] for e in all_examples))
    for h in hops:
        hop_ex = [e for e in all_examples if e['hop_number'] == h]
        g = sum(1 for e in hop_ex if e['label'] == 'FACT-GROUNDED')
        print(f"    Hop {h}: {g}/{len(hop_ex)} fact-grounded ({g/len(hop_ex)*100:.1f}%)")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
