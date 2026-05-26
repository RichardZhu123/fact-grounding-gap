#!/usr/bin/env python3
"""
Fact-Grounded Sufficiency via Answerability (Definition B)

For each hop in a QA trajectory, asks an LLM judge:
"Can this sub-question be answered using ONLY the retrieved passages?"

Key design choices (following professor's guidance):
- Uses raw MuSiQue data with question_decomposition for proper sub-questions
- Resolves #N references (e.g., "#1 >> place of birth" -> "Trey Parker >> place of birth")
- Does NOT show the expected intermediate answer to the judge (no answer leakage)
- Validates against paragraph_support_idx as sanity check

Labels:
  ANSWERABLE: The sub-question can be answered from the retrieved passages.
              The needed fact is present (Definition B: answerability).
  NOT-ANSWERABLE: The sub-question cannot be answered from the passages.
                  The retriever found relevant documents but missed the specific fact.

Usage:
    # Test on 10 examples first
    python fact_grounding.py \
        --results results/full_run_all_4.1.json \
        --raw_data raw_data/musique/musique_ans_v1.0_dev.jsonl \
        --output results/fact_grounded_v2_test.jsonl \
        --limit 10 --stats

    # Full dev set (~3-4 hours, ~$10)
    python fact_grounding.py \
        --results results/full_run_all_4.1.json \
        --raw_data raw_data/musique/musique_ans_v1.0_dev.jsonl \
        --output results/fact_grounded_v2_dev.jsonl \
        --stats
"""

import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter
from openai import OpenAI
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Sub-question resolution
# ──────────────────────────────────────────────────────────────────────

def resolve_sub_question(question_decomposition: list, hop_idx: int) -> str:
    """
    Resolve #N references in a sub-question using previous hop answers.

    Example:
        Hop 0: "The Hobbit >> part of the series" -> answer "South Park"
        Hop 1: "who does the voice of stan on #1" -> resolves to
                "who does the voice of stan on South Park"
        Hop 2: "#2 >> place of birth" -> resolves to
                "Trey Parker >> place of birth"
    """
    step = question_decomposition[hop_idx]
    sub_q = step['question']

    # Replace #N references with answers from previous hops
    for prev_idx in range(hop_idx):
        prev_answer = question_decomposition[prev_idx]['answer']
        ref = f"#{prev_idx + 1}"
        sub_q = sub_q.replace(ref, prev_answer)

    return sub_q


def format_sub_question_readable(sub_q: str) -> str:
    """
    Convert shorthand format to more readable form for the LLM judge.
    "Green >> performer" -> "Green: performer"
    "Trey Parker >> place of birth" -> "Trey Parker: place of birth"

    Already natural language questions pass through unchanged.
    """
    # Replace >> with : for readability
    sub_q = sub_q.replace(" >> ", ": ")
    return sub_q


# ──────────────────────────────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "sub_question": "Which company distributed the film UHF?",
        "passages": "[1] UHF (film): UHF is a 1989 American comedy film starring Weird Al Yankovic. The film was distributed by Orion Pictures and was released on July 21, 1989.",
        "reasoning": "The passage states that UHF 'was distributed by Orion Pictures', which directly answers the sub-question.",
        "verdict": "YES"
    },
    {
        "sub_question": "Which company distributed the film UHF?",
        "passages": "[1] UHF: UHF refers to ultra-high frequency radio waves with frequencies between 300 MHz and 3 GHz. UHF is used for television broadcasting, cell phones, and satellite communication.",
        "reasoning": "This passage is about UHF radio frequencies, not the film UHF. It contains no information about film distribution.",
        "verdict": "NO"
    },
    {
        "sub_question": "Trey Parker: place of birth",
        "passages": "[1] Trey Parker: Randolph Severn 'Trey' Parker III is an American animator, filmmaker, and actor. He was born in Conifer, Colorado, and later moved to Denver with his family.",
        "reasoning": "The passage states Parker 'was born in Conifer, Colorado', which answers the question about his place of birth.",
        "verdict": "YES"
    },
    {
        "sub_question": "Trey Parker: place of birth",
        "passages": "[1] South Park: South Park is an American animated sitcom created by Trey Parker and Matt Stone. The show premiered on August 13, 1997, on Comedy Central.",
        "reasoning": "The passage mentions Trey Parker as a creator of South Park, but says nothing about where he was born.",
        "verdict": "NO"
    },
    {
        "sub_question": "Who is the spouse of Steve Hillage?",
        "passages": "[1] Steve Hillage: Steve Hillage is an English musician. He has been in a long-term partnership with synthesizer player Miquette Giraudy since the 1970s, and they frequently collaborate.",
        "reasoning": "The passage mentions Miquette Giraudy as Hillage's long-term partner. While it says 'partnership' rather than 'spouse', this provides enough information to answer the question about his spouse/partner.",
        "verdict": "YES"
    },
    {
        "sub_question": "Who is the spouse of Steve Hillage?",
        "passages": "[1] Gong (band): Gong is a rock band founded by Daevid Allen in 1967. Steve Hillage joined as guitarist in 1973 and played on several albums before leaving for a solo career.",
        "reasoning": "The passage discusses Steve Hillage's career with the band Gong but contains no information about his spouse or personal relationships.",
        "verdict": "NO"
    },
]


def build_judge_prompt(sub_question: str, passages: str) -> str:
    """
    Build the judge prompt. Note: NO expected answer is shown.
    The judge evaluates answerability purely from the sub-question and passages.
    """
    prompt = """You are evaluating whether retrieved passages contain enough information to answer a specific sub-question in a multi-hop reasoning chain.

Given a sub-question and retrieved passages, determine:
- YES: The passages contain enough information to answer the sub-question (even if the answer is implied, paraphrased, or requires minor inference).
- NO: The passages do NOT contain enough information to answer the sub-question. The needed fact is missing.

IMPORTANT: Judge whether the SPECIFIC FACT needed to answer this sub-question is present. A passage can be topically related but still lack the specific fact needed.

Examples:

"""
    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"""Sub-question: {ex['sub_question']}
Passages: {ex['passages']}
Reasoning: {ex['reasoning']}
Verdict: {ex['verdict']}

"""

    prompt += f"""Now evaluate:

Sub-question: {sub_question}
Passages: {passages}
Reasoning:"""

    return prompt


def parse_verdict(response_text: str) -> str:
    """Extract YES or NO from the judge response."""
    text = response_text.strip().upper()

    # Look for verdict line
    if "VERDICT:" in text:
        after = text.split("VERDICT:")[-1].strip()
        if after.startswith("NO"):
            return "NO"
        if after.startswith("YES"):
            return "YES"

    # Check last line
    lines = text.strip().split("\n")
    last_line = lines[-1].strip()
    if last_line.startswith("NO") or last_line == "NO":
        return "NO"
    if last_line.startswith("YES") or last_line == "YES":
        return "YES"

    # Search whole text (last occurrence)
    # Find the last YES or NO
    last_yes = text.rfind("YES")
    last_no = text.rfind("NO")

    if last_yes == -1 and last_no == -1:
        return "UNKNOWN"
    if last_yes > last_no:
        return "YES"
    if last_no > last_yes:
        # Make sure it's not part of another word like "KNOW", "NOTION"
        # Check if NO is at word boundary
        if last_no > 0 and text[last_no - 1].isalpha():
            # Part of another word, check for standalone YES
            if last_yes > -1:
                return "YES"
            return "UNKNOWN"
        return "NO"

    return "UNKNOWN"


def judge_answerability(
    client: OpenAI,
    sub_question: str,
    passages_text: str,
    model: str = "gpt-4.1-mini",
    max_retries: int = 5,
) -> tuple:
    """Returns (verdict, reasoning). Verdict is YES, NO, or UNKNOWN."""
    prompt = build_judge_prompt(sub_question, passages_text)

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


# ──────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────────────

def get_retrieval_recall(retrieved_passages: list, gold_passages: list) -> float:
    if not gold_passages:
        return 0.0
    gold_titles = set(g['title'].lower().strip() for g in gold_passages)
    retrieved_titles = set(p['title'].lower().strip() for p in retrieved_passages)
    if not gold_titles:
        return 0.0
    return len(gold_titles & retrieved_titles) / len(gold_titles)


def check_gold_paragraph_retrieved(retrieved_passages: list, gold_paragraph: dict) -> bool:
    """Check if the specific gold supporting paragraph was retrieved."""
    gold_title = gold_paragraph.get('title', '').lower().strip()
    for p in retrieved_passages:
        if p.get('title', '').lower().strip() == gold_title:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fact-grounding via answerability (Definition B)")
    parser.add_argument("--results", required=True, help="QA results JSON")
    parser.add_argument("--raw_data", required=True, help="Raw MuSiQue data (musique_ans_v1.0_dev.jsonl)")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Judge model")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    args = parser.parse_args()

    client = OpenAI()

    # Load QA results
    with open(args.results) as f:
        results_data = json.load(f)

    # Load raw MuSiQue data
    musique_raw = {}
    with open(args.raw_data) as f:
        for line in f:
            item = json.loads(line)
            musique_raw[item['id']] = item

    results_list = results_data['results']
    if args.limit:
        results_list = results_list[:args.limit]

    print(f"Loaded {len(results_list)} QA results")
    print(f"Loaded {len(musique_raw)} raw MuSiQue examples")
    print(f"Judge model: {args.model}")

    # Process
    all_examples = []
    unknown_count = 0

    # Validation counters
    gold_para_retrieved_and_answerable = 0
    gold_para_retrieved_and_not_answerable = 0
    gold_para_not_retrieved_and_answerable = 0
    gold_para_not_retrieved_and_not_answerable = 0

    for r in tqdm(results_list, desc="Fact-grounding"):
        qid = r.get('qid', '')
        question = r['question']
        trajectory = r.get('trajectory', {})
        steps = trajectory.get('steps', [])

        raw_item = musique_raw.get(qid)
        if not raw_item:
            continue

        decomposition = raw_item['question_decomposition']
        paragraphs = raw_item['paragraphs']

        # Get all gold supporting passages
        gold_passages = [
            {'title': p['title'], 'text': p['paragraph_text']}
            for p in paragraphs if p.get('is_supporting', False)
        ]

        # Process each hop
        num_hops_to_process = min(len(steps), len(decomposition))

        for hop_idx in range(num_hops_to_process):
            step = steps[hop_idx]
            decomp_step = decomposition[hop_idx]

            # Resolve sub-question with #N references
            resolved_sub_q = resolve_sub_question(decomposition, hop_idx)
            readable_sub_q = format_sub_question_readable(resolved_sub_q)

            # Get intermediate answer (for validation only, NOT shown to judge)
            intermediate_answer = decomp_step['answer']

            # Get gold supporting paragraph for this hop
            gold_para_idx = decomp_step['paragraph_support_idx']
            gold_paragraph = paragraphs[gold_para_idx] if gold_para_idx < len(paragraphs) else None

            # Get retrieved passages
            retrieved = step.get('retrieved_passages', [])
            hop_num = hop_idx + 1

            # Format passages for judge
            passage_parts = []
            for i, p in enumerate(retrieved, 1):
                title = p.get('title', '')
                text = p.get('text', '')[:500]
                passage_parts.append(f"[{i}] {title}: {text}")
            passages_text = "\n".join(passage_parts) if passage_parts else "No passages retrieved."

            # LLM judge — note: NO intermediate answer shown
            verdict, reasoning = judge_answerability(
                client, readable_sub_q, passages_text, args.model
            )

            if verdict == "UNKNOWN":
                unknown_count += 1

            # Map to labels
            label = "ANSWERABLE" if verdict == "YES" else "NOT-ANSWERABLE" if verdict == "NO" else "UNKNOWN"
            label_binary = 1 if verdict == "YES" else 0

            # Retrieval metrics
            recall = get_retrieval_recall(retrieved, gold_passages)
            gold_para_retrieved = check_gold_paragraph_retrieved(
                retrieved, gold_paragraph
            ) if gold_paragraph else False

            # Validation: gold paragraph retrieved vs answerability
            if gold_para_retrieved and verdict == "YES":
                gold_para_retrieved_and_answerable += 1
            elif gold_para_retrieved and verdict == "NO":
                gold_para_retrieved_and_not_answerable += 1
            elif not gold_para_retrieved and verdict == "YES":
                gold_para_not_retrieved_and_answerable += 1
            elif not gold_para_retrieved and verdict == "NO":
                gold_para_not_retrieved_and_not_answerable += 1

            # String match for comparison
            norm_answer = re.sub(r'[^\w\s]', '', intermediate_answer.lower().strip())
            all_text = " ".join(p.get('text', '') + ' ' + p.get('title', '') for p in retrieved)
            norm_text = re.sub(r'[^\w\s]', '', all_text.lower().strip())
            string_match = norm_answer in norm_text if norm_answer else False

            # Simple features
            query_words = set(re.sub(r'[^\w\s]', '', readable_sub_q.lower()).split())
            passage_words = set(re.sub(r'[^\w\s]', '', all_text.lower()).split())

            example = {
                'qid': qid,
                'original_question': question,
                'sub_question_raw': decomp_step['question'],
                'sub_question_resolved': readable_sub_q,
                'hop_number': hop_num,
                'intermediate_answer': intermediate_answer,
                'gold_para_title': gold_paragraph['title'] if gold_paragraph else '',
                'gold_para_retrieved': gold_para_retrieved,
                'passages': retrieved,
                'passage_text_combined': passages_text,
                'retrieval_recall': round(recall, 4),
                'label': label,
                'label_binary': label_binary,
                'string_match_label': 1 if string_match else 0,
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

    # ── Statistics ──
    labeled = [e for e in all_examples if e['label'] != 'UNKNOWN']
    total = len(labeled)
    answerable = sum(1 for e in labeled if e['label'] == 'ANSWERABLE')
    not_answerable = sum(1 for e in labeled if e['label'] == 'NOT-ANSWERABLE')

    print(f"\n{'='*60}")
    print(f"FACT-GROUNDED SUFFICIENCY (ANSWERABILITY, DEFINITION B)")
    print(f"{'='*60}")
    print(f"Total labeled: {total} (+ {unknown_count} unknown)")
    print(f"  ANSWERABLE:     {answerable} ({answerable/total*100:.1f}%)")
    print(f"  NOT-ANSWERABLE: {not_answerable} ({not_answerable/total*100:.1f}%)")

    # Validation: gold paragraph alignment
    total_validation = (gold_para_retrieved_and_answerable + gold_para_retrieved_and_not_answerable +
                        gold_para_not_retrieved_and_answerable + gold_para_not_retrieved_and_not_answerable)
    if total_validation > 0:
        print(f"\n  VALIDATION (gold paragraph vs answerability):")
        print(f"  {'':30s} {'Answerable':>12s} {'Not-Answerable':>15s}")
        print(f"  {'Gold para retrieved':30s} {gold_para_retrieved_and_answerable:>12d} {gold_para_retrieved_and_not_answerable:>15d}")
        print(f"  {'Gold para NOT retrieved':30s} {gold_para_not_retrieved_and_answerable:>12d} {gold_para_not_retrieved_and_not_answerable:>15d}")

        if gold_para_retrieved_and_answerable + gold_para_retrieved_and_not_answerable > 0:
            precision = gold_para_retrieved_and_answerable / (gold_para_retrieved_and_answerable + gold_para_retrieved_and_not_answerable)
            print(f"\n  When gold paragraph IS retrieved: {precision:.1%} judged answerable")
            print(f"  (This should be high — validates the judge)")

    # Key finding
    high_recall = [e for e in labeled if e['retrieval_recall'] > 0]
    if high_recall:
        fact_rate = sum(1 for e in high_recall if e['label'] == 'ANSWERABLE') / len(high_recall)
        print(f"\n  KEY FINDING:")
        print(f"    Among hops with recall > 0 (right document retrieved):")
        print(f"    {fact_rate:.1%} are answerable from the retrieved passages")
        print(f"    {1-fact_rate:.1%} are NOT answerable despite 'successful' retrieval")

    # Recall gap
    recall_answerable = [e['retrieval_recall'] for e in labeled if e['label'] == 'ANSWERABLE']
    recall_not = [e['retrieval_recall'] for e in labeled if e['label'] == 'NOT-ANSWERABLE']
    if recall_answerable and recall_not:
        gap = abs(sum(recall_answerable)/len(recall_answerable) - sum(recall_not)/len(recall_not))
        print(f"\n  Recall gap: {gap:.3f}")
        print(f"    Avg recall when ANSWERABLE:     {sum(recall_answerable)/len(recall_answerable):.3f}")
        print(f"    Avg recall when NOT-ANSWERABLE: {sum(recall_not)/len(recall_not):.3f}")

    # Per-hop
    print(f"\n  Per-hop breakdown:")
    for h in sorted(set(e['hop_number'] for e in labeled)):
        hop_ex = [e for e in labeled if e['hop_number'] == h]
        a = sum(1 for e in hop_ex if e['label'] == 'ANSWERABLE')
        print(f"    Hop {h}: {a}/{len(hop_ex)} answerable ({a/len(hop_ex)*100:.1f}%)")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
