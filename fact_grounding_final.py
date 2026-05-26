#!/usr/bin/env python3
"""
Fact-Grounded Sufficiency via Answerability (Final Version)

For each MuSiQue decomposition step, checks whether the system's
ACCUMULATED retrieved passages (across all hops) contain enough
information to answer the sub-question.

Key design choices:
- Uses natural language sub-questions (from convert_subquestions.py)
- Checks ALL accumulated passages, not per-hop (decouples from system decomposition)
- No answer leakage — judge sees only sub-question + passages
- Validates against paragraph_support_idx
- Deduplicates passages by text content, not title

Usage:
    # Test on 10 examples
    python fact_grounding_final.py \
        --results results/full_run_all_4.1.json \
        --raw_data raw_data/musique/musique_ans_v1.0_dev.jsonl \
        --subquestions results/musique_dev_subquestions.json \
        --output results/fact_grounded_final_test.jsonl \
        --limit 10 --stats

    # Full dev set (~3-4 hours, ~$10)
    python fact_grounding_final.py \
        --results results/full_run_all_4.1.json \
        --raw_data raw_data/musique/musique_ans_v1.0_dev.jsonl \
        --subquestions results/musique_dev_subquestions.json \
        --output results/fact_grounded_final_dev.jsonl \
        --stats
"""

import json
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


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
        "sub_question": "Where was Trey Parker born?",
        "passages": "[1] Trey Parker: Randolph Severn 'Trey' Parker III is an American animator, filmmaker, and actor. He was born in Conifer, Colorado, and later moved to Denver with his family.",
        "reasoning": "The passage states Parker 'was born in Conifer, Colorado', which directly answers the question about his birthplace.",
        "verdict": "YES"
    },
    {
        "sub_question": "Where was Trey Parker born?",
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
    text = response_text.strip().upper()

    # Look for explicit verdict line
    if "VERDICT:" in text:
        after = text.split("VERDICT:")[-1].strip()
        if after.startswith("NO"):
            return "NO"
        if after.startswith("YES"):
            return "YES"

    # Check last line
    lines = text.strip().split("\n")
    last_line = lines[-1].strip()
    if last_line == "NO" or last_line.startswith("NO.") or last_line.startswith("NO "):
        return "NO"
    if last_line == "YES" or last_line.startswith("YES.") or last_line.startswith("YES "):
        return "YES"

    # Search for standalone YES/NO (avoid matching inside words)
    import re
    yes_matches = list(re.finditer(r'\bYES\b', text))
    no_matches = list(re.finditer(r'\bNO\b', text))

    if not yes_matches and not no_matches:
        return "UNKNOWN"

    # Use the last occurrence
    last_yes = yes_matches[-1].start() if yes_matches else -1
    last_no = no_matches[-1].start() if no_matches else -1

    if last_yes > last_no:
        return "YES"
    elif last_no > last_yes:
        return "NO"

    return "UNKNOWN"


def judge_answerability(
    client: OpenAI,
    sub_question: str,
    passages_text: str,
    model: str = "gpt-4.1-mini",
    max_retries: int = 5,
) -> tuple:
    prompt = build_judge_prompt(sub_question, passages_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
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
# Passage accumulation and deduplication
# ──────────────────────────────────────────────────────────────────────

def get_accumulated_passages(trajectory: dict) -> list:
    """
    Get ALL unique passages retrieved across all hops.
    Deduplicates by passage text content (not title).
    """
    seen_texts = set()
    accumulated = []

    for step in trajectory.get('steps', []):
        for p in step.get('retrieved_passages', []):
            text = p.get('text', '')
            if text not in seen_texts:
                seen_texts.add(text)
                accumulated.append(p)

    return accumulated


def format_passages_for_judge(passages: list, max_chars: int = 8000) -> str:
    """Format passages for the judge prompt, with truncation."""
    if not passages:
        return "No passages retrieved."

    parts = []
    total_chars = 0
    for i, p in enumerate(passages, 1):
        title = p.get('title', '')
        text = p.get('text', '')
        entry = f"[{i}] {title}: {text}"

        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(entry) > remaining:
            entry = entry[:remaining] + "..."
        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────────────

def check_gold_paragraph_in_passages(passages: list, gold_paragraph: dict) -> bool:
    """Check if the gold supporting paragraph is among the passages (by title match)."""
    if not gold_paragraph:
        return False
    gold_title = gold_paragraph.get('title', '').lower().strip()
    for p in passages:
        if p.get('title', '').lower().strip() == gold_title:
            return True
    return False


def compute_retrieval_recall(passages: list, all_gold_paragraphs: list) -> float:
    """Fraction of gold supporting paragraph titles found in passages."""
    if not all_gold_paragraphs:
        return 0.0
    gold_titles = set(g['title'].lower().strip() for g in all_gold_paragraphs)
    retrieved_titles = set(p.get('title', '').lower().strip() for p in passages)
    if not gold_titles:
        return 0.0
    return len(gold_titles & retrieved_titles) / len(gold_titles)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fact-grounding via answerability (final version)")
    parser.add_argument("--results", required=True, help="QA results JSON")
    parser.add_argument("--raw_data", required=True, help="Raw MuSiQue data")
    parser.add_argument("--subquestions", required=True, help="Converted sub-questions JSON")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Judge model")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    args = parser.parse_args()

    client = OpenAI()

    # Load QA results
    with open(args.results) as f:
        results_data = json.load(f)

    # Load raw MuSiQue data (for gold paragraphs)
    musique_raw = {}
    with open(args.raw_data) as f:
        for line in f:
            item = json.loads(line)
            musique_raw[item['id']] = item

    # Load converted sub-questions
    with open(args.subquestions) as f:
        subq_lookup = json.load(f)

    results_list = results_data['results']
    if args.limit:
        results_list = results_list[:args.limit]

    print(f"Loaded {len(results_list)} QA results")
    print(f"Loaded {len(musique_raw)} raw MuSiQue examples")
    print(f"Loaded {len(subq_lookup)} sub-question lookups")
    print(f"Judge model: {args.model}")

    # Process each example
    all_examples = []
    unknown_count = 0

    # Validation counters
    val_gold_yes = 0  # gold para present, judge says YES
    val_gold_no = 0   # gold para present, judge says NO
    val_nogold_yes = 0  # gold para absent, judge says YES
    val_nogold_no = 0   # gold para absent, judge says NO

    for r in tqdm(results_list, desc="Fact-grounding"):
        qid = r.get('qid', '')
        question = r['question']
        trajectory = r.get('trajectory', {})

        raw_item = musique_raw.get(qid)
        subq_data = subq_lookup.get(qid)
        if not raw_item or not subq_data:
            continue

        paragraphs = raw_item['paragraphs']
        decomposition = raw_item['question_decomposition']
        natural_sub_questions = subq_data['natural_sub_questions']
        intermediate_answers = subq_data['intermediate_answers']
        para_support_idxs = subq_data['paragraph_support_idxs']

        # Get ALL accumulated passages (deduplicated by text)
        accumulated_passages = get_accumulated_passages(trajectory)

        # Get all gold supporting passages
        all_gold_paragraphs = [
            {'title': p['title'], 'text': p['paragraph_text']}
            for p in paragraphs if p.get('is_supporting', False)
        ]

        # Overall retrieval recall for this question
        overall_recall = compute_retrieval_recall(accumulated_passages, all_gold_paragraphs)

        # Format accumulated passages for judge
        passages_text = format_passages_for_judge(accumulated_passages)

        # Evaluate each MuSiQue decomposition step
        for step_idx in range(len(decomposition)):
            sub_q = natural_sub_questions[step_idx]
            intermediate_answer = intermediate_answers[step_idx]
            gold_para_idx = para_support_idxs[step_idx]
            hop_num = step_idx + 1

            # Get gold paragraph for this step
            gold_paragraph = None
            if gold_para_idx is not None and gold_para_idx < len(paragraphs):
                gold_paragraph = {
                    'title': paragraphs[gold_para_idx]['title'],
                    'text': paragraphs[gold_para_idx]['paragraph_text'],
                }

            # Check if gold paragraph is in accumulated passages
            gold_para_present = check_gold_paragraph_in_passages(
                accumulated_passages, gold_paragraph
            )

            # LLM judge — NO intermediate answer shown
            verdict, reasoning = judge_answerability(
                client, sub_q, passages_text, args.model
            )

            if verdict == "UNKNOWN":
                unknown_count += 1

            label = "ANSWERABLE" if verdict == "YES" else "NOT-ANSWERABLE" if verdict == "NO" else "UNKNOWN"
            label_binary = 1 if verdict == "YES" else 0

            # Validation tracking
            if verdict != "UNKNOWN":
                if gold_para_present and verdict == "YES":
                    val_gold_yes += 1
                elif gold_para_present and verdict == "NO":
                    val_gold_no += 1
                elif not gold_para_present and verdict == "YES":
                    val_nogold_yes += 1
                elif not gold_para_present and verdict == "NO":
                    val_nogold_no += 1

            # String match for comparison (check if intermediate answer in accumulated passages)
            norm_answer = re.sub(r'[^\w\s]', '', intermediate_answer.lower().strip())
            all_text = " ".join(p.get('text', '') + ' ' + p.get('title', '') for p in accumulated_passages)
            norm_text = re.sub(r'[^\w\s]', '', all_text.lower().strip())
            string_match = norm_answer in norm_text if norm_answer else False

            example = {
                'qid': qid,
                'original_question': question,
                'sub_question': sub_q,
                'hop_number': hop_num,
                'intermediate_answer': intermediate_answer,
                'gold_para_title': gold_paragraph['title'] if gold_paragraph else '',
                'gold_para_present': gold_para_present,
                'num_accumulated_passages': len(accumulated_passages),
                'overall_recall': round(overall_recall, 4),
                'label': label,
                'label_binary': label_binary,
                'string_match_label': 1 if string_match else 0,
                'llm_reasoning': reasoning[:500],
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
    print(f"FACT-GROUNDED SUFFICIENCY (FINAL)")
    print(f"{'='*60}")
    print(f"Total labeled: {total} (+ {unknown_count} unknown)")
    print(f"  ANSWERABLE:     {answerable} ({answerable/total*100:.1f}%)")
    print(f"  NOT-ANSWERABLE: {not_answerable} ({not_answerable/total*100:.1f}%)")

    # Validation
    val_total_gold = val_gold_yes + val_gold_no
    val_total_nogold = val_nogold_yes + val_nogold_no

    print(f"\n  VALIDATION (gold paragraph vs judge verdict):")
    print(f"  {'':35s} {'YES (answerable)':>18s} {'NO (not answerable)':>20s}")
    print(f"  {'Gold para IN accumulated passages':35s} {val_gold_yes:>18d} {val_gold_no:>20d}")
    print(f"  {'Gold para NOT in passages':35s} {val_nogold_yes:>18d} {val_nogold_no:>20d}")

    if val_total_gold > 0:
        precision = val_gold_yes / val_total_gold
        print(f"\n  When gold paragraph IS present: {precision:.1%} judged answerable")
        print(f"  (Target: ≥85% — validates judge accuracy)")

    if val_total_nogold > 0:
        false_pos = val_nogold_yes / val_total_nogold
        print(f"  When gold paragraph NOT present: {false_pos:.1%} judged answerable")
        print(f"  (Expected: low — these are lucky hits from other passages)")

    # Key finding
    print(f"\n  KEY FINDING:")
    print(f"    Total decomposition steps evaluated: {total}")
    print(f"    Steps where fact IS answerable from retrieved passages: {answerable} ({answerable/total*100:.1f}%)")
    print(f"    Steps where fact is NOT answerable: {not_answerable} ({not_answerable/total*100:.1f}%)")

    # String match comparison
    sm_answerable = sum(1 for e in labeled if e['string_match_label'] == 1)
    print(f"\n  Comparison with string match:")
    print(f"    String match says answerable: {sm_answerable} ({sm_answerable/total*100:.1f}%)")
    print(f"    LLM judge says answerable: {answerable} ({answerable/total*100:.1f}%)")

    # Per-hop
    print(f"\n  Per-hop breakdown:")
    for h in sorted(set(e['hop_number'] for e in labeled)):
        hop_ex = [e for e in labeled if e['hop_number'] == h]
        a = sum(1 for e in hop_ex if e['label'] == 'ANSWERABLE')
        print(f"    Hop {h}: {a}/{len(hop_ex)} answerable ({a/len(hop_ex)*100:.1f}%)")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
