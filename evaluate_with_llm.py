#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation for Multi-Hop QA

Methodology:
- Following Ho et al. (2025) and Joren et al. (2025, ICLR), we use an LLM-based
  evaluator to assess answer correctness, as exact match (EM) is not robust to
  syntactic variations in generated answers.
- Uses few-shot binary classification (CORRECT / INCORRECT) with chain-of-thought
  reasoning, which achieves up to 0.85 Pearson correlation with human judgments
  for extractive QA (Ho et al., 2025).
- Few-shot examples are manually curated to cover common edge cases:
  verbose-but-correct, partially correct, completely wrong, and synonym/alias matches.
- We additionally report standard EM and token-level F1 for comparability with
  prior work (Trivedi et al., 2023; Min et al., 2019).

Usage:
    python evaluate_with_llm.py \
        --results results/full_run_all_4.1.json \
        --output results/eval_llm_judge.json \
        --model gpt-4.1-mini \
        --sample_size 0  # 0 = evaluate all examples
"""

import json
import argparse
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from openai import OpenAI


# ──────────────────────────────────────────────────────────────────────
# Few-shot examples for the judge prompt.
# These are manually curated to cover the main edge cases in multi-hop QA
# evaluation, following best practices from Ho et al. (2025) and the
# LLMs-as-Judges survey (Gu et al., 2025):
#   - verbose but correct answers
#   - synonym / alias matches
#   - partial overlaps that are actually wrong
#   - completely wrong answers
#   - date/number formatting variations
#   - correct answer embedded in extra context
# ──────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "question": "When does Real Time with the Pizza Man cast member start again in 2018?",
        "gold": "January 19, 2018",
        "predicted": "Real Time with Bill Maher started again on January 19, 2018, for its sixteenth season.",
        "reasoning": "The predicted answer contains the correct date 'January 19, 2018' which matches the gold answer exactly. The additional context about Bill Maher and the season number does not change the correctness.",
        "verdict": "CORRECT"
    },
    {
        "question": "Who started out his career on adult contemporary radio along with the performer of All That Echoes?",
        "gold": "Michael Bublé",
        "predicted": "Michael Bublé started out his career on adult contemporary radio along with Josh Groban.",
        "reasoning": "The predicted answer identifies Michael Bublé as the person, which matches the gold answer. The mention of Josh Groban is additional context but the core answer entity is correct.",
        "verdict": "CORRECT"
    },
    {
        "question": "What is the river that the Kettle Generating Station is located on a tributary of?",
        "gold": "Hudson's Bay",
        "predicted": "The Kettle Generating Station is located on the Nelson River.",
        "reasoning": "The gold answer is 'Hudson's Bay' (the body of water the Nelson River is a tributary OF). The predicted answer says 'Nelson River', which is the river itself, not what it is a tributary of. This answers a different part of the chain and is therefore incorrect.",
        "verdict": "INCORRECT"
    },
    {
        "question": "The city where WOCA is located is in which part of Florida?",
        "gold": "in Northern Florida",
        "predicted": "Ocala is located in Central Florida.",
        "reasoning": "The gold answer says 'Northern Florida' while the predicted answer says 'Central Florida'. These are different regions. The core factual claim is wrong.",
        "verdict": "INCORRECT"
    },
    {
        "question": "When did the person who ended the Archaemenid Empire by conquering Persia die?",
        "gold": "323 BC",
        "predicted": "Alexander the Great died in 323 BCE.",
        "reasoning": "The predicted answer gives '323 BCE' which is the same date as '323 BC'. BC and BCE are equivalent notations. The answer is correct.",
        "verdict": "CORRECT"
    },
    {
        "question": "Who is the spouse of the actor who plays Paul in Breakfast at Tiffany's?",
        "gold": "Sherry Boucher",
        "predicted": "George Peppard played Paul in Breakfast at Tiffany's. His spouse was Elizabeth Ashley.",
        "reasoning": "The gold answer is 'Sherry Boucher' but the prediction says 'Elizabeth Ashley'. George Peppard had multiple spouses; Elizabeth Ashley was one but Sherry Boucher (his last wife) is the gold answer. The predicted spouse is wrong.",
        "verdict": "INCORRECT"
    },
]


def build_judge_prompt(question: str, gold: str, predicted: str) -> str:
    """
    Construct the few-shot judge prompt.

    Design choices following established best practices:
    - Binary CORRECT/INCORRECT classification (Ho et al., 2025 show this
      outperforms scalar scoring for extractive QA).
    - Chain-of-thought reasoning before the verdict (G-Eval, Liu et al.,
      EMNLP 2023; improves alignment with human judgments).
    - Few-shot examples covering known edge cases (recommended by
      Gu et al., 2025 survey).
    - Explicit rubric defining what counts as correct (reduces ambiguity
      and improves inter-rater consistency).
    """
    prompt = """You are an expert evaluator for a question answering system. Your task is to determine whether a predicted answer is correct by comparing it to the gold (reference) answer.

EVALUATION RUBRIC:
- CORRECT: The predicted answer contains the same factual information as the gold answer, even if it includes additional context, uses different phrasing, or uses equivalent representations (e.g., "323 BC" = "323 BCE", "USA" = "United States").
- INCORRECT: The predicted answer gives different factual information, answers a different aspect of the question, or does not contain the gold answer's key information.

IMPORTANT GUIDELINES:
- Focus on whether the CORE FACTUAL CLAIM matches, not surface-level text overlap.
- A verbose answer that contains the correct information is CORRECT.
- A partial answer that gets the key entity/fact right is CORRECT.
- An answer that contains the right entity but attributes the wrong property is INCORRECT.
- If the predicted answer says "I don't know" or "not enough information", it is INCORRECT.

Here are some examples:

"""
    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"""Question: {ex['question']}
Gold answer: {ex['gold']}
Predicted answer: {ex['predicted']}
Reasoning: {ex['reasoning']}
Verdict: {ex['verdict']}

"""

    prompt += f"""Now evaluate this example:

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}
Reasoning:"""

    return prompt


def parse_verdict(response_text: str) -> str:
    """Extract CORRECT or INCORRECT from the judge's response."""
    text = response_text.strip().upper()
    # Look for the verdict after "Verdict:" if present
    if "VERDICT:" in text:
        after_verdict = text.split("VERDICT:")[-1].strip()
        if "CORRECT" in after_verdict[:20]:
            # Check it's not "INCORRECT"
            if after_verdict.strip().startswith("INCORRECT"):
                return "INCORRECT"
            return "CORRECT"
    # Fallback: look at the last line or last word
    lines = text.strip().split("\n")
    last_line = lines[-1].strip()
    if "INCORRECT" in last_line:
        return "INCORRECT"
    if "CORRECT" in last_line:
        return "CORRECT"
    # Last resort: search the whole response
    if "INCORRECT" in text:
        return "INCORRECT"
    if "CORRECT" in text:
        return "CORRECT"
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────
# Standard metrics (EM / F1) for comparability with prior work
# ──────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Normalize answer string for EM/F1 computation.
    Following the standard normalization from Rajpurkar et al. (2016)."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    return " ".join(s.split())


def compute_em(predicted: str, gold: str) -> float:
    return float(normalize_answer(predicted) == normalize_answer(gold))


def compute_f1(predicted: str, gold: str) -> float:
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ──────────────────────────────────────────────────────────────────────

def evaluate_with_llm_judge(
    results_path: str,
    output_path: str,
    model: str = "gpt-4.1-mini",
    sample_size: int = 0,
    max_retries: int = 5,
):
    """
    Run LLM-as-judge evaluation on QA results.

    Reports three metrics:
    1. LLM-Judge Accuracy: % of examples judged CORRECT by the LLM
    2. Standard EM: Exact match (for comparability with prior work)
    3. Standard F1: Token-level F1 (for comparability with prior work)
    """
    client = OpenAI()

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    results_list = data["results"]
    if sample_size > 0:
        results_list = results_list[:sample_size]

    print(f"Evaluating {len(results_list)} examples with LLM judge ({model})...")
    print()

    evaluated = []
    judge_correct = 0
    total_em = 0.0
    total_f1 = 0.0
    unknown_verdicts = 0

    for r in tqdm(results_list, desc="LLM Judge Evaluation"):
        question = r.get("question", "")
        gold = r.get("gold_answer", "")
        predicted = r.get("predicted_answer", "")

        # Compute standard metrics
        em = compute_em(predicted, gold)
        f1 = compute_f1(predicted, gold)
        total_em += em
        total_f1 += f1

        # LLM judge evaluation with retry
        prompt = build_judge_prompt(question, gold, predicted)
        verdict = "UNKNOWN"
        reasoning = ""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )
                response_text = response.choices[0].message.content.strip()

                # Split reasoning and verdict
                reasoning = response_text
                verdict = parse_verdict(response_text)

                if verdict != "UNKNOWN":
                    break
                else:
                    # Retry if we couldn't parse the verdict
                    if attempt < max_retries - 1:
                        time.sleep(5)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"  Error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  FAILED after {max_retries} attempts: {e}")
                    verdict = "UNKNOWN"

        is_correct = verdict == "CORRECT"
        if verdict == "UNKNOWN":
            unknown_verdicts += 1
        if is_correct:
            judge_correct += 1

        evaluated.append({
            "qid": r.get("qid", ""),
            "question": question,
            "gold_answer": gold,
            "predicted_answer": predicted[:200],
            "em": em,
            "f1": f1,
            "llm_verdict": verdict,
            "llm_reasoning": reasoning[:500],
        })

    n = len(evaluated)
    metrics = {
        "count": n,
        "llm_judge_accuracy": judge_correct / n if n > 0 else 0,
        "llm_judge_correct": judge_correct,
        "standard_em": total_em / n if n > 0 else 0,
        "standard_f1": total_f1 / n if n > 0 else 0,
        "unknown_verdicts": unknown_verdicts,
        "judge_model": model,
    }

    # Save results
    output = {"metrics": metrics, "results": evaluated}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total examples:         {n}")
    print(f"Judge model:            {model}")
    print()
    print(f"LLM-Judge Accuracy:     {metrics['llm_judge_accuracy']:.1%}  ({judge_correct}/{n})")
    print(f"Standard EM:            {metrics['standard_em']:.1%}")
    print(f"Standard F1:            {metrics['standard_f1']:.3f}")
    if unknown_verdicts > 0:
        print(f"Unknown verdicts:       {unknown_verdicts} (could not parse judge response)")
    print()
    print(f"Saved to {output_path}")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Human validation support
# ──────────────────────────────────────────────────────────────────────

def export_for_human_validation(eval_path: str, output_path: str, n: int = 100):
    """
    Export a random sample for human validation of the LLM judge.

    Following best practices (Gu et al., 2025), we validate the LLM judge
    by computing agreement with human judgments on a random subset.
    The recommended sample size is 100-200 examples.
    """
    import random
    random.seed(42)

    with open(eval_path) as f:
        data = json.load(f)

    sample = random.sample(data["results"], min(n, len(data["results"])))

    validation_items = []
    for item in sample:
        validation_items.append({
            "qid": item["qid"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "predicted_answer": item["predicted_answer"],
            "llm_verdict": item["llm_verdict"],
            "human_verdict": "",  # To be filled by human annotator
        })

    with open(output_path, "w") as f:
        json.dump(validation_items, f, indent=2)

    print(f"Exported {len(validation_items)} examples for human validation to {output_path}")
    print("Have your co-author fill in 'human_verdict' as CORRECT or INCORRECT.")


def compute_agreement(validation_path: str):
    """
    Compute Cohen's Kappa between LLM judge and human annotations.
    Standard inter-annotator agreement metric (Cohen, 1960).
    """
    with open(validation_path) as f:
        data = json.load(f)

    # Filter to examples where human annotated
    annotated = [d for d in data if d.get("human_verdict") in ("CORRECT", "INCORRECT")]
    if not annotated:
        print("No human annotations found. Fill in 'human_verdict' field first.")
        return

    llm = [1 if d["llm_verdict"] == "CORRECT" else 0 for d in annotated]
    human = [1 if d["human_verdict"] == "CORRECT" else 0 for d in annotated]

    # Agreement
    agree = sum(1 for l, h in zip(llm, human) if l == h)
    total = len(annotated)
    accuracy = agree / total

    # Cohen's Kappa
    p_o = accuracy  # observed agreement
    p_llm_pos = sum(llm) / total
    p_human_pos = sum(human) / total
    p_e = p_llm_pos * p_human_pos + (1 - p_llm_pos) * (1 - p_human_pos)  # expected agreement
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

    print(f"Human-LLM Agreement:")
    print(f"  Annotated examples: {total}")
    print(f"  Raw agreement:      {accuracy:.1%}")
    print(f"  Cohen's Kappa:      {kappa:.3f}")
    print()
    if kappa > 0.8:
        print("  Interpretation: Almost perfect agreement")
    elif kappa > 0.6:
        print("  Interpretation: Substantial agreement")
    elif kappa > 0.4:
        print("  Interpretation: Moderate agreement")
    else:
        print("  Interpretation: Fair or poor agreement — consider revising the judge prompt")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation for Multi-Hop QA")
    subparsers = parser.add_subparsers(dest="command")

    # Main evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Run LLM judge evaluation")
    eval_parser.add_argument("--results", required=True, help="Path to QA results JSON")
    eval_parser.add_argument("--output", default="results/eval_llm_judge.json", help="Output path")
    eval_parser.add_argument("--model", default="gpt-4.1-mini", help="Judge model")
    eval_parser.add_argument("--sample_size", type=int, default=0, help="0 = all examples")

    # Export for human validation
    export_parser = subparsers.add_parser("export", help="Export sample for human validation")
    export_parser.add_argument("--eval_results", required=True, help="Path to LLM judge results")
    export_parser.add_argument("--output", required=True, help="Output path")
    export_parser.add_argument("--n", type=int, default=100, help="Number of examples to sample")

    # Compute agreement
    agree_parser = subparsers.add_parser("agreement", help="Compute human-LLM agreement")
    agree_parser.add_argument("--validation", required=True, help="Path to annotated validation file")

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_with_llm_judge(
            results_path=args.results,
            output_path=args.output,
            model=args.model,
            sample_size=args.sample_size,
        )
    elif args.command == "export":
        export_for_human_validation(args.eval_results, args.output, args.n)
    elif args.command == "agreement":
        compute_agreement(args.validation)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

