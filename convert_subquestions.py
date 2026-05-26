#!/usr/bin/env python3
"""
Convert MuSiQue Shorthand Sub-Questions to Natural Language

MuSiQue's question_decomposition uses two formats:
  - Natural language: "Who founded Australia's liberal party?" (keep as-is)
  - Shorthand: "UHF >> distributed by" (needs conversion)

This script:
1. Resolves #N references (e.g., "#1 >> spouse" -> "Steve Hillage >> spouse")
2. Converts shorthand to natural language via GPT-4.1-mini
3. Saves a lookup file: {question_id -> [resolved_sub_questions]}
4. Saves conversion mapping separately for auditing

Usage:
    python convert_subquestions.py \
        --input raw_data/musique/musique_ans_v1.0_dev.jsonl \
        --output results/musique_dev_subquestions.json

    python convert_subquestions.py \
        --input raw_data/musique/musique_ans_v1.0_train.jsonl \
        --output results/musique_train_subquestions.json
"""

import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


def resolve_references(question_decomposition: list) -> list:
    """
    Resolve #N references using previous hop answers.
    Returns list of resolved sub-question strings.
    """
    resolved = []
    for i, step in enumerate(question_decomposition):
        q = step['question']
        for prev_idx in range(i):
            ref = f"#{prev_idx + 1}"
            prev_answer = question_decomposition[prev_idx]['answer']
            q = q.replace(ref, prev_answer)
        resolved.append(q)
    return resolved


def is_shorthand(q: str) -> bool:
    """Check if a sub-question is in shorthand format."""
    return '>>' in q


def convert_shorthand_llm(client, shorthand_questions: list, model: str = "gpt-4.1-mini") -> dict:
    """
    Convert shorthand sub-questions to natural language using LLM.
    Deduplicates first to minimize API calls.
    Returns dict mapping shorthand -> natural language.
    """
    conversions = {}

    # Deduplicate
    unique_questions = list(set(shorthand_questions))
    print(f"  {len(unique_questions)} unique shorthand questions to convert")

    prompt_template = """Convert the following shorthand sub-question into a natural language question.

The shorthand format is "entity >> relation" which means "What is the [relation] of [entity]?"

Examples:
- "UHF >> distributed by" -> "Which company distributed UHF?"
- "Steve Hillage >> spouse" -> "Who is the spouse of Steve Hillage?"
- "Green >> performer" -> "Who is the performer of Green?"
- "German Aerospace Center >> headquarters location" -> "Where is the headquarters of German Aerospace Center?"
- "Adelphia Communications Corporation >> followed by" -> "What company followed Adelphia Communications Corporation?"
- "Learjet 60 >> manufacturer" -> "Who manufactures the Learjet 60?"
- "Ciudad Deportiva >> owned by" -> "Who owns Ciudad Deportiva?"
- "Denver >> located in the administrative territorial entity" -> "What administrative territorial entity is Denver located in?"

Now convert this:
"{question}"

Output ONLY the natural language question, nothing else."""

    for q in tqdm(unique_questions, desc="  Converting"):
        prompt = prompt_template.format(question=q)

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                natural_q = response.choices[0].message.content.strip()
                # Remove quotes if present
                natural_q = natural_q.strip('"').strip("'")
                # Make sure it ends with ?
                if not natural_q.endswith('?'):
                    natural_q += '?'
                conversions[q] = natural_q
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(10)
                else:
                    # Fallback: simple template conversion
                    parts = q.split('>>')
                    if len(parts) == 2:
                        entity = parts[0].strip()
                        relation = parts[1].strip()
                        conversions[q] = f"What is the {relation} of {entity}?"
                    else:
                        conversions[q] = q

    return conversions


def main():
    parser = argparse.ArgumentParser(description="Convert MuSiQue shorthand sub-questions")
    parser.add_argument("--input", required=True, help="MuSiQue JSONL file")
    parser.add_argument("--output", required=True, help="Output JSON lookup file")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model for conversion")
    args = parser.parse_args()

    client = OpenAI()

    # Load data
    print(f"Loading {args.input}...")
    examples = []
    with open(args.input) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  {len(examples)} examples")

    # Step 1: Resolve #N references for all examples
    print("Resolving #N references...")
    all_resolved = {}
    shorthand_to_convert = []

    for ex in examples:
        qid = ex['id']
        decomp = ex['question_decomposition']
        resolved = resolve_references(decomp)
        all_resolved[qid] = {
            'question': ex['question'],
            'resolved_sub_questions': resolved,
            'intermediate_answers': [step['answer'] for step in decomp],
            'paragraph_support_idxs': [step['paragraph_support_idx'] for step in decomp],
        }

        for q in resolved:
            if is_shorthand(q):
                shorthand_to_convert.append(q)

    total_sub_q = sum(len(v['resolved_sub_questions']) for v in all_resolved.values())
    print(f"  Total resolved sub-questions: {total_sub_q}")
    print(f"  Shorthand needing conversion: {len(shorthand_to_convert)}")
    print(f"  Already natural language: {total_sub_q - len(shorthand_to_convert)}")

    # Step 2: Convert shorthand to natural language
    if shorthand_to_convert:
        print("\nConverting shorthand to natural language...")
        conversions = convert_shorthand_llm(client, shorthand_to_convert, args.model)
        print(f"  Converted {len(conversions)} unique shorthand questions")

        # Save conversions separately for auditing
        conversions_path = args.output.replace('.json', '_conversions.json')
        with open(conversions_path, 'w') as cf:
            json.dump(conversions, cf, indent=2)
        print(f"  Saved conversion mapping to {conversions_path}")

        # Apply conversions
        for qid, data in all_resolved.items():
            natural_sub_questions = []
            for q in data['resolved_sub_questions']:
                if is_shorthand(q) and q in conversions:
                    natural_sub_questions.append(conversions[q])
                else:
                    natural_sub_questions.append(q)
            data['natural_sub_questions'] = natural_sub_questions
    else:
        for qid, data in all_resolved.items():
            data['natural_sub_questions'] = data['resolved_sub_questions']

    # Step 3: Save lookup file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_resolved, f, indent=2)

    print(f"\nSaved to {args.output}")

    # Print samples
    print("\nSample conversions:")
    count = 0
    for qid, data in all_resolved.items():
        for orig, natural in zip(data['resolved_sub_questions'], data['natural_sub_questions']):
            if orig != natural:
                print(f"  {orig}")
                print(f"  -> {natural}")
                print()
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break


if __name__ == "__main__":
    main()
