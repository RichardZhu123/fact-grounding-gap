#!/usr/bin/env python3
"""
Query Reformulation Helper

When DeBERTa flags a hop as NOT-ANSWERABLE, this module asks GPT-4.1-mini
to generate a more targeted search query that addresses the missing fact.

Usage as a module:
    from query_reformulator import QueryReformulator
    reformulator = QueryReformulator()
    new_query = reformulator.reformulate(sub_question, failed_passages)

Standalone test:
    python query_reformulator.py
"""

import time
from openai import OpenAI


REFORMULATION_PROMPT = """You are helping a multi-hop question answering system.
A retrieval step failed: the retrieved passages do not contain the information needed
to answer this sub-question. Generate a better, more targeted search query.

Sub-question: {sub_question}

Retrieved passages (which DO NOT answer the sub-question):
{passages}

Analyze why the retrieval failed (e.g., wrong entity disambiguation, missing key term,
too vague, etc.) and produce a SHORT search query (3-8 words) that would better target
the specific fact needed. Focus on the key entity and relation.

Output ONLY the new search query, nothing else. No quotes, no explanation."""


class QueryReformulator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model

    def reformulate(self, sub_question: str, failed_passages: str, max_retries: int = 3) -> str:
        """Generate a better search query for a failed retrieval."""
        # Truncate passages to keep prompt size reasonable
        passages_truncated = failed_passages[:2000]
        prompt = REFORMULATION_PROMPT.format(
            sub_question=sub_question,
            passages=passages_truncated,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50,
                )
                new_query = response.choices[0].message.content.strip()
                # Strip quotes if present
                new_query = new_query.strip('"').strip("'")
                return new_query
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    # Fallback: return original sub-question
                    return sub_question

        return sub_question


def main():
    """Test reformulation on a few examples."""
    reformulator = QueryReformulator()

    test_cases = [
        {
            "sub_question": "Which company distributed UHF?",
            "passages": "[1] UHF: UHF refers to ultra-high frequency radio waves between 300 MHz and 3 GHz. Used for TV, cell phones, satellites.\n[2] World Film Company: A film production company organized in 1914.",
        },
        {
            "sub_question": "Who is the spouse of Steve Hillage?",
            "passages": "[1] Gong (band): Gong is a rock band founded by Daevid Allen in 1967. Steve Hillage joined as guitarist in 1973.",
        },
        {
            "sub_question": "Where was Trey Parker born?",
            "passages": "[1] South Park: South Park is an American animated sitcom created by Trey Parker and Matt Stone. Premiered August 13, 1997.",
        },
    ]

    print("Testing query reformulation:\n")
    for i, case in enumerate(test_cases, 1):
        print(f"--- Test {i} ---")
        print(f"Original sub-question: {case['sub_question']}")
        print(f"Failed passages: {case['passages'][:200]}...")
        new_query = reformulator.reformulate(case['sub_question'], case['passages'])
        print(f"Reformulated query: {new_query}")
        print()


if __name__ == "__main__":
    main()
