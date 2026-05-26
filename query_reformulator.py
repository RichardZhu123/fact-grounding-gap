#!/usr/bin/env python3
"""
Part 2 (v3): Query Reformulator without failed-passage context.

Takes the original full question + current sub-question and generates a
disambiguated search query. Does NOT use failed passages as context, to
avoid entity poisoning (where the reformulator copies wrong entities from
the failed retrievals into the new query).
"""

import os
from openai import OpenAI


REFORMULATION_PROMPT = """The original multi-hop question is: "{original_question}"
The current sub-question to search for is: "{sub_question}"

Generate a SHORT search query (3-8 words) that focuses ONLY on what the sub-question asks.
Do NOT include words from other parts of the original question that aren't relevant to
this specific sub-question. Add DISTINCTIVE disambiguators (years, domain terms, alternate
names, proper nouns) that distinguish the correct entity from common name collisions.

Output ONLY the new search query, nothing else."""


class QueryReformulator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model

    def reformulate(self, original_question: str, sub_question: str, failed_passages=None) -> str:
        """
        Generate a focused search query for the given sub-question.

        failed_passages parameter is kept for backwards compatibility with
        older callers but is intentionally NOT passed to the LLM, to prevent
        entity poisoning from incorrect retrievals.
        """
        prompt = REFORMULATION_PROMPT.format(
            original_question=original_question,
            sub_question=sub_question,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip().strip('"').strip("'")


if __name__ == "__main__":
    r = QueryReformulator()

    print("=== Test 1: Green step 1 (was previously poisoned by Joni Mitchell passages) ===")
    q = r.reformulate(
        original_question="Who is the spouse of the Green performer?",
        sub_question="Who is the performer of Green?",
    )
    print(f"Query: {q}\n")

    print("=== Test 2: Steve Hillage step 2 ===")
    q = r.reformulate(
        original_question="Who is the spouse of the Green performer?",
        sub_question="Who is the spouse of Steve Hillage?",
    )
    print(f"Query: {q}\n")

    print("=== Test 3: UHF step 1 (sanity check) ===")
    q = r.reformulate(
        original_question="Who founded the company that distributed the film UHF?",
        sub_question="Which company distributed UHF?",
    )
    print(f"Query: {q}\n")

    print("=== Test 4: Orion Pictures step 2 (bridge entity) ===")
    q = r.reformulate(
        original_question="Who founded the company that distributed the film UHF?",
        sub_question="Who founded Orion Pictures?",
    )
    print(f"Query: {q}\n")
