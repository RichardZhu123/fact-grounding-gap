#!/usr/bin/env python3
"""
Simple Self-Ask style Multi-Hop QA System
Uses Elasticsearch for retrieval and GPT-4.1-mini for reasoning.
Logs full trajectories for analysis.
"""
import json
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from contriever_retriever import ContrieverRetriever
from openai import OpenAI


@dataclass
class RetrievalStep:
    """A single retrieval step in the multi-hop process."""
    hop_number: int
    query: str
    retrieved_passages: List[Dict]
    reasoning: str


@dataclass
class Trajectory:
    """Full trajectory of a multi-hop QA run."""
    question: str
    steps: List[RetrievalStep]
    final_answer: str
    gold_answer: Optional[str] = None


class SimpleMultiHopQA:
    """Self-Ask style multi-hop QA system."""

    def __init__(
        self,
        es_host: str = "localhost",
        es_port: int = 9200,
        corpus_name: str = "musique",
        model: str = "gpt-4.1-mini",
        max_hops: int = 4,
        retrieval_count: int = 3,
    ):
        self.retriever = ContrieverRetriever()
        self.client = OpenAI()
        self.corpus_name = corpus_name
        self.model = model
        self.max_hops = max_hops
        self.retrieval_count = retrieval_count

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve passages from Contriever."""
        return self.retriever.retrieve(query, k=3)

    def format_context(self, passages: List[Dict]) -> str:
        """Format retrieved passages as context string."""
        if not passages:
            return "No passages retrieved."
        context_parts = []
        for i, p in enumerate(passages, 1):
            context_parts.append(f"[{i}] {p['title']}: {p['text']}")
        return "\n\n".join(context_parts)

    def reason_and_decide(
        self,
        question: str,
        context: str,
        previous_reasoning: List[str]
    ) -> tuple:
        """
        Ask LLM to reason and decide next step.
        Returns (reasoning, search_query, is_done).
        """
        prev_reasoning_str = ""
        if previous_reasoning:
            prev_reasoning_str = "\n\nPrevious reasoning:\n" + "\n".join(previous_reasoning)

        prompt = f"""You are answering a multi-hop question that may require multiple pieces of information.
Question: {question}
Retrieved Information:
{context}
{prev_reasoning_str}
Based on the information above, do ONE of the following:
1. If you have enough information to answer the question, respond with:
ANSWER: [your concise answer]
2. If you need more information, respond with:
SEARCH: [a specific search query to find the missing information]
REASONING: [brief explanation of what you're looking for and why]
Be concise. If you can answer, give just the answer. If you need to search, give a focused query."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        output = response.choices[0].message.content.strip()

        if output.startswith("ANSWER:"):
            return output, None, True
        elif "SEARCH:" in output:
            lines = output.split("\n")
            search_query = None
            for line in lines:
                if line.startswith("SEARCH:"):
                    search_query = line[7:].strip()
                    break
            return output, search_query, False
        else:
            return output, None, True

    def answer(self, question: str, gold_answer: Optional[str] = None) -> Trajectory:
        """Answer a multi-hop question using Self-Ask style reasoning."""
        steps = []
        all_passages = []
        previous_reasoning = []
        current_query = question

        for hop in range(self.max_hops):
            passages = self.retrieve(current_query)
            all_passages.extend(passages)

            seen_texts = set()
            unique_passages = []
            for p in all_passages:
                if p["text"] not in seen_texts:
                    seen_texts.add(p["text"])
                    unique_passages.append(p)
            all_passages = unique_passages

            context = self.format_context(all_passages)
            reasoning, next_query, is_done = self.reason_and_decide(
                question, context, previous_reasoning
            )

            step = RetrievalStep(
                hop_number=hop + 1,
                query=current_query,
                retrieved_passages=passages,
                reasoning=reasoning,
            )
            steps.append(step)
            previous_reasoning.append(reasoning)

            if is_done:
                if "ANSWER:" in reasoning:
                    final_answer = reasoning.split("ANSWER:")[1].strip()
                else:
                    final_answer = reasoning
                return Trajectory(
                    question=question,
                    steps=steps,
                    final_answer=final_answer,
                    gold_answer=gold_answer,
                )
            current_query = next_query

        # Max hops reached, force an answer
        context = self.format_context(all_passages)
        prompt = f"""Question: {question}
Information:
{context}
Based on all the information above, provide your best answer. Be concise.
Answer:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        final_answer = response.choices[0].message.content.strip()
        return Trajectory(
            question=question,
            steps=steps,
            final_answer=final_answer,
            gold_answer=gold_answer,
        )

    def answer_with_interventions(
        self,
        question: str,
        interventions: Optional[Dict[int, List[Dict]]] = None,
        gold_answer: Optional[str] = None,
    ) -> Trajectory:
        """
        Same as answer() but accepts a per-hop intervention dict.
        interventions: {hop_number: [extra_passages]}
        At each hop, after normal retrieval, any extra passages for that hop
        are added to all_passages BEFORE reasoning.
        """
        if interventions is None:
            interventions = {}

        steps = []
        all_passages = []
        previous_reasoning = []
        current_query = question

        for hop in range(self.max_hops):
            hop_num = hop + 1

            # Normal retrieval
            passages = self.retrieve(current_query)
            all_passages.extend(passages)

            # Inject intervention passages for this hop, if any
            if hop_num in interventions:
                all_passages.extend(interventions[hop_num])

            # Dedupe by text
            seen_texts = set()
            unique_passages = []
            for p in all_passages:
                if p["text"] not in seen_texts:
                    seen_texts.add(p["text"])
                    unique_passages.append(p)
            all_passages = unique_passages

            context = self.format_context(all_passages)
            reasoning, next_query, is_done = self.reason_and_decide(
                question, context, previous_reasoning
            )

            step = RetrievalStep(
                hop_number=hop_num,
                query=current_query,
                retrieved_passages=passages,
                reasoning=reasoning,
            )
            steps.append(step)
            previous_reasoning.append(reasoning)

            if is_done:
                if "ANSWER:" in reasoning:
                    final_answer = reasoning.split("ANSWER:")[1].strip()
                else:
                    final_answer = reasoning
                return Trajectory(
                    question=question,
                    steps=steps,
                    final_answer=final_answer,
                    gold_answer=gold_answer,
                )
            current_query = next_query

        # Max hops reached, force an answer
        context = self.format_context(all_passages)
        prompt = f"""Question: {question}
Information:
{context}
Based on all the information above, provide your best answer. Be concise.
Answer:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        final_answer = response.choices[0].message.content.strip()
        return Trajectory(
            question=question,
            steps=steps,
            final_answer=final_answer,
            gold_answer=gold_answer,
        )


def main():
    parser = argparse.ArgumentParser(description="Simple Multi-Hop QA")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument("--corpus", type=str, default="musique")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--es-host", type=str, default="localhost")
    parser.add_argument("--es-port", type=int, default=9200)
    args = parser.parse_args()

    if not args.question:
        args.question = "When was the employer of Neville A. Stanton founded?"

    qa = SimpleMultiHopQA(
        es_host=args.es_host,
        es_port=args.es_port,
        corpus_name=args.corpus,
        model=args.model,
        max_hops=args.max_hops,
    )

    print(f"Question: {args.question}\n")
    trajectory = qa.answer(args.question)
    print("=" * 60)
    for step in trajectory.steps:
        print(f"\n--- Hop {step.hop_number} ---")
        print(f"Query: {step.query}")
        print(f"Retrieved {len(step.retrieved_passages)} passages:")
        for p in step.retrieved_passages:
            print(f"  - {p['title']}: {p['text'][:100]}...")
        print(f"Reasoning: {step.reasoning}")
    print("\n" + "=" * 60)
    print(f"Final Answer: {trajectory.final_answer}")
    print("\n--- JSON Output ---")
    print(json.dumps(asdict(trajectory), indent=2))


if __name__ == "__main__":
    main()
