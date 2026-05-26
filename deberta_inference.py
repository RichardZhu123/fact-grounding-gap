#!/usr/bin/env python3
"""
DeBERTa Inference Helper for Fact-Grounded Answerability

Loads the trained DeBERTa-v3-large fact-grounded answerability classifier
and provides a simple predict() interface.

Usage as a module:
    from deberta_inference import DebertaPredictor
    predictor = DebertaPredictor("models/deberta_fg_v2_best")
    label, prob = predictor.predict(sub_question, passages_text)
    # label: "ANSWERABLE" or "NOT-ANSWERABLE"
    # prob: probability of ANSWERABLE (0.0-1.0)

Standalone test (run on a few examples to verify model loads correctly):
    python deberta_inference.py --model_dir models/deberta_fg_v2_best
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DebertaPredictor:
    """Wrapper for DeBERTa fact-grounded answerability predictor."""

    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading DeBERTa from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Model loaded on {self.device}")

    def format_input(self, sub_question: str, passages_text: str, max_passage_chars: int = 1500) -> str:
        """Format input matching the training format."""
        passages = passages_text[:max_passage_chars]
        return f"Sub-question: {sub_question}\n\nPassages:\n{passages}"

    def predict(self, sub_question: str, passages_text: str) -> tuple:
        """
        Predict answerability for a single (sub_question, passages) pair.
        Returns (label, probability_of_answerable).
        """
        text = self.format_input(sub_question, passages_text)
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            answerable_prob = probs[1].item()

        label = "ANSWERABLE" if answerable_prob >= 0.5 else "NOT-ANSWERABLE"
        return label, answerable_prob

    def predict_batch(self, items: list, batch_size: int = 16) -> list:
        """
        Predict for a list of (sub_question, passages_text) tuples.
        Returns list of (label, probability) tuples.
        """
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            texts = [self.format_input(sq, p) for sq, p in batch]
            inputs = self.tokenizer(
                texts,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                answerable_probs = probs[:, 1].cpu().tolist()

            for prob in answerable_probs:
                label = "ANSWERABLE" if prob >= 0.5 else "NOT-ANSWERABLE"
                results.append((label, prob))

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to trained DeBERTa model")
    parser.add_argument("--test_jsonl", default="results/fact_grounded_final_dev.jsonl",
                        help="JSONL with labeled examples to verify against")
    parser.add_argument("--n", type=int, default=20, help="Number of test examples")
    args = parser.parse_args()

    predictor = DebertaPredictor(args.model_dir)

    # Load some labeled examples
    examples = []
    with open(args.test_jsonl) as f:
        for line in f:
            e = json.loads(line)
            if e['label'] != 'UNKNOWN':
                examples.append(e)
            if len(examples) >= args.n:
                break

    print(f"\nTesting on {len(examples)} examples from {args.test_jsonl}\n")

    correct = 0
    for i, e in enumerate(examples):
        sub_q = e['sub_question']
        passages = e.get('passage_text_combined', '')
        true_label = e['label']

        pred_label, prob = predictor.predict(sub_q, passages)
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        marker = "✓" if is_correct else "✗"
        print(f"{marker} [{i+1}] Pred: {pred_label} ({prob:.2f}) | True: {true_label}")
        print(f"    Sub-Q: {sub_q[:80]}")

    print(f"\nAccuracy on {len(examples)} samples: {correct/len(examples)*100:.1f}%")
    print(f"(Expected ~79% based on dev evaluation)")


if __name__ == "__main__":
    main()
