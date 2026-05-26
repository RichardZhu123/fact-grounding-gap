#!/usr/bin/env python3
"""
DeBERTa-v3-large Fine-Tuning for Fact-Grounded Answerability

Fine-tunes microsoft/deberta-v3-large as a binary classifier to predict
whether retrieved passages contain enough information to answer a
sub-question in multi-hop QA.

Uses standard HuggingFace Trainer with AutoModelForSequenceClassification.

Usage:
    python train_deberta.py \
        --train_csv data/deberta_fg_train.csv \
        --dev_csv data/deberta_fg_dev.csv \
        --output_dir models/deberta_fact_grounded \
        --epochs 10 \
        --batch_size 16 \
        --learning_rate 2e-5

Requirements (install on GPU machine):
    pip install transformers==4.40.0 datasets torch scikit-learn accelerate sentencepiece protobuf
"""

import argparse
import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ──────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────

class SufficiencyDataset(Dataset):
    """Simple dataset for text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
        }


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_csv(path: str):
    """Load CSV with (text, label) columns."""
    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(int(row['label']))
    return texts, labels


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for the Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────

def train(args):
    print("=" * 60)
    print("DeBERTa-v3-large Fact-Grounded Answerability Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading training data from {args.train_csv}...")
    train_texts, train_labels = load_csv(args.train_csv)
    print(f"  {len(train_texts)} examples, {sum(train_labels)} answerable ({sum(train_labels)/len(train_labels)*100:.1f}%)")

    print(f"Loading dev data from {args.dev_csv}...")
    dev_texts, dev_labels = load_csv(args.dev_csv)
    print(f"  {len(dev_texts)} examples, {sum(dev_labels)} answerable ({sum(dev_labels)/len(dev_labels)*100:.1f}%)")

    # Load tokenizer and model
    model_name = args.model_name
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NOT-ANSWERABLE", 1: "ANSWERABLE"},
        label2id={"NOT-ANSWERABLE": 0, "ANSWERABLE": 1},
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenize data
    print(f"\nTokenizing (max_length={args.max_length})...")
    train_dataset = SufficiencyDataset(train_texts, train_labels, tokenizer, args.max_length)
    dev_dataset = SufficiencyDataset(dev_texts, dev_labels, tokenizer, args.max_length)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Dev: {len(dev_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to='none',
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup: 10%")
    print(f"  Best model metric: F1")
    print(f"  Early stopping patience: 3")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    trainer.train()

    # Evaluate
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON DEV SET")
    print("=" * 60)

    results = trainer.evaluate()
    print(f"  Accuracy:  {results['eval_accuracy']:.1%}")
    print(f"  F1:        {results['eval_f1']:.3f}")
    print(f"  Precision: {results['eval_precision']:.3f}")
    print(f"  Recall:    {results['eval_recall']:.3f}")
    print(f"  Loss:      {results['eval_loss']:.4f}")

    # Save model
    trainer.save_model(os.path.join(args.output_dir, 'best_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
    print(f"\nModel saved to {os.path.join(args.output_dir, 'best_model')}")

    # Save results
    import json
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model': model_name,
            'train_examples': len(train_texts),
            'dev_examples': len(dev_texts),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'best_model_metric': 'f1',
            'eval_accuracy': results['eval_accuracy'],
            'eval_f1': results['eval_f1'],
            'eval_precision': results['eval_precision'],
            'eval_recall': results['eval_recall'],
            'eval_loss': results['eval_loss'],
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for fact-grounded answerability")
    parser.add_argument("--train_csv", required=True, help="Training CSV (text, label)")
    parser.add_argument("--dev_csv", required=True, help="Dev CSV (text, label)")
    parser.add_argument("--output_dir", default="models/deberta_fact_grounded")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
