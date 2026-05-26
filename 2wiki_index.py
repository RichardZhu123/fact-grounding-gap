#!/usr/bin/env python3
"""
Build Elasticsearch index for 2WikiMultihopQA from context paragraphs in train+dev JSON.

Same format as HotpotQA: context is list of [title, [sentences]].
Extracts unique paragraphs (deduplicated by title), joins sentences into
paragraph text, and bulk indexes into ES as '2wikimultihopqa'.
"""

import json
import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


def extract_paragraphs(filepath, seen_titles):
    """Extract unique (title, paragraph_text) pairs from a 2WikiMQA JSON file."""
    print(f"Loading {filepath}...")
    with open(filepath) as f:
        data = json.load(f)

    paragraphs = []
    for q in data:
        for title, sentences in q['context']:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = " ".join(sentences)
            if text.strip():
                paragraphs.append({
                    'title': title,
                    'paragraph_text': text,
                })

    print(f"  Extracted {len(paragraphs)} new unique paragraphs")
    del data
    return paragraphs


def create_index(es, index_name):
    """Create the ES index with the same schema as MuSiQue/HotpotQA."""
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Deleting...")
        es.indices.delete(index=index_name)

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "paragraph_text": {"type": "text"},
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)
    print(f"Created index '{index_name}'")


def index_batch(es, index_name, paragraphs, batch_size=5000):
    """Bulk index paragraphs in batches."""
    total = len(paragraphs)
    indexed = 0

    for start in tqdm(range(0, total, batch_size), desc="Indexing"):
        batch = paragraphs[start:start + batch_size]
        actions = [
            {
                "_index": index_name,
                "_source": {
                    "title": p['title'],
                    "paragraph_text": p['paragraph_text'],
                }
            }
            for p in batch
        ]
        success, errors = bulk(es, actions, raise_on_error=False)
        indexed += success
        if errors:
            print(f"  Batch errors: {len(errors)}")

    return indexed


def main():
    parser = argparse.ArgumentParser(description="Index 2WikiMultihopQA paragraphs into ES")
    parser.add_argument("--es_host", default="localhost")
    parser.add_argument("--es_port", type=int, default=9200)
    parser.add_argument("--index_name", default="2wikimultihopqa")
    parser.add_argument("--dev_path", default="raw_data/2wikimultihopqa/dev.json")
    parser.add_argument("--train_path", default="raw_data/2wikimultihopqa/train.json")
    args = parser.parse_args()

    es = Elasticsearch([f"http://{args.es_host}:{args.es_port}"], timeout=60)

    if not es.ping():
        print("ERROR: Cannot connect to Elasticsearch. Is it running?")
        return

    create_index(es, args.index_name)

    seen_titles = set()
    total_indexed = 0

    # Process dev first (smaller)
    dev_paragraphs = extract_paragraphs(args.dev_path, seen_titles)
    total_indexed += index_batch(es, args.index_name, dev_paragraphs)
    del dev_paragraphs

    # Process train (larger)
    train_paragraphs = extract_paragraphs(args.train_path, seen_titles)
    total_indexed += index_batch(es, args.index_name, train_paragraphs)
    del train_paragraphs

    # Re-enable refresh
    es.indices.put_settings(
        index=args.index_name,
        body={"index": {"refresh_interval": "1s"}}
    )
    es.indices.refresh(index=args.index_name)

    # Verify
    count = es.count(index=args.index_name)['count']
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Unique titles seen: {len(seen_titles)}")
    print(f"Documents indexed: {total_indexed}")
    print(f"ES document count: {count}")
    print(f"Index: {args.index_name}")

    # Sanity test
    result = es.search(
        index=args.index_name,
        body={
            "size": 3,
            "_source": ["title", "paragraph_text"],
            "query": {"match": {"paragraph_text": "Polish Russian War director"}}
        }
    )
    print(f"\nSanity test query: 'Polish Russian War director'")
    for hit in result['hits']['hits']:
        print(f"  [{hit['_score']:.1f}] {hit['_source']['title']}: {hit['_source']['paragraph_text'][:80]}...")


if __name__ == "__main__":
    main()
