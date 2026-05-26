#!/usr/bin/env python3
"""
Part 3: Elasticsearch Retriever
Wrapper matching simple_multihop_qa.py's retrieve function exactly,
so re-retrievals are directly comparable to original passages.
"""

from elasticsearch import Elasticsearch
from typing import List, Dict


class ESRetriever:
    def __init__(self, index: str = "musique", host: str = "localhost", port: int = 9200):
        self.es = Elasticsearch([f"http://{host}:{port}"], timeout=30)
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k passages. Same query as simple_multihop_qa.py."""
        es_query = {
            "size": k,
            "_source": ["title", "paragraph_text"],
            "query": {
                "bool": {
                    "should": [
                        {"match": {"paragraph_text": query}},
                        {"match": {"title": query}},
                    ]
                }
            }
        }
        result = self.es.search(index=self.index, body=es_query)
        passages = []
        if result.get("hits") and result["hits"].get("hits"):
            for hit in result["hits"]["hits"]:
                passages.append({
                    "title": hit["_source"]["title"],
                    "text": hit["_source"]["paragraph_text"],
                    "score": hit["_score"]
                })
        return passages


if __name__ == "__main__":
    r = ESRetriever()
    print("Test 1: 'UHF movie distributor company'")
    for p in r.retrieve("UHF movie distributor company", k=3):
        print(f"  [{p['score']:.2f}] {p['title']}: {p['text'][:100]}")
    print("\nTest 2: 'Steve Hillage spouse name'")
    for p in r.retrieve("Steve Hillage spouse name", k=3):
        print(f"  [{p['score']:.2f}] {p['title']}: {p['text'][:100]}")
