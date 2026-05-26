#!/usr/bin/env python3
"""
Phase 1: Build Contriever FAISS index for MuSiQue passages.
Run separately before the experiment (no DeBERTa/cross-encoder loaded).
Saves embeddings + metadata to disk.
"""

import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch
from tqdm import tqdm


def mean_pooling(token_embeddings, attention_mask):
    """Contriever uses mean pooling."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_batch(model, tokenizer, texts, batch_size=32):
    """Encode a list of texts into embeddings."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        all_embeddings.append(embeddings.numpy())
    return np.vstack(all_embeddings)


def main():
    print("Loading Contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")
    model.eval()
    print("  Model loaded")

    print("Loading passages from Elasticsearch...")
    es = Elasticsearch(['http://localhost:9200'], timeout=60)

    # Scroll through all passages
    passages = []
    result = es.search(
        index='musique',
        body={"size": 1000, "_source": ["title", "paragraph_text"], "query": {"match_all": {}}},
        scroll='5m'
    )
    scroll_id = result['_scroll_id']
    hits = result['hits']['hits']

    while hits:
        for hit in hits:
            passages.append({
                'id': hit['_id'],
                'title': hit['_source']['title'],
                'text': hit['_source']['paragraph_text'],
            })
        result = es.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = result['_scroll_id']
        hits = result['hits']['hits']

    print(f"  Loaded {len(passages)} passages")

    # Encode passages in batches
    print("Encoding passages (this takes ~50 min on CPU)...")
    texts = [f"{p['title']}: {p['text']}" for p in passages]

    # Process in chunks to manage memory
    chunk_size = 5000
    all_embeddings = []
    for start in tqdm(range(0, len(texts), chunk_size), desc="Encoding"):
        chunk = texts[start:start + chunk_size]
        embeddings = encode_batch(model, tokenizer, chunk, batch_size=16)
        all_embeddings.append(embeddings)

    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"  Embeddings shape: {embeddings_matrix.shape}")

    # Normalize for inner product search (cosine similarity)
    faiss.normalize_L2(embeddings_matrix)

    # Build FAISS index
    print("Building FAISS index...")
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    print(f"  Index size: {index.ntotal} vectors")

    # Save index + metadata
    faiss.write_index(index, "models/contriever_musique.index")

    # Save passage metadata (id, title, text) for retrieval
    with open("models/contriever_musique_meta.json", 'w') as f:
        json.dump(passages, f)

    print(f"\nSaved:")
    print(f"  Index: models/contriever_musique.index")
    print(f"  Metadata: models/contriever_musique_meta.json")
    print(f"  Total passages: {len(passages)}")

    # Quick sanity test
    print("\nSanity test: 'Who directed the film UHF?'")
    query_emb = encode_batch(model, tokenizer, ["Who directed the film UHF?"], batch_size=1)
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, 5)
    for score, idx in zip(scores[0], indices[0]):
        p = passages[idx]
        print(f"  [{score:.3f}] {p['title']}: {p['text'][:80]}...")


if __name__ == "__main__":
    main()
