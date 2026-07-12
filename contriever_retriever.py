import json, numpy as np, faiss, torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

class ContrieverRetriever:
    def __init__(self, index_path="models/contriever_musique.index",
                 meta_path="models/contriever_musique_meta.json"):
        print("Loading Contriever retriever...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.model.eval()
        self.index = faiss.read_index(index_path)
        with open(meta_path) as f:
            self.passages = json.load(f)
        print(f"  Loaded {self.index.ntotal} passages")

    def _encode(self, text):
        inputs = self.tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = emb.numpy().astype('float32')
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        emb = self._encode(query)
        scores, indices = self.index.search(emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            p = self.passages[idx]
            results.append({"title": p["title"], "text": p["text"], "score": float(score)})
        return results
