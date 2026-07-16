import json, numpy as np, faiss, torch
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch
from tqdm import tqdm

print("Loading Contriever model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever")
model.eval()

print("Fetching HotpotQA passages from Elasticsearch...")
es = Elasticsearch()
passages = []
batch_size = 1000
result = es.search(index="hotpotqa", body={"query": {"match_all": {}}, "size": batch_size}, scroll="5m")
scroll_id = result['_scroll_id']
while True:
    hits = result['hits']['hits']
    if not hits:
        break
    for h in hits:
        passages.append({"title": h['_source'].get('title',''), "text": h['_source'].get('paragraph_text','')})
    result = es.scroll(scroll_id=scroll_id, scroll="5m")
print(f"Loaded {len(passages)} passages")

print("Encoding passages...")
embeddings = []
batch = 32
for i in tqdm(range(0, len(passages), batch), desc="Encoding"):
    texts = [p['text'][:1000] for p in passages[i:i+batch]]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = emb.numpy().astype('float32')
    faiss.normalize_L2(emb)
    embeddings.append(emb)

all_emb = np.vstack(embeddings)
print(f"Building FAISS index: {all_emb.shape}")
index = faiss.IndexFlatIP(all_emb.shape[1])
index.add(all_emb)
faiss.write_index(index, "models/contriever_hotpotqa.index")
with open("models/contriever_hotpotqa_meta.json", "w") as f:
    json.dump(passages, f)
print("Done!")
