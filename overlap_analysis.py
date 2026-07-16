import json

def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d[(r['qid'], r['hop_number'])] = r
    return d

bm25 = load('results/fact_grounded_final_dev.jsonl')
cont = load('results/contriever_fact_grounded.jsonl')

shared = set(bm25) & set(cont)
print(f"Shared hops: {len(shared)}")

both_gold = [k for k in shared if bm25[k]['gold_para_present'] and cont[k]['gold_para_present']]
print(f"Gold present under both: {len(both_gold)}")

bm25_ext = {k for k in both_gold if bm25[k]['label_binary'] == 0}
cont_ext = {k for k in both_gold if cont[k]['label_binary'] == 0}
overlap = bm25_ext & cont_ext
print(f"BM25 extraction failures (gold present under both): {len(bm25_ext)}")
print(f"Contriever extraction failures (gold present under both): {len(cont_ext)}")
print(f"Overlap: {len(overlap)}")
if bm25_ext:
    print(f"Of BM25 ext failures, also ext failures under Contriever: {len(overlap)/len(bm25_ext)*100:.1f}%")
if cont_ext:
    print(f"Of Contriever ext failures, also under BM25: {len(overlap)/len(cont_ext)*100:.1f}%")
