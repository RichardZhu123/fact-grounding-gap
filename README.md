# Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering

Code for the paper *Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering*.

We measure, per reasoning hop, whether retrieved passages actually contain the fact a hop
needs — distinguishing **retrieval failures** (gold passage not retrieved) from **extraction
failures** (gold passage retrieved but the needed fact is absent). The pipeline is a Self-Ask
loop using GPT-4.1-mini for reasoning and BM25 over Elasticsearch for retrieval, with a
DeBERTa fact-presence predictor used to flag deficient hops for targeted re-retrieval.

> This repository began as a fork of [IRCoT](https://github.com/StonyBrookNLP/ircot); the
> IRCoT pipeline has been removed and only the components used by this paper remain.

# Installation

```bash
conda create -n fgg python=3.8.0 -y && conda activate fgg
pip install -r requirements.txt
```

Set your OpenAI key (GPT-4.1-mini is used for the Self-Ask reasoner and the LLM fact-presence judge):

```bash
export OPENAI_API_KEY=...
```

# Prepare Data

Download the raw datasets (MuSiQue, HotpotQA, 2WikiMultihopQA):

```bash
./download/raw_data.sh
```

Data is downloaded to `raw_data/{dataset_name}/`. The HotpotQA and 2WikiMultihopQA scripts read
from `raw_data/` directly. For MuSiQue, generate the processed split used by the labeling scripts:

```bash
python processing_scripts/process_musique.py   # writes processed_data/musique/{train,dev}.jsonl
```

# Elasticsearch Setup and Indexing

The retriever is BM25 over Elasticsearch. Install Elasticsearch 7.10 and start it on the default
port (9200).

<details>
<summary>Install Elasticsearch 7.10</summary>

### Mac (Homebrew)
```bash
# source: https://www.elastic.co/guide/en/elasticsearch/reference/current/brew.html
brew tap elastic/tap
brew install elastic/tap/elasticsearch-full
brew services start elastic/tap/elasticsearch-full   # start
brew services stop elastic/tap/elasticsearch-full    # stop
```

### Mac (tarball)
```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-darwin-x86_64.tar.gz
tar -xzf elasticsearch-7.10.2-darwin-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch          # start
pkill -f elasticsearch       # stop
```

### Linux
```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch          # start
pkill -f elasticsearch       # stop
```

</details>

Build the indices (one Elasticsearch index per dataset, queried by `es_retriever.py` and
`simple_multihop_qa.py`):

```bash
# MuSiQue (primary dataset)
python retriever_server/build_index.py musique

# HotpotQA and 2WikiMultihopQA
python hotpotqa_index.py
python 2wiki_index.py
```

Check the indices with `curl localhost:9200/_cat/indices`.

# Running the Pipeline

The Self-Ask QA pipeline lives in `simple_multihop_qa.py`; `run_multihop_eval.py` runs it over a
dataset and saves trajectories. Fact-presence labeling, the DeBERTa predictor, and the
intervention experiments are driven by the top-level scripts (e.g. `fact_grounding.py`,
`llm_fact_grounding.py`, `train_deberta.py`, `deberta_inference.py`, and the `section4_*.py`
experiments). Each script documents its expected arguments in a module docstring at the top of
the file.
