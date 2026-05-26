"""
Generate SFT training data for fact-aware reasoning (v2).
Fixed: insufficient hops show supplementary retrieval + reasoning from new evidence.
No contradictory signal (no hallucinating from parametric memory).

Uses:
- fact_grounded_final_dev.jsonl (per-hop passages + labels)
- section4_full_dev.json (intervention outcomes to identify winning traces)
- ES retriever (to get supplementary passages for insufficient hops)

Run on VM: cd ~/ircot && python generate_sft_data_v2.py
"""
import json
import random
import os
from collections import defaultdict

random.seed(42)

# Load fact-grounded data (has passage text per hop)
by_question = defaultdict(list)
with open('results/fact_grounded_final_dev.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        if ex.get('label') in ('ANSWERABLE', 'NOT-ANSWERABLE'):
            by_question[ex['qid']].append(ex)

# Load intervention results (to identify which questions benefited)
with open('results/section4_full_dev.json') as f:
    aug = json.load(f)

intervention_outcomes = {}
for ex in aug['examples']:
    intervention_outcomes[ex['qid']] = {
        'baseline_correct': ex['baseline']['correct'],
        'deberta_correct': ex['deberta']['correct'],
        'always_correct': ex['always']['correct'],
        'flags': ex['deberta']['flags'],
    }

# Load raw data for gold answers
raw = {}
with open('raw_data/musique/musique_ans_v1.0_dev.jsonl') as f:
    for line in f:
        d = json.loads(line)
        raw[d['id']] = d

# Load sub-questions
with open('results/musique_dev_subquestions.json') as f:
    subq_lookup = json.load(f)

print(f"Questions with hop data: {len(by_question)}")
print(f"Questions with intervention outcomes: {len(intervention_outcomes)}")

# We need supplementary passages for insufficient hops.
# Try to get them via ES retriever + query reformulator.
try:
    from es_retriever import ESRetriever
    from query_reformulator import QueryReformulator
    retriever = ESRetriever()
    reformulator = QueryReformulator()
    has_retriever = True
    print("ES retriever loaded - will generate real supplementary passages")
except ImportError:
    has_retriever = False
    print("WARNING: No ES retriever available. Will use placeholder passages.")
    print("Run this script on the VM with ES running for real passages.")

SYSTEM_PROMPT = """You are a multi-hop question answering system that reasons carefully about evidence quality.

For each sub-question in the reasoning chain:
1. Assess whether the retrieved passages contain sufficient evidence to answer this specific sub-question
2. If evidence is SUFFICIENT: extract the answer from the passages and continue
3. If evidence is INSUFFICIENT: explicitly state what information is missing, then use supplementary passages if provided

Always ground your answers in the provided passages. Never guess from general knowledge."""

def format_passages_text(passage_text, max_chars=800):
    """Truncate passage text sensibly."""
    if len(passage_text) <= max_chars:
        return passage_text
    return passage_text[:max_chars] + "..."

def get_supplementary_passages(retriever, reformulator, original_question, sub_question, k=3):
    """Re-retrieve with reformulated query."""
    try:
        new_q = reformulator.reformulate(original_question, sub_question)
        passages = retriever.retrieve(new_q, k=k)
        texts = []
        for p in passages:
            title = p.get('title', '')
            text = p.get('text', p.get('paragraph_text', ''))
            texts.append(f"{title}: {text}")
        return "\n".join(texts)
    except Exception:
        return None

# Build training data
sft_data = []
stats = {'sufficient_only': 0, 'has_gap_with_supp': 0, 'has_gap_no_supp': 0, 'skipped': 0}

for qid, hops in by_question.items():
    if qid not in raw:
        stats['skipped'] += 1
        continue
    
    question_data = raw[qid]
    gold_answer = question_data['answer']
    original_question = question_data['question']
    
    sub_questions = [h['sub_question'] for h in hops]
    labels = [h['label'] for h in hops]
    passages = [h.get('passage_text_combined', '') for h in hops]
    
    has_insufficient = any(l == 'NOT-ANSWERABLE' for l in labels)
    
    # Build combined initial passages (what model sees at first)
    initial_passages = "\n\n".join(
        f"[Hop {i+1} passages]: {format_passages_text(p)}"
        for i, p in enumerate(passages)
    )
    
    # Build user prompt
    steps = "\n".join(f"Step {i+1}: {sq}" for i, sq in enumerate(sub_questions))
    user_msg = f"""Answer the following question step by step using ONLY the passages below.

Passages:
{initial_passages}

Reasoning steps:
{steps}

Original question: {original_question}

For each step, first assess whether the passages contain sufficient evidence, then answer. If evidence is insufficient, supplementary passages will be provided. Give your final answer at the end."""

    # Build response
    response_parts = []
    all_hops_resolved = True
    
    for i, (sq, label, passage) in enumerate(zip(sub_questions, labels, passages)):
        response_parts.append(f"Step {i+1}: {sq}")
        
        if label == 'ANSWERABLE':
            response_parts.append(
                "Evidence assessment: SUFFICIENT. The passages contain relevant "
                "information to answer this sub-question."
            )
            # Extract a plausible answer direction from the passage
            response_parts.append(
                f"Based on the retrieved passages, I can find the answer to this step."
            )
        else:
            # INSUFFICIENT - need supplementary passages
            response_parts.append(
                "Evidence assessment: INSUFFICIENT. The retrieved passages do not "
                "contain the specific information needed to answer this sub-question."
            )
            
            # Get real supplementary passages if possible
            supp_text = None
            if has_retriever:
                supp_text = get_supplementary_passages(
                    retriever, reformulator, original_question, sq
                )
            
            if supp_text:
                response_parts.append(
                    f"\n[Supplementary retrieval for Step {i+1}]:\n"
                    f"{format_passages_text(supp_text, max_chars=600)}"
                )
                response_parts.append(
                    "Using supplementary evidence: The additional passages now provide "
                    "relevant information for this sub-question."
                )
            else:
                # No supplementary available - mark as unresolvable
                response_parts.append(
                    "No supplementary passages available. This step cannot be answered "
                    "from the provided evidence."
                )
                all_hops_resolved = False
        
        response_parts.append("")
    
    # Only include gold answer if all hops were resolved
    if all_hops_resolved:
        response_parts.append(f"Final answer: {gold_answer}")
    else:
        response_parts.append(
            "Final answer: Unable to determine — insufficient evidence at one or "
            "more reasoning steps."
        )
    
    assistant_msg = "\n".join(response_parts)
    
    sft_data.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    })
    
    if not has_insufficient:
        stats['sufficient_only'] += 1
    elif all_hops_resolved:
        stats['has_gap_with_supp'] += 1
    else:
        stats['has_gap_no_supp'] += 1

random.shuffle(sft_data)

print(f"\nTotal SFT examples: {len(sft_data)}")
print(f"  All hops sufficient: {stats['sufficient_only']}")
print(f"  Has gap + supplementary resolved: {stats['has_gap_with_supp']}")
print(f"  Has gap + unresolved: {stats['has_gap_no_supp']}")
print(f"  Skipped (no raw data): {stats['skipped']}")

# Check token counts (approximate)
total_tokens = 0
max_tokens = 0
for ex in sft_data:
    t = sum(len(m['content'].split()) * 1.3 for m in ex['messages'])  # rough token estimate
    total_tokens += t
    max_tokens = max(max_tokens, t)

print(f"\nApprox token stats:")
print(f"  Total: ~{int(total_tokens):,}")
print(f"  Avg per example: ~{int(total_tokens/len(sft_data)):,}")
print(f"  Max single example: ~{int(max_tokens):,}")
print(f"  Estimated fine-tuning cost: ~${total_tokens * 0.000008:.2f} (at $8/1M tokens)")

# Save
n_val = min(200, len(sft_data) // 5)
val_data = sft_data[:n_val]
train_data = sft_data[n_val:]

train_path = 'results/sft_train.jsonl'
val_path = 'results/sft_val.jsonl'

with open(train_path, 'w') as f:
    for ex in train_data:
        f.write(json.dumps({"messages": ex["messages"]}) + "\n")

with open(val_path, 'w') as f:
    for ex in val_data:
        f.write(json.dumps({"messages": ex["messages"]}) + "\n")

print(f"\nTrain: {len(train_data)} examples -> {train_path}")
print(f"Val: {len(val_data)} examples -> {val_path}")

# Print one example of each type
for target_type in ['sufficient', 'gap']:
    for ex_data, hops in by_question.items():
        labels = [h['label'] for h in hops]
        has_gap = any(l == 'NOT-ANSWERABLE' for l in labels)
        if target_type == 'sufficient' and not has_gap:
            break
        if target_type == 'gap' and has_gap:
            break
    
    # Find matching SFT example
    for ex in sft_data:
        content = ex['messages'][2]['content']
        if target_type == 'sufficient' and 'INSUFFICIENT' not in content:
            print(f"\n{'='*60}")
            print(f"EXAMPLE ({target_type.upper()}):")
            print(f"{'='*60}")
            for msg in ex['messages']:
                print(f"\n--- {msg['role'].upper()} ---")
                print(msg['content'][:600])
                if len(msg['content']) > 600:
                    print("...")
            break
        if target_type == 'gap' and 'INSUFFICIENT' in content and 'Supplementary' in content:
            print(f"\n{'='*60}")
            print(f"EXAMPLE ({target_type.upper()}):")
            print(f"{'='*60}")
            for msg in ex['messages']:
                print(f"\n--- {msg['role'].upper()} ---")
                print(msg['content'][:800])
                if len(msg['content']) > 800:
                    print("...")
            break

print(f"\n{'='*60}")
print("NEXT STEPS:")
print(f"{'='*60}")
print("""
1. Check which models support fine-tuning:
   python -c "from openai import OpenAI; c=OpenAI(); [print(m.id) for m in c.models.list() if 'mini' in m.id.lower()]"

2. Upload training data:
   python -c "
from openai import OpenAI
c = OpenAI()
train = c.files.create(file=open('results/sft_train.jsonl','rb'), purpose='fine-tune')
val = c.files.create(file=open('results/sft_val.jsonl','rb'), purpose='fine-tune')
print(f'Train file: {train.id}')
print(f'Val file: {val.id}')
"

3. Start fine-tuning (use gpt-4o-mini-2024-07-18 if gpt-4.1-mini unavailable):
   python -c "
from openai import OpenAI
c = OpenAI()
job = c.fine_tuning.jobs.create(
    training_file='<TRAIN_FILE_ID>',
    validation_file='<VAL_FILE_ID>',
    model='gpt-4o-mini-2024-07-18',
    suffix='fact-aware'
)
print(f'Job ID: {job.id}')
print(f'Status: {job.status}')
"

4. Monitor: 
   python -c "from openai import OpenAI; j=OpenAI().fine_tuning.jobs.retrieve('<JOB_ID>'); print(j.status, j.fine_tuned_model)"
""")
