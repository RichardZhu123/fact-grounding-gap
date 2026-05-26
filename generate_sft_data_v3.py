"""
Generate SFT training data v3: Real Self-Ask multi-turn traces.

Each training example is the full multi-turn conversation that
SimpleMultiHopQA generates — exactly matching inference format.

Uses existing intervention outcomes to select winning traces.
Generates traces by running the actual pipeline (costs API calls).

Run on VM: cd ~/ircot && python generate_sft_data_v3.py
"""
import json
import sys
import time
import random
import argparse
import copy

sys.path.insert(0, '.')
from simple_multihop_qa import SimpleMultiHopQA, Trajectory, RetrievalStep
from deberta_inference import DebertaPredictor

random.seed(42)

class TracingMultiHopQA(SimpleMultiHopQA):
    """Subclass that records the exact prompts and responses at each hop."""
    
    def answer_with_trace(self, question, interventions=None, gold_answer=None):
        """Same as answer_with_interventions but records prompt/response pairs."""
        if interventions is None:
            interventions = {}
        
        steps = []
        all_passages = []
        previous_reasoning = []
        current_query = question
        turns = []  # list of {"prompt": str, "response": str}
        
        for hop in range(self.max_hops):
            hop_num = hop + 1
            
            passages = self.retrieve(current_query)
            all_passages.extend(passages)
            
            # Inject interventions
            if hop_num in interventions:
                all_passages.extend(interventions[hop_num])
            
            # Dedupe
            seen_texts = set()
            unique_passages = []
            for p in all_passages:
                if p["text"] not in seen_texts:
                    seen_texts.add(p["text"])
                    unique_passages.append(p)
            all_passages = unique_passages
            
            # Use the parent's reason_and_decide to get exact same prompt + response
            context = self.format_context(all_passages)
            
            # Reconstruct the exact prompt (matching reason_and_decide exactly)
            prev_reasoning_str = ""
            if previous_reasoning:
                prev_reasoning_str = "\n\nPrevious reasoning:\n" + "\n".join(previous_reasoning)
            
            prompt = f"""You are answering a multi-hop question that may require multiple pieces of information.
Question: {question}
Retrieved Information:
{context}
{prev_reasoning_str}
Based on the information above, do ONE of the following:
1. If you have enough information to answer the question, respond with:
ANSWER: [your concise answer]
2. If you need more information, respond with:
SEARCH: [a specific search query to find the missing information]
REASONING: [brief explanation of what you're looking for and why]
Be concise. If you can answer, give just the answer. If you need to search, give a focused query."""
            
            # Call the API (same params as reason_and_decide)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            output = response.choices[0].message.content.strip()
            
            turns.append({"role": "user", "content": prompt})
            turns.append({"role": "assistant", "content": output})
            
            step = RetrievalStep(
                hop_number=hop_num,
                query=current_query,
                retrieved_passages=passages,
                reasoning=output,
            )
            steps.append(step)
            previous_reasoning.append(output)
            
            if output.startswith("ANSWER:"):
                final_answer = output.split("ANSWER:")[1].strip()
                return Trajectory(
                    question=question, steps=steps,
                    final_answer=final_answer, gold_answer=gold_answer
                ), turns
            elif "SEARCH:" in output:
                for line in output.split("\n"):
                    if line.startswith("SEARCH:"):
                        current_query = line[7:].strip()
                        break
            else:
                final_answer = output
                return Trajectory(
                    question=question, steps=steps,
                    final_answer=final_answer, gold_answer=gold_answer
                ), turns
        
        # Max hops — force answer
        context = self.format_context(all_passages)
        prompt = f"""Question: {question}
Information:
{context}
Based on all the information above, provide your best answer. Be concise. Answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        final_answer = response.choices[0].message.content.strip()
        
        turns.append({"role": "user", "content": prompt})
        turns.append({"role": "assistant", "content": final_answer})
        
        return Trajectory(
            question=question, steps=steps,
            final_answer=final_answer, gold_answer=gold_answer
        ), turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--skip-generation', action='store_true')
    args = parser.parse_args()
    
    # Load dev data
    print("Loading data...")
    examples = []
    with open('raw_data/musique/musique_ans_v1.0_dev.jsonl') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get('answerable', True):
                examples.append(ex)
    
    if args.n:
        examples = examples[:args.n]
    print(f"Processing {len(examples)} examples")
    
    # Load intervention outcomes
    print("Loading intervention outcomes...")
    with open('results/section4_full_dev.json') as f:
        aug = json.load(f)
    
    outcomes = {}
    for ex in aug['examples']:
        outcomes[ex['qid']] = {
            'baseline_correct': ex['baseline']['correct'],
            'deberta_correct': ex['deberta']['correct'],
            'flags': ex['deberta']['flags'],
        }
    
    traces_file = f'{args.output_dir}/selfask_traces_v3.json'
    
    if not args.skip_generation:
        print("\nGenerating traces with TracingMultiHopQA...")
        qa = TracingMultiHopQA(model="gpt-4.1-mini")
        predictor = DebertaPredictor('models/deberta_fg_v2_best')
        
        traces = {}
        api_calls = 0
        skipped = 0
        
        for i, ex in enumerate(examples):
            qid = ex['id']
            question = ex['question']
            gold = ex['answer']
            
            outcome = outcomes.get(qid)
            if not outcome:
                skipped += 1
                continue
            
            base_correct = outcome['baseline_correct']
            deberta_correct = outcome['deberta_correct']
            flags = outcome['flags']
            
            # Skip if neither got it right
            if not base_correct and not deberta_correct:
                skipped += 1
                continue
            
            try:
                # Always generate baseline trace
                base_traj, base_turns = qa.answer_with_trace(question, gold_answer=gold)
                api_calls += len(base_traj.steps)
                
                # Check if answer is actually correct
                base_pred = base_traj.final_answer.lower().strip()
                gold_lower = gold.lower().strip()
                base_actually_correct = (gold_lower in base_pred) or (base_pred in gold_lower)
                
                # If intervention helped in the original experiment, generate intervention trace too
                intervention_turns = None
                intervention_correct = False
                
                if deberta_correct and not base_correct:
                    # Build interventions using DeBERTa on the base trace
                    interventions = {}
                    for step in base_traj.steps:
                        passage_text = ' '.join(
                            p.get('text', '') for p in step.retrieved_passages
                        )
                        is_sufficient = predictor.predict(step.query, passage_text)
                        if not is_sufficient:
                            extra = qa.retrieve(step.query)
                            interventions[step.hop_number] = extra
                    
                    if interventions:
                        int_traj, intervention_turns = qa.answer_with_trace(
                            question, interventions=interventions, gold_answer=gold
                        )
                        api_calls += len(int_traj.steps)
                        
                        int_pred = int_traj.final_answer.lower().strip()
                        intervention_correct = (gold_lower in int_pred) or (int_pred in gold_lower)
                
                traces[qid] = {
                    'question': question,
                    'gold': gold,
                    'base_turns': base_turns,
                    'base_correct': base_actually_correct,
                    'intervention_turns': intervention_turns,
                    'intervention_correct': intervention_correct,
                }
                
            except Exception as e:
                print(f"  Error on {qid}: {e}")
                continue
            
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(examples)}: {len(traces)} traces, {api_calls} API calls, {skipped} skipped")
        
        with open(traces_file, 'w') as f:
            json.dump(traces, f)
        print(f"\nSaved {len(traces)} traces to {traces_file}")
        print(f"API calls: {api_calls}, Skipped: {skipped}")
    
    else:
        print(f"Loading cached traces from {traces_file}")
        with open(traces_file) as f:
            traces = json.load(f)
    
    # Build SFT data
    print(f"\nBuilding SFT data from {len(traces)} traces...")
    
    sft_data = []
    stats = {'base_used': 0, 'intervention_used': 0, 'neither_correct': 0}
    
    for qid, trace in traces.items():
        # Pick the best trace: intervention if it worked, else baseline if it worked
        if trace['intervention_turns'] and trace['intervention_correct']:
            turns = trace['intervention_turns']
            stats['intervention_used'] += 1
        elif trace['base_correct']:
            turns = trace['base_turns']
            stats['base_used'] += 1
        else:
            stats['neither_correct'] += 1
            continue
        
        # OpenAI format: messages array with alternating user/assistant
        # No system message needed — the prompt IS the system context
        sft_data.append({
            "messages": turns
        })
    
    random.shuffle(sft_data)
    
    print(f"Total SFT examples: {len(sft_data)}")
    print(f"  From correct baselines: {stats['base_used']}")
    print(f"  From successful interventions: {stats['intervention_used']}")
    print(f"  Skipped (neither correct at generation time): {stats['neither_correct']}")
    
    # Token stats
    total_tokens = 0
    max_tokens = 0
    for ex in sft_data:
        t = sum(len(m['content'].split()) * 1.3 for m in ex['messages'])
        total_tokens += t
        max_tokens = max(max_tokens, t)
    
    print(f"\nToken stats:")
    print(f"  Total: ~{int(total_tokens):,}")
    print(f"  Avg: ~{int(total_tokens/max(len(sft_data),1)):,}")
    print(f"  Max: ~{int(max_tokens):,}")
    print(f"  Est. fine-tuning cost: ~${total_tokens * 0.000008:.2f}")
    
    # Save
    n_val = min(200, len(sft_data) // 5)
    train_data = sft_data[n_val:]
    val_data = sft_data[:n_val]
    
    train_path = f'{args.output_dir}/sft_v3_train.jsonl'
    val_path = f'{args.output_dir}/sft_v3_val.jsonl'
    
    with open(train_path, 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    
    with open(val_path, 'w') as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\nTrain: {len(train_data)} -> {train_path}")
    print(f"Val: {len(val_data)} -> {val_path}")
    
    # Print example
    if sft_data:
        print(f"\n{'='*60}")
        print("EXAMPLE (first 2 turns):")
        print(f"{'='*60}")
        for msg in sft_data[0]['messages'][:4]:
            print(f"\n--- {msg['role'].upper()} ---")
            print(msg['content'][:400])
            if len(msg['content']) > 400:
                print("...")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"""
1. Upload and fine-tune:
   python -c "
from openai import OpenAI
c = OpenAI()
train = c.files.create(file=open('{train_path}','rb'), purpose='fine-tune')
val = c.files.create(file=open('{val_path}','rb'), purpose='fine-tune')
job = c.fine_tuning.jobs.create(
    training_file=train.id, validation_file=val.id,
    model='gpt-4.1-mini-2025-04-14', suffix='fact-aware-v3'
)
print(f'Job: {{job.id}}, Status: {{job.status}}')
"

2. Evaluate with sft_evaluate.py (update FT_MODEL to new model ID)
""")

if __name__ == '__main__':
    main()
