"""
Beyond Token Entropy: Disentangling Diversity Axes
for OOD Reasoning in RL-Trained LLMs

Experiment code for computing accuracy and diversity 
metrics across three models and three benchmarks.
"""

import torch
import re
import numpy as np
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ---- Configuration ----
N_QUESTIONS = 30
N_SAMPLES = 4
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 150

MODEL_IDS = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
    "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
]

# ---- Quantization config ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ---- Prompt builder ----
def build_prompt(question, choices=None):
    if choices:
        choice_str = "\n".join([f"{k}: {v}" for k, v in choices.items()])
        return (
            f"Answer the following question with only the letter "
            f"inside \\boxed{{}}.\n"
            f"Question: {question}\n{choice_str}\nAnswer:"
        )
    return (
        f"Answer the following question with YES or NO "
        f"inside \\boxed{{}}.\n"
        f"Question: {question}\nAnswer:"
    )

# ---- Answer extraction ----
def extract_answer(text):
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip().split()[0].rstrip(".,;").upper()
    match = re.match(r"^\s*([A-D])\s*[:\.\)]\s*", text)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([A-D])\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(YES|NO)\b", text.upper())
    if match:
        return match.group(1)
    return None

# ---- Generation ----
def generate(model, tokenizer, prompt, max_new_tokens=150,
             do_sample=False, temperature=0.7, top_p=0.9, n=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    results = []
    for _ in range(n):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()
        results.append(decoded)
    return results

# ---- Diversity metrics ----
def delta_T(samples):
    ratios = []
    for s in samples:
        words = s.lower().split()
        if len(words) >= 3:
            ratios.append(len(set(words)) / len(words))
    return float(np.mean(ratios)) if ratios else 0.0

def delta_M(samples, sbert):
    if len(samples) < 2:
        return 0.0
    embs = sbert.encode(samples, convert_to_tensor=True)
    embs = embs / embs.norm(dim=1, keepdim=True)
    sim_matrix = (embs @ embs.T).cpu().numpy()
    n = len(samples)
    dists = [1 - sim_matrix[i][j] 
             for i in range(n) for j in range(i+1, n)]
    return float(np.mean(dists))

def delta_S(samples):
    patterns = set()
    for s in samples:
        found = re.findall(
            r"[\d\w]+\s*[+\-*/=<>]+\s*[\d\w]+", s
        )
        for f in found:
            patterns.add(re.sub(r"\s+", "", f).lower())
    return len(patterns)

# ---- Dataset loaders ----
def load_strategyqa(n=30):
    ds = load_dataset("ChilleD/StrategyQA", split="train")
    return [{"question": row["question"],
             "answer": "YES" if row["answer"] else "NO",
             "choices": None}
            for row in ds.select(range(n))]

def load_mmlu_math(n=30):
    ds = load_dataset(
        "cais/mmlu", "college_mathematics", split="test"
    )
    keys = ["A", "B", "C", "D"]
    return [{"question": row["question"],
             "answer": keys[row["answer"]],
             "choices": dict(zip(keys, row["choices"]))}
            for row in ds.select(range(min(n, len(ds))))]

def load_arc(n=30):
    ds = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split="test"
    )
    samples = []
    for row in ds.select(range(n)):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices = {}
        for label, text in zip(labels, texts):
            key = (label if label in "ABCD"
                   else str(chr(65 + int(label) - 1)))
            choices[key] = text
        answer = row["answerKey"]
        if answer not in "ABCD":
            answer = str(chr(65 + int(answer) - 1))
        samples.append({
            "question": row["question"],
            "answer": answer,
            "choices": choices
        })
    return samples

# ---- Main experiment ----
def main():
    print("Loading sentence-BERT...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading benchmarks...")
    benchmarks = {
        "StrategyQA":    load_strategyqa(N_QUESTIONS),
        "MMLU-Math":     load_mmlu_math(N_QUESTIONS),
        "ARC-Challenge": load_arc(N_QUESTIONS),
    }

    all_results = {}

    for model_id in MODEL_IDS:
        print(f"\nModel: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()

        model_results = {}

        for bench_name, questions in benchmarks.items():
            print(f"  Benchmark: {bench_name}")
            correct = 0
            dT_scores, dM_scores, dS_scores = [], [], []

            for item in questions:
                prompt = build_prompt(
                    item["question"], item["choices"]
                )
                greedy = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )[0]
                pred = extract_answer(greedy)
                if pred == item["answer"]:
                    correct += 1

                sampled = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    n=N_SAMPLES
                )
                dT_scores.append(delta_T(sampled))
                dM_scores.append(delta_M(sampled, sbert))
                dS_scores.append(delta_S(sampled))

            model_results[bench_name] = {
                "accuracy": correct / len(questions),
                "delta_T":  float(np.mean(dT_scores)),
                "delta_M":  float(np.mean(dM_scores)),
                "delta_S":  float(np.mean(dS_scores)),
            }

        all_results[model_id] = model_results

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results.json")

if __name__ == "__main__":
    main()
