# Beyond Token Entropy: Disentangling Diversity Axes for OOD Reasoning in RL-Trained LLMs

**Samuel Maduabuchi**
University of Nigeria, Nsukka

---

## Overview

This repository contains the code and data for the paper:

> *Beyond Token Entropy: Disentangling Diversity Axes for Out-of-Distribution Reasoning in RL-Trained LLMs*
> Submitted to NeurIPS 2026

We propose a three-axis decomposition of output diversity in RL-trained LLMs — token-level (ΔT), semantic (ΔM), and structural (ΔS) — and conduct an observational study of their relationship with OOD accuracy across three benchmarks.

---

## Key Finding

| Diversity Axis | Metric | r | p |
|---|---|---|---|
| Token-level | ΔT | +0.777 | 0.014 |
| Semantic | ΔM | +0.205 | 0.597 |
| Structural | ΔS | -0.834 | 0.005 |

Token-level diversity is a strong positive predictor of OOD accuracy. Structural diversity is a strong **negative** predictor.

---

## Repository Structure

beyond-token-entropy/
├── experiment.py
├── results.json
├── README.md
├── LICENSE
└── .gitignore

---

## Models Evaluated

- Qwen/Qwen2.5-Math-1.5B-Instruct
- hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo
- Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B

---

## Benchmarks

- StrategyQA — 30 questions, commonsense reasoning
- MMLU College Mathematics — 30 questions, university-level math
- ARC-Challenge — 30 questions, scientific reasoning

---

## Requirements

torch
transformers
bitsandbytes
sentence-transformers
datasets
scipy
numpy

Install with: pip install torch transformers bitsandbytes sentence-transformers datasets scipy numpy

---

## Reproducing Results

Run: python experiment.py

Results will be saved to results.json. Expected runtime is approximately 60-90 minutes on a single T4 GPU.

---

## Hardware

Single NVIDIA T4 GPU (16GB VRAM), 4-bit quantisation (NF4, float16 compute dtype) via bitsandbytes. Models loaded and unloaded sequentially.

---

## Citation

@article{maduabuchi2026beyond,
  title={Beyond Token Entropy: Disentangling Diversity Axes for Out-of-Distribution Reasoning in RL-Trained LLMs},
  author={Maduabuchi, Samuel},
  journal={arXiv preprint},
  year={2026}
}

---

## License

MIT License. See LICENSE file for details.
