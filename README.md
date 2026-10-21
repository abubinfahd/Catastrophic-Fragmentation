# Catastrophic Fragmentation: A Large-Scale Cross-Domain Benchmark of Tokenization Inefficiency in Frontier Large Language Models: A Case Study on Bengali

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Pinned Locks](https://img.shields.io/badge/Reproducibility-Pinned%20SHA--256-brightgreen.svg)]()

This repository provides the official implementation, deterministic data pipeline, statistical hypothesis testing suite, and reproduction artifacts for the paper:

> **"Catastrophic Fragmentation: A Large-Scale Cross-Domain Benchmark of Tokenization Inefficiency in Frontier Large Language Models: A Case Study on Bengali"**  
> *Neural Computing and Applications* (Springer Nature), October 2026.

---

## 📌 Overview

Sub-word tokenizers serve as the primary computational interface between natural language and large language models (LLMs). This benchmark investigates cross-lingual tokenization disparity between Bengali and English across **$N = 300{,}000$ parallel sentence pairs** ($100{,}000$ pairs per domain), evaluating **$1.5$ million tokenizer–pair observations** across five open-weight frontier language models.

### Key Empirical Findings:
- **Severe Disparity:** Tokenization disparity reaches up to **$4.85\times$** (95% CI: $[4.837, 4.859]$) for Mistral 7B in formal literary Bengali relative to semantically equivalent English.
- **Statistical Significance:** Across all 30 pairwise model comparisons on identical sentence pairs, cross-model differences are statistically significant ($p \le 0.000879$ after Holm–Bonferroni correction; Samanantar Qwen3 vs. Mistral: $t(99{,}999) = -46.88, p < 10^{-15}$). All 15 cross-domain comparisons are likewise statistically significant ($p \le 0.000431$).
- **Vocabulary Allocation Driver:** Mistral allocates only 33 tokens (0.10%) and Qwen3 allocates 0 tokens (0.00%) to the Bengali Unicode block (`U+0980`–`U+09FF`), causing character-by-character or 3-byte UTF-8 fallback. In contrast, Google Gemma 4 allocates 9,979 tokens (3.81%) to Bengali, compressing identical text to an aggregate disparity of $1.24\times$.
- **Sample-Scale Stability:** Multi-scale evaluation across nested prefix expansions ($N \in \{10\text{K}, 50\text{K}, 100\text{K}\}$) demonstrates high asymptotic numerical stability, with coefficients of variation ($\text{CV}$) below $0.014$ across all models.

---

## 🤖 Evaluated Model Checkpoints

All models are evaluated using deterministic, fast-tokenizer implementations loaded directly via pinned repository revisions:

| Model Identifier | Model Family | Parameters | Vocab Size ($|V|$) | Tokenizer Backend | Pinned HF Revision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral 7B | 7.2B | 32,768 | BPE (`tokenizers`) | `e0bc8...` |
| `Qwen/Qwen3-1.7B` | Qwen3 | 1.7B | 151,669 | Byte-level BPE | `d72a4...` |
| `unsloth/Llama-4-Scout-17B-16E-Instruct` | Llama 4 Scout | 17B (16E) | 201,135 | Byte-level BPE | `a93b1...` |
| `microsoft/Phi-4-mini-instruct` | Phi-4-mini | 3.8B | 200,029 | Byte-level BPE | `c14f8...` |
| `google/gemma-4-E4B-it` | Gemma 4 | 4.0B | 262,144 | SentencePiece | `f28a0...` |

Exact commit hashes and SHA-256 digests are pinned in [`repositories.lock.json`](repositories.lock.json).

---

## 📂 Benchmarked Parallel Corpora

We evaluate across three linguistically diverse parallel Bengali–English corpora:

1. **OPUS-100 (General / Conversational):** Sourced from [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100), configuration `bn-en`.
2. **Samanantar (Formal Literature):** Sourced from [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar), configuration `bn`.
3. **CCMatrix (Web & Encyclopedic):** Sourced from [`sentence-transformers/parallel-sentences-ccmatrix`](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix), configuration `en-bn`.

Each dataset stream is processed sequentially in deterministic order using fixed random seed `42`.

---

## 📁 Repository Structure

```text
Code/
├── benchmark.py               # Deterministic sentence extraction, encoding & density logic
├── main.py                    # Unified entry point for benchmark execution and validation
├── statistical_analysis.py    # Inferential statistical suite (t-tests, Wilcoxon, Welch, Holm CIs)
├── plots.py                   # Generation of publication-ready vector figures (PDF & PNG)
├── config.json                # Pinned evaluation configurations (models, corpora, sample scales)
├── repositories.lock.json     # Cryptographic lockfile for Hugging Face model revisions
├── requirements.lock          # Cryptographic package lockfile with SHA-256 hashes
├── tests/                     # Unit and verification tests
└── outputs/
    └── notebook_full_with_statistics/
        ├── VERIFICATION.json  # Comprehensive verification log (all 150 points matching)
        ├── n_10000/           # Output artifacts at N = 10,000 pairs
        ├── n_50000/           # Output artifacts at N = 50,000 pairs
        └── n_100000/          # Primary evaluation artifacts at N = 100,000 pairs:
            ├── frontier_2026_final_benchmark.csv  # Point estimates & densities
            ├── confidence_intervals.csv           # Empirical SE and 95% Confidence Intervals
            ├── paired_model_tests.csv             # 30 paired t-tests and Wilcoxon tests
            ├── domain_tests.csv                   # 15 Welch cross-domain tests
            ├── model_rankings.csv                 # Pooled arithmetic disparity rankings
            └── figures/
                ├── fig1_cross_domain_disparity.pdf
                ├── fig1_cross_domain_disparity.png
                ├── fig2_grand_inefficiency_ranking.pdf
                └── fig2_grand_inefficiency_ranking.png
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.12+
- CUDA-enabled GPU (optional; benchmark tokenization runs efficiently on standard CPUs)

### 1. Environment Setup
Clone the repository and install all dependencies strictly pinned with cryptographic hashes:

```bash
git clone https://github.com/abubinfahd/Catastrophic-Fragmentation.git
cd Catastrophic-Fragmentation/Code

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using hash verification
pip install --require-hashes -r requirements.lock
```

### 2. Environment Variables (Optional)
If accessing gated Hugging Face repositories, copy the example environment file and provide your token:

```bash
cp .env.example .env
# Edit .env and set: HF_TOKEN=your_huggingface_token
```

---

## 🔬 Reproducing the Benchmark

### Run Complete End-to-End Evaluation
To execute the streaming evaluation across all three corpora, five models, and three sample scales ($N \in \{10\text{K}, 50\text{K}, 100\text{K}\}$):

```bash
python main.py
```

This will:
1. Stream and tokenize the parallel sentence pairs deterministically under seed `42`.
2. Compute language token densities, per-pair disparity multipliers, and pooled averages.
3. Perform paired two-sided Student's $t$-tests, Wilcoxon signed-rank tests, and Welch tests.
4. Export high-resolution vector PDF and 300 DPI PNG figures to `outputs/`.

---

## 📊 Summary of Main Results ($N = 100{,}000$ per Domain)

| Domain | Tokenizer Checkpoint | English Density | Bengali Density | Std. Error ($SE$) | Disparity Multiplier [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OPUS-100** | Mistral 7B Instruct v0.3 | 1.57 | 6.50 | 0.0067 | **4.62** [4.607, 4.633] |
| | Alibaba Qwen3 1.7B | 1.43 | 5.93 | 0.0069 | **4.61** [4.598, 4.625] |
| | Meta Llama 4 Scout | 1.40 | 2.80 | 0.0032 | **2.18** [2.170, 2.183] |
| | Microsoft Phi-4-mini | 1.38 | 2.35 | 0.0029 | **1.84** [1.836, 1.848] |
| | Google Gemma 4 E4B-it | 1.43 | 1.79 | 0.0024 | **1.37** [1.367, 1.376] |
| **Samanantar** | Mistral 7B Instruct v0.3 | 1.45 | 7.29 | 0.0054 | **4.85** [4.837, 4.859] |
| | Alibaba Qwen3 1.7B | 1.34 | 6.68 | 0.0053 | **4.75** [4.744, 4.765] |
| | Meta Llama 4 Scout | 1.32 | 2.95 | 0.0022 | **2.09** [2.088, 2.097] |
| | Microsoft Phi-4-mini | 1.31 | 2.37 | 0.0019 | **1.69** [1.691, 1.698] |
| | Google Gemma 4 E4B-it | 1.31 | 1.73 | 0.0014 | **1.24** [1.236, 1.242] |
| **CCMatrix** | Mistral 7B Instruct v0.3 | 1.38 | 6.72 | 0.0040 | **4.59** [4.585, 4.601] |
| | Alibaba Qwen3 1.7B | 1.29 | 6.26 | 0.0038 | **4.54** [4.537, 4.552] |
| | Meta Llama 4 Scout | 1.26 | 2.62 | 0.0015 | **1.90** [1.897, 1.903] |
| | Microsoft Phi-4-mini | 1.26 | 2.16 | 0.0012 | **1.57** [1.566, 1.570] |
| | Google Gemma 4 E4B-it | 1.29 | 1.58 | 0.0009 | **1.12** [1.120, 1.124] |
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE). Datasets and model checkpoints remain governed by their respective upstream licenses.
