# Group-Wise Calibration in Pruned and Quantized Tabular Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Passing](https://img.shields.io/badge/tests-20%2F20%20passed-brightgreen.svg)]()

> **Research Question**: *Does shrinking a machine learning model make its predictions less trustworthy for some demographic groups?*

This repository provides a modular, reproducible research implementation to investigate how **model compression** (structured/unstructured pruning, dynamic quantization, and half-precision conversion) impacts **group-wise probability calibration** across tabular classification models.

---

## Table of Contents

- [Overview](#overview)
- [Key Research Questions](#key-research-questions)
- [Understanding the Experiment: What is Happening Internally?](#understanding-the-experiment-what-is-happening-internally)
  - [1. What are the 120 Runs? (Runs vs. Epochs)](#1-what-are-the-120-runs-runs-vs-epochs)
  - [2. Why Iterate Over 10 Random Seeds?](#2-why-iterate-over-10-random-seeds-statistical-rigor)
  - [3. Lifecycle of a Single Experimental Run](#3-lifecycle-of-a-single-experimental-run-inside-run_single_experiment)
  - [4. Hardware Acceleration: Utilizing Your GPU](#4-hardware-acceleration-utilizing-your-gpu)
- [Repository Structure](#repository-structure)
- [Software Architecture & SOLID Principles](#software-architecture--solid-principles)
- [Installation & Setup](#installation--setup)
- [Quick Start & Running Experiments](#quick-start--running-experiments)
  - [1. Run Unit Tests](#1-run-unit-tests)
  - [2. Fast Smoke Test](#2-fast-smoke-test)
  - [3. Full Research Experiment](#3-full-research-experiment)
  - [4. Generating Tables, Plots & Summary Report](#4-generating-tables-plots--summary-report)
- [Experimental Methodology](#experimental-methodology)
  - [Benchmark Datasets](#benchmark-datasets)
  - [Model Families](#model-families)
  - [Compression Protocols](#compression-protocols)
  - [Calibration Metrics & Proper Scoring Rules](#calibration-metrics--proper-scoring-rules)
  - [Post-Hoc Recalibration Framework](#post-hoc-recalibration-framework)
  - [Hardware Efficiency Benchmarking](#hardware-efficiency-benchmarking)
  - [Statistical Significance Testing](#statistical-significance-testing)
- [Outputs & Visualizations](#outputs--visualizations)
- [Reproducibility Guarantee](#reproducibility-guarantee)
- [License](#license)

---

## Overview

When machine learning models are deployed to edge devices, smartphones, or resource-constrained servers, they are typically compressed via pruning and quantization to meet latency, RAM, and energy constraints. Performance is customarily validated using population-wide averages (Accuracy, ROC-AUC, or aggregate Expected Calibration Error).

However, **population-level averages can conceal severe degradation on minority or under-represented demographic groups**. If a loan default or census income model becomes severely miscalibrated for a specific demographic slice after compression, its decision thresholds and risk estimates become untrustworthy.

This project delivers an end-to-end experimental framework to systematically quantify, visualize, explain, and mitigate subgroup calibration drift in compressed tabular models.

---

## Key Research Questions

1. **Subgroup Calibration Degradation**: Does compression disproportionately degrade probability calibration (ECE, MCE, Brier score, Log Loss) for specific demographic groups?
2. **Disparity Widening**: Do calibration gaps between privileged and unprivileged groups widen under aggressive compression (up to 90% capacity reduction)?
3. **Recalibration Efficacy**: Can post-hoc calibration (Platt scaling, Isotonic regression) recover reliability? Does **group-specific** recalibration outperform global recalibration?
4. **Attribution Drift**: How does compression alter internal feature importances (via SHAP values) across demographic slices?

---

## Understanding the Experiment: What is Happening Internally?

If you are running `python run_experiment.py` and seeing counters like `[RUN 82/120]`, this section breaks down the entire internal architecture, execution hierarchy, and scientific reasoning behind the protocol.

### 1. What are the 120 Runs? (Runs vs. Epochs)

It is crucial to distinguish between an **Experimental Run** and a **Training Epoch**:

* **What is an Epoch?**
  An **epoch** is a single pass through the training dataset during gradient descent optimization for an individual model. For example, our PyTorch Neural Network trains for up to **100 epochs** (with early stopping) on a single seed.
* **What is an Experimental Run?**
  A **run** is a complete, standalone empirical study of **one dataset**, with **one model architecture**, under **one specific random seed**.
* **The 120-Run Matrix Calculation**:
  $$\text{Total Runs} = 4\text{ Benchmark Datasets} \times 3\text{ Model Architectures} \times 10\text{ Random Seeds} = 120\text{ Independent Runs}$$

| Dimension | Count | Details |
| :--- | :---: | :--- |
| **Datasets** | 4 | `adult`, `german_credit`, `credit_default`, `bank_marketing` |
| **Model Families** | 3 | `RandomForest` (Ensemble Trees), `GradientBoosting` (Sequential Trees), `NeuralNetwork` (Deep Tabular MLP) |
| **Random Seeds** | 10 | `[42, 123, 456, 789, 1010, 2024, 3141, 4567, 5678, 7890]` |
| **Total Independent Runs** | **120** | **$4 \times 3 \times 10 = 120$ complete training & evaluation pipelines** |

---

### 2. Why Iterate Over 10 Random Seeds? (Statistical Rigor)

In top-tier machine learning research (NeurIPS, ICML, FAccT), evaluating on a single seed or arbitrary train/test split is considered scientifically insufficient:
1. **Eliminating Lucky / Unlucky Splits**: A single partition could place an unrepresentative distribution of minority groups in the test set. 10 stratified random splits average out stochastic variance.
2. **Quantifying Variance & Confidence Intervals**: Replicating across 10 independent initializations yields authentic empirical standard deviations ($\mu \pm \sigma$) for accuracy, ECE, Brier score, and group calibration gaps.
3. **Hypothesis Testing**: Enables paired statistical tests (**Paired Student's $t$-test** and **Wilcoxon Signed-Rank Test** with **Bonferroni correction** in `src/statistical_tests.py`) to formally verify whether compression-induced miscalibration is statistically significant ($p < 0.05$) rather than random chance.

---

### 3. Lifecycle of a Single Experimental Run (Inside `run_single_experiment`)

Each of the 120 runs performs an extensive multi-tier evaluation pipeline:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. DATA SPLITTING & STRATIFICATION (Zero Data Leakage)                │
│    • 60% Train Set: Model parameter optimization                      │
│    • 20% Calibration Set: Held-out for post-hoc recalibrators          │
│    • 20% Test Set: Strictly untouched final benchmark                  │
│    • Sensitive demographic attributes isolated before encoding         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. TRAIN UNCOMPRESSED BASELINE MODEL (Tier 0.0)                        │
│    • Fits full-capacity model (500 trees in RF/GB, or 100 epochs in NN)│
│    • GPU-accelerated on CUDA for PyTorch Neural Networks               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. PROGRESSIVE CAPACITY COMPRESSION (5 Tiers)                          │
│    • Tier 0.0: Baseline (100% capacity preserved)                     │
│    • Tier 0.30: 30% parameters/trees pruned                           │
│    • Tier 0.50: 50% parameters/trees pruned                           │
│    • Tier 0.70: 70% parameters/trees pruned                           │
│    • Tier 0.90: 90% parameters/trees pruned (extreme compression)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. POST-HOC PROBABILITY RECALIBRATION                                  │
│    Evaluates 5 recalibration regimes on each compression tier:         │
│    1. Uncalibrated (Raw model probabilities)                           │
│    2. Global Platt Scaling (Logistic regression calibrator)            │
│    3. Global Isotonic Regression (Piecewise non-parametric)            │
│    4. Group-Specific Platt Scaling (Slices fitted per demographic group│
│    5. Group-Specific Isotonic Regression                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. MULTI-LEVEL STATISTICAL EVALUATION & PROFILING                      │
│    • Overall Calibration: ECE, MCE, Brier Score, Log Loss, Accuracy    │
│    • Group-Wise Calibration: ECE & MCE computed per demographic slice  │
│    • Calibration Disparity: Max calibration gap between subgroups      │
│    • Hardware Efficiency: Parameters, sparsity, disk bytes & latency ms│
│    • Explainability: SHAP feature attributions on baseline vs 90% prune│
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 6. PROGRESSIVE CHECKPOINTING                                           │
│    Immediately persists all results to disk (results/results_{dataset})│
└────────────────────────────────────────────────────────────────────────┘
```

**Total Snapshot Evaluations per Run**:
- 5 compression levels $\times$ 5 recalibration methods $\times$ (1 overall + $G$ demographic groups) $\approx$ **50 to 125 evaluated condition slices per run**.
- Across all 120 runs, this produces over **10,000+ structured evaluation points** saved into `results/results_all.csv`!

---

### 4. Hardware Acceleration: Utilizing Your GPU

* **PyTorch Tabular MLP (`NeuralNetwork`)**: Automatically detects and leverages your **NVIDIA CUDA GPU** (e.g. GeForce RTX 4080) for batch training, gradient updates, and probability estimation. When dynamic INT8 quantization is evaluated, quantized linear layers automatically execute on CPU as required by PyTorch's quantization engine.
* **Scikit-Learn Tree Ensembles (`RandomForest`, `GradientBoosting`)**: Scikit-learn CPU algorithms run with full multi-core multiprocessing (`n_jobs=-1`) across all available CPU threads.

> [!TIP]
> **Enabling CUDA Acceleration in your Virtual Environment**:
> If your PyTorch was installed from standard PyPI, it may default to the CPU build. To enable full CUDA 12.x hardware acceleration on your NVIDIA GPU:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```
> Verify with: `python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"`

## Repository Structure

```text
ml_project26/
├── config.py                     # Central configuration (hyperparameters, seeds, dataset metadata)
├── requirements.txt              # Pinned project dependencies
├── METHODOLOGY.md                # Comprehensive mathematical formulation & research protocol
├── README.md                     # Project documentation, architecture & usage instructions
│
├── models/                       # Model family implementations (OOP / SOLID wrappers)
│   ├── __init__.py
│   ├── base.py                   # BaseModelWrapper abstract base class
│   ├── random_forest_model.py    # Random Forest wrapper (scikit-learn)
│   ├── gradient_boosting_model.py# Gradient Boosted Trees wrapper (scikit-learn)
│   └── neural_network_model.py   # Multi-Layer Perceptron wrapper (PyTorch)
│
├── src/                          # Functional core modules
│   ├── __init__.py
│   ├── data_loader.py            # UCI ML Repo & OpenML dataset fetcher and cache manager
│   ├── preprocessing.py          # 3-way stratified partitioner & leak-free feature transformers
│   ├── compression.py            # RF tree pruning, GB stage truncation, NN L1 pruning & INT8/FP16
│   ├── calibration_metrics.py    # ECE, MCE, Brier score, Log Loss & calibration gap metrics
│   ├── recalibration.py          # Post-hoc Platt scaling & Isotonic regression (Global & Group-specific)
│   ├── efficiency.py             # Model parameter counting, disk footprint & CPU inference latency
│   ├── statistical_tests.py      # Paired t-test, Wilcoxon signed-rank test & Bonferroni correction
│   └── explainability.py         # SHAP attribution extraction & reliability diagram generators
│
├── tests/                        # Automated unit tests
│   ├── __init__.py
│   ├── test_calibration.py       # Unit tests for ECE, MCE, Brier, gaps, and proper scoring rules
│   └── test_compression.py       # Unit tests for RF pruning, GB truncation, and immutability
│
├── run_experiment.py             # Main experiment runner (orchestrates datasets × models × seeds × compression)
└── results_analysis.py           # Analysis pipeline generating all 4 CSV tables, 9 plots & executive report
```

---

## Software Architecture & SOLID Principles

The codebase is engineered strictly following modern Python best practices and **SOLID object-oriented principles**:

- **Single Responsibility Principle (SRP)**: Each module handles exactly one operational domain. `src/data_loader.py` fetches raw data; `src/preprocessing.py` manages encodings without leakage; `src/compression.py` handles model compression; `src/calibration_metrics.py` calculates statistical error metrics; `src/efficiency.py` benchmarks latency and size.
- **Open/Closed Principle (OCP)**: Adding a new model family (e.g., XGBoost, LightGBM, CatBoost) requires subclassing `BaseModelWrapper` in `models/base.py` without modifying existing training, calibration, or evaluation pipelines.
- **Liskov Substitution Principle (LSP)**: All model wrappers (`RandomForestModelWrapper`, `GradientBoostingModelWrapper`, `NeuralNetworkModelWrapper`) implement identical signatures (`train`, `predict_proba`, `compress`, `get_model`, `model_type`) and can be substituted transparently in the experiment loop.
- **Interface Segregation Principle (ISP)**: Abstract contracts in `BaseModelWrapper` provide concise, purposeful hooks required for downstream evaluation without bloat.
- **Dependency Inversion Principle (DIP)**: High-level scripts (`run_experiment.py`, `results_analysis.py`) depend on abstract abstractions and central configurations (`config.py`) rather than hardcoded parameters.

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Linux / macOS / Windows WSL2

### Environment Setup

```bash
# Clone the repository
cd /path/to/ml_project26

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start & Running Experiments

### 1. Run Unit Tests

Execute the automated test suite to verify metric correctness, zero-leakage guarantees, and compression immutability:

```bash
python3 -m unittest discover -s tests -v
```

Expected output:
```text
Ran 20 tests in 1.082s

OK
```

### 2. Fast Smoke Test

Run a single dataset with a single seed without computing SHAP values (completes in ~30 seconds):

```bash
python3 run_experiment.py --dataset adult --seed 42 --no-shap
```

### 3. Full Research Experiment

Execute the full experimental grid across all 4 datasets, 3 model families, 5 compression levels, and 10 random seeds:

```bash
python3 run_experiment.py
```

#### Customized Runs via CLI Arguments:

```bash
# Run specific datasets
python3 run_experiment.py --dataset adult --dataset german_credit

# Run specific models
python3 run_experiment.py --model RandomForest --model NeuralNetwork

# Test alternative Neural Network compression modes (prune_only, quantize_only, fp16)
python3 run_experiment.py --model NeuralNetwork --nn-mode quantize_only

# Fast run skipping SHAP explanations
python3 run_experiment.py --no-shap
```

### 4. Generating Tables, Plots & Summary Report

After `run_experiment.py` finishes, run `results_analysis.py` to compile the CSV results into structured summary tables, high-resolution figures, and an executive text report:

```bash
python3 results_analysis.py
```

To analyze an explicit results file:
```bash
python3 results_analysis.py --results-file results/results_all.csv
```

---

## Experimental Methodology

### Benchmark Datasets

The study benchmarks 4 diverse tabular datasets spanning different sample sizes, class balances, and demographic variables:

1. **Adult (Census Income)** [UCI ID 2]:
   - 48,842 instances, predicting binary income (`>50K` vs `<=50K`).
   - Evaluated groups: `sex` (Male, Female), `race` (White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other).
2. **Statlog German Credit** [UCI ID 144]:
   - 1,000 instances, predicting Good vs Bad credit risk.
   - Evaluated groups: `sex` (Male, Female; derived from Attribute 9), `age_group` (`<25` vs `\ge 25`).
3. **Default of Credit Card Clients** [UCI ID 350]:
   - 30,000 instances, predicting credit card default next month.
   - Evaluated groups: `SEX`, `EDUCATION`, `MARRIAGE`.
4. **Bank Marketing** [UCI ID 222]:
   - 45,211 instances, predicting term deposit subscription (`yes` vs `no`).
   - Evaluated groups: `age_group` (`<30` vs `\ge 30`), `marital` (Married, Single, Divorced).

### Model Families

1. **Random Forest (scikit-learn)**:
   - 500 estimators, bootstrap sampling (`oob_score=True`), parallel execution.
2. **Gradient Boosted Decision Trees (scikit-learn)**:
   - 300 boosting stages, max depth 5, learning rate 0.1.
3. **Tabular Multi-Layer Perceptron (PyTorch)**:
   - PyTorch architecture: `Input(d) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(64) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(32) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(1) -> Sigmoid`.
   - Trained via Adam ($lr=10^{-3}$) with early stopping (patience 10) on an internal 15% train-validation split.

### Compression Protocols

Compression levels evaluate model capacity reduction: $\lambda \in \{0.0, 0.30, 0.50, 0.70, 0.90\}$.

- **Random Forest**: Sequential tree ensemble truncation:
  $$n_{\text{keep}} = \max\left(1, \lfloor n_{\text{total}} \cdot (1 - \lambda) \rfloor\right)$$
  Retaining trees in initial bootstrap order preserves the unbiased ensemble expectation.
- **Gradient Boosting**: Boosting stage truncation slicing the first $n_{\text{keep}}$ stages, preserving the additive gradient descent trajectory.
- **Neural Network**:
  - `prune_only`: Global $L_1$ unstructured weight pruning of hidden `nn.Linear` layers, made permanent via `torch.nn.utils.prune.remove`.
  - `quantize_only`: Dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic`) converting Linear weights to 8-bit integers (`torch.qint8`) with INT8 matrix multiplication on CPU.
  - `prune+quantize` (default): Sequential $L_1$ pruning followed by dynamic INT8 quantization.
  - `fp16`: 16-bit half-precision floating point conversion.

### Calibration Metrics & Proper Scoring Rules

All probabilities $\hat{p}_i \in [0, 1]$ are evaluated against ground truth $y_i \in \{0, 1\}$:

- **Expected Calibration Error (ECE)**: Equal-width binning ($M=15$ bins) on $[0, 1]$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Maximum Calibration Error (MCE)**: Worst-case populated bin error:
  $$\text{MCE} = \max_{m: |B_m| > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score**: Mean squared probability error:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
- **Log Loss**: Strictly proper scoring rule with extreme penalty for overconfident mispredictions ($\epsilon = 10^{-15}$):
  $$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i + \epsilon) + (1 - y_i) \ln(1 - \hat{p}_i + \epsilon) \right]$$
- **Calibration Disparity Gaps**: For sensitive attribute $A$ with subgroups $\{g_1, \dots, g_K\}$:
  $$\Delta \text{ECE}_A = \max_{k} \text{ECE}(g_k) - \min_{k} \text{ECE}(g_k)$$

### Post-Hoc Recalibration Framework

All recalibrators are fit **exclusively** on the held-out 20% calibration split and applied to the untouched 20% test split:
1. **Platt Scaling**: Univariate unregularized logistic regression ($C=10^5$, L-BFGS).
2. **Isotonic Regression**: Non-parametric piecewise constant monotonic step function with boundary clipping.
3. **Group-Specific Recalibration**: Fits dedicated calibrators independently per demographic subgroup ($f_g: \hat{p}_{\text{cal}}^{(g)} \to y_{\text{cal}}^{(g)}$), automatically falling back to the global calibrator if a subgroup has $<10$ calibration samples or single-class observations.

### Hardware Efficiency Benchmarking

[`src/efficiency.py`](file:///home/nquzet/Codes/Projects/ml_project26/src/efficiency.py) measures:
- **Total Parameters / Node Count**: Total tree nodes across estimators or total weight parameters.
- **Sparsity (%)**: Percentage of zeroed parameters.
- **Model Footprint (Bytes)**: In-memory byte footprint measured via serialization.
- **Inference Latency (ms)**: CPU execution time over the test set, measured over 5 iterations after warm-up ($\text{mean} \pm \text{std}$).

### Statistical Significance Testing

- **10 Random Seeds**: Guarantees paired tests have sufficient sample size ($N=10$) for non-parametric tests.
- **Paired Student's t-test**: Tests whether mean difference in calibration error between baseline and compressed condition is significantly different from zero.
- **Wilcoxon Signed-Rank Test**: Non-parametric test of paired rank differences (minimum achievable two-sided $p$-value: $2 \times 2^{-10} \approx 0.00195 < 0.05$).
- **Bonferroni Correction**: Controls Family-Wise Error Rate (FWER) across all tested comparisons: $\alpha_{\text{adjusted}} = 0.05 / K$.

---

## Outputs & Visualizations

Running `python3 results_analysis.py` automatically populates `results/` with structured tables and high-resolution plots:

### Tables (`results/*.csv`)
- **Table 1 (`table_overall_performance.csv`)**: Mean $\pm$ std for Accuracy, ECE, Brier Score, and Log Loss per model and compression level.
- **Table 2 (`table_groupwise_ece.csv`)**: Detailed breakdown of calibration error across demographic subgroups.
- **Table 3 (`table_recalibration_effectiveness.csv`)**: Side-by-side comparison of uncalibrated vs. Platt vs. Isotonic recalibration.
- **Table 4 (`table_statistical_significance.csv`)**: Paired t-test and Wilcoxon test statistics, raw $p$-values, adjusted $\alpha$, and significance flags.

### Figures (`results/plots/*.png`)
- **Plot 1 (`plot1_calibration_degradation.png`)**: Calibration degradation curves (ECE vs. Compression Level), faceted by dataset.
- **Plot 2 (`plot2_groupwise_gap_*.png`)**: Bar charts of subgroup calibration gaps ($\Delta \text{ECE}$) across compression levels.
- **Plot 3 (`plot3_reliability_diagrams_*.png`)**: Calibration reliability curves comparing empirical accuracy vs. predicted probability per demographic group.
- **Plot 4 (`plot4_recalibration_effectiveness.png`)**: Paired bar charts showing ECE before and after post-hoc recalibration.
- **Plot 5 (`plot5_heatmap_*.png`)**: Heatmap matrix of Model $\times$ Compression Level $\times$ Subgroup ECE.
- **Plot 6 (`plot6_shap_*.png`)**: Feature importance attribution drift between uncompressed baseline and 90% compressed model.
- **Plot 7 (`plot7_significance_matrix_*.png`)**: Statistical significance heatmap with asterisks highlighting significant degradation ($p < \alpha$).
- **Plot 8 (`plot8_accuracy_vs_ece.png`)**: Scatter plot displaying the Accuracy vs. Calibration trade-off trajectory under compression.
- **Plot 9 (`plot9_efficiency_vs_calibration.png`)**: Pareto frontier plotting Model Size (KB) and Inference Latency (ms) vs. ECE.

---

## Reproducibility Guarantee

1. **Strict Random Seed Control**: All random number generators (`random`, `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) are deterministically seeded across all 10 experimental seeds (`config.RANDOM_SEEDS`).
2. **Zero Data Leakage**: Encoders and scalers are fitted strictly on the 60% training partition; calibration sets are kept strictly disjoint from training and test partitions.
3. **Immutable Compression**: Compression methods return new model instances via deep copying, ensuring baseline models are never mutated in-place.
4. **Deterministic Unit Tests**: 20/20 unit tests verify mathematical bounds, edge cases, and algorithmic correctness before running experiments.

---

## License

This project is licensed under the [MIT License](LICENSE).
