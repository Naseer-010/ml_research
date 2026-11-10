# Methodology: Group-Wise Calibration in Pruned and Quantized Tabular Models

This document details the mathematical formulation, experimental design, and algorithmic implementation of our study on how model compression impacts subgroup calibration in tabular machine learning models.

---

## 1. Research Objective & Hypotheses

Compression techniques (structured/unstructured pruning, quantization) optimize models for latency, energy, and parameter footprint. However, compression is not uniform across data distributions: low-density or minority demographic subgroups risk disproportionate predictive degradation.

**Core Research Questions:**
1. Does shrinking a tabular model (Random Forest, Gradient Boosting, or MLP) systematically degrade probability calibration (ECE, MCE, Brier score, Log Loss) for demographic subgroups?
2. Do calibration disparities widen between privileged and unprivileged groups under aggressive compression (up to 90% capacity reduction)?
3. Can held-out post-hoc recalibration (Platt scaling, Isotonic regression) recover subgroup calibration, and does group-specific recalibration outperform global recalibration?

---

## 2. Model Families & Compression Protocols

We examine three model families commonly deployed on tabular data:

### 2.1 Random Forest (Scikit-Learn)
- **Baseline**: 500 decision trees (`n_estimators=500`), default unlimited depth (`max_depth=None`).
- **Compression**: Sequential ensemble truncation:
  $$n_{\text{keep}} = \max(1, \lfloor n_{\text{total}} \cdot (1 - \lambda) \rfloor)$$
  where $\lambda \in \{0.0, 0.3, 0.5, 0.7, 0.9\}$.
  Trees are retained in initial ensemble order (preserving the unbiased i.i.d. ensemble expectation rather than ranking by standalone validation accuracy).

### 2.2 Gradient Boosted Decision Trees (Scikit-Learn GradientBoostingClassifier)
- **Baseline**: 300 boosting stages (`n_estimators=300`, `max_depth=5`, `learning_rate=0.1`).
- **Compression**: Boosting stage truncation slicing the first $\lfloor n_{\text{total}} \cdot (1 - \lambda) \rfloor$ trees, preserving the additive residual learning trajectory.

### 2.3 Tabular Multi-Layer Perceptron (PyTorch)
- **Baseline**: Input $\to$ Linear(128) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(64) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(32) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(1) $\to$ Sigmoid.
- **Training**: Adam optimizer, initial LR $10^{-3}$, batch size 256, early stopping patience 10 (with internal 15% train-validation split to strictly safeguard the held-out calibration set).
- **Compression Modes**:
  1. `prune_only`: Global $L_1$ unstructured weight pruning applied to hidden `nn.Linear` layers at sparsity $\lambda$, finalized via `torch.nn.utils.prune.remove`.
  2. `quantize_only`: Dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic`) converting Linear weights to 8-bit integers (`qint8`).
  3. `prune+quantize`: Compound pruning followed by INT8 dynamic quantization.
  4. `fp16`: Half-precision float16 inference conversion.

---

## 3. Data Integrity & Partitioning

To prevent data leakage, every dataset is split into three strictly disjoint partitions using stratified sampling:
- **Train Set (60%)**: Model parameter training and internal feature transformer fitting (`StandardScaler`, `OrdinalEncoder`, `SimpleImputer`).
- **Calibration Set (20%)**: Held-out partition exclusively used to fit post-hoc calibration models (Platt scaling and Isotonic regression).
- **Test Set (20%)**: Untouched evaluation partition evaluated across all baseline, compressed, uncalibrated, and recalibrated configurations.

### Benchmark Datasets & Evaluated Sensitive Attributes
1. **Adult (Census Income)** [UCI ID 2]: Binary income (`>50K` vs `<=50K`), sensitive attributes: `sex` (Male, Female), `race` (White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other).
2. **Statlog German Credit** [UCI ID 144]: Credit risk (Good vs Bad), sensitive attributes: `sex` (derived from Attribute 9), `age_group` (`<25` vs `\ge 25`).
3. **Default of Credit Card Clients** [UCI ID 350]: Default next month, sensitive attributes: `SEX`, `EDUCATION`, `MARRIAGE`.
4. **Bank Marketing** [UCI ID 222]: Term deposit subscription, sensitive attributes: `age_group` (`<30` vs `\ge 30`), `marital`.

---

## 4. Calibration Metrics & Proper Scoring Rules

For predictions $\hat{p}_i \in [0, 1]$ and binary labels $y_i \in \{0, 1\}$ over sample size $N$:

### 4.1 Expected Calibration Error (ECE)
Partition predictions into $M=15$ equal-width bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ and $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.

### 4.2 Maximum Calibration Error (MCE)
$$\text{MCE} = \max_{m: |B_m| > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 4.3 Brier Score
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### 4.4 Log Loss (Binary Cross-Entropy)
Strictly proper scoring rule heavily penalizing overconfident inaccuracies:
$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i + \epsilon) + (1 - y_i) \log(1 - \hat{p}_i + \epsilon) \right]$$

### 4.5 Group Calibration Disparity
For sensitive attribute $A$ with domain $\{g_1, \dots, g_K\}$:
$$\Delta \text{ECE}_A = \max_{k} \text{ECE}(g_k) - \min_{k} \text{ECE}(g_k)$$

---

## 5. Recalibration Methods

### 5.1 Global Recalibration
- **Platt Scaling**: Univariate unregularized logistic regression ($C=10^5$) fit on $(p_{\text{cal}}, y_{\text{cal}})$ and applied to $p_{\text{test}}$.
- **Isotonic Regression**: Non-parametric piecewise constant monotonic step function fit on $(p_{\text{cal}}, y_{\text{cal}})$ with boundary clipping.

### 5.2 Group-Specific Recalibration
Fits separate calibrators per subgroup:
$$f_{g}: p_{\text{cal}}^{(g)} \to y_{\text{cal}}^{(g)}$$
If a subgroup has fewer than 10 calibration samples or single-class observations, it automatically falls back to the globally fitted calibrator.

---

## 6. Efficiency Metrics & Pareto Frontiers

For every compressed model snapshot, `src/efficiency.py` measures:
- **Total Parameters / Node Count**: Summed nodes for tree ensembles; total weights for PyTorch models.
- **Sparsity**: $1 - \frac{\text{nonzero\_parameters}}{\text{total\_parameters}}$.
- **Serialized Model Size (bytes)**: Exact byte footprint on disk.
- **Inference Latency (ms)**: Warm-started CPU inference latency averaged across 5 iterations ($\mu \pm \sigma$).

---

## 7. Statistical Testing Framework

- **Seeds**: 10 independent random seeds (`config.RANDOM_SEEDS = [42, 123, 456, 789, 1010, 2024, 3141, 4567, 5678, 7890]`).
- **Paired Student's t-test**: Tests null hypothesis that mean difference in metric scores between baseline and compressed conditions equals zero.
- **Wilcoxon Signed-Rank Test**: Non-parametric test of paired rank differences (with $N=10$, minimum achievable two-sided $p$-value is $2 \cdot 2^{-10} \approx 0.00195 < 0.05$).
- **Family-Wise Error Rate Control**: Bonferroni correction adjusts $\alpha = \frac{0.05}{K}$ across tested comparisons.
