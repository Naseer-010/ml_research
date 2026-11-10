# Methodology: Group-Wise Calibration in Pruned and Quantized Tabular Models

This document details the theoretical motivation, mathematical formulation, experimental design, and algorithmic implementation for our research study: **Does Shrinking a Model Make Its Predictions Less Trustworthy for Some Groups? Group-Wise Calibration in Pruned and Quantized Tabular Models**.

---

## 1. Research Motivation & Core Hypotheses

Machine learning classifiers deployed in high-stakes domains (credit scoring, financial default prediction, census analytics, marketing) are increasingly compressed via pruning and quantization to meet latency, storage, and compute budgets on resource-constrained devices.

Standard model evaluations measure performance through aggregate accuracy, ROC-AUC, or population-level Expected Calibration Error (ECE). However, aggregate metrics mask disparities: model compression disproportionately impacts low-density, minority, or under-represented demographic subgroups.

### Core Research Questions
1. **Calibration Degradation under Compression**: Does shrinking a tabular model (Random Forest, Gradient Boosted Decision Trees, or Multi-Layer Perceptron) systematically degrade probability calibration (ECE, MCE, Brier Score, Log Loss) for demographic subgroups?
2. **Disparity Widening**: Does aggressive capacity reduction (up to 90% parameter/tree pruning or INT8 quantization) widen the calibration gap between privileged and unprivileged groups?
3. **Post-Hoc Recalibration Recovery**: Can held-out recalibration (Platt scaling, Isotonic regression) restore reliability after compression? Does group-specific recalibration outperform global recalibration without causing overfitting on small demographic slices?
4. **Explainability & Attribution Drift**: How does compression alter internal feature importances (via SHAP values) across demographic slices?

---

## 2. Experimental Design Matrix

The study evaluates a complete Cartesian product of datasets, model architectures, compression levels, and random seeds:

| Experimental Dimension | Specification |
| :--- | :--- |
| **Datasets ($D=4$)** | Adult (Census Income), Statlog German Credit, Default of Credit Card Clients, Bank Marketing |
| **Model Families ($M=3$)** | Random Forest (`scikit-learn`), Gradient Boosted Trees (`scikit-learn`), Tabular MLP (`PyTorch`) |
| **Compression Levels ($\lambda$)** | $0.0$ (Baseline FP32), $0.30$ (30% pruned), $0.50$ (50% pruned), $0.70$ (70% pruned), $0.90$ (90% pruned) |
| **NN Compression Modes** | `prune+quantize` (compound default), `prune_only`, `quantize_only`, `fp16` |
| **Random Seeds ($S=10$)** | `[42, 123, 456, 789, 1010, 2024, 3141, 4567, 5678, 7890]` |
| **Data Partitioning** | 60% Train, 20% Calibration (held-out), 20% Test (strictly untouched) |
| **Recalibration Variants** | Uncalibrated (`none`), Global Platt Scaling, Global Isotonic Regression, Group-Specific Platt, Group-Specific Isotonic |
| **Total Model Evaluations** | $4 \times 3 \times 5 \times 10 = 600$ model snapshots evaluated across dozens of demographic groups and calibration regimes |

---

## 3. Benchmark Datasets & Demographic Attributes

All datasets originate from the UCI Machine Learning Repository and OpenML. Sensitive attributes are retained in feature inputs for real-world fidelity while also isolated for group-wise evaluation.

| Dataset | UCI ID | Instances | Features | Target Variable | Sensitive Group Attributes & Slices | Rationale |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Adult (Census Income)** | 2 | 48,842 | 14 | Income $>50\text{K}$ (1) vs $\le 50\text{K}$ (0) | • `sex`: Male, Female<br>• `race`: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other | Large-scale census benchmark; statistical power for fine-grained racial subgroups. |
| **Statlog German Credit** | 144 | 1,000 | 20 | Credit Risk Good (1) vs Bad (0) | • `sex`: Male, Female (derived from Attribute 9)<br>• `age_group`: $<25$ vs $\ge 25$ | Small sample regime ($N=1,000$); assesses compression vulnerabilities when training samples are sparse. |
| **Credit Default** | 350 | 30,000 | 23 | Default Next Month (1 vs 0) | • `SEX`: Male, Female<br>• `EDUCATION`: Grad School, University, High School, Other<br>• `MARRIAGE`: Married, Single, Other | Financial domain; multiple intersecting demographic and socio-economic variables. |
| **Bank Marketing** | 222 | 45,211 | 16 | Term Deposit `yes` (1) vs `no` (0) | • `age_group`: $<30$ vs $\ge 30$<br>• `marital`: Married, Single, Divorced | Class-imbalanced direct marketing (~11.7% positive class); tests calibration under skewed priors. |

---

## 4. Preprocessing & Leakage-Free Data Protocol

To guarantee mathematical reproducibility and prevent data contamination:
1. **Three-Way Stratified Partitioning**:
   - **Training Set (60%)**: Used exclusively for parameter fitting and neural network gradient updates.
   - **Calibration Set (20%)**: Held out completely during training; used exclusively for fitting post-hoc calibration functions.
   - **Test Set (20%)**: Held out completely; evaluated across all baseline, compressed, and recalibrated variants.
2. **Transformer Isolation**:
   - Missing numerical values are imputed via median; missing categorical values via mode.
   - Categorical columns are encoded with `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
   - Continuous numerical columns are standardized with `StandardScaler()`.
   - All transformers are fit **strictly** on the 60% training partition and applied out-of-fold to calibration and test splits.
3. **Neural Network Validation Separation**:
   - Neural network early stopping utilizes an internal 15% split drawn strictly from the 60% training partition. The held-out 20% calibration partition remains completely untouched until model parameters are frozen.

---

## 5. Model Architectures & Compression Algorithms

### 5.1 Random Forest (`models/random_forest_model.py`)
- **Architecture**: Bagged ensemble of 500 decision trees (`n_estimators=500`, `max_depth=None`, `oob_score=True`, `n_jobs=-1`).
- **Compression**: Sequential ensemble truncation:
  $$n_{\text{keep}} = \max\left(1, \text{round}\left(n_{\text{total}} \cdot (1 - \lambda)\right)\right)$$
  Trees are retained in initial ensemble generation order.
  *Design Rationale*: Ranking trees by individual accuracy violates the independent and identically distributed (i.i.d.) bootstrap assumption of bagging, biasing diversity. Slicing sequentially preserves ensemble expectation.

### 5.2 Gradient Boosted Decision Trees (`models/gradient_boosting_model.py`)
- **Architecture**: Sequentially boosted decision trees (`n_estimators=300`, `max_depth=5`, `learning_rate=0.1`).
- **Compression**: Boosting stage truncation:
  $$F_m(x) = F_0(x) + \sum_{i=1}^{n_{\text{keep}}} \gamma_i h_i(x)$$
  where $n_{\text{keep}} = \max(1, \text{round}(n_{\text{total}} \cdot (1 - \lambda)))$.
  *Design Rationale*: Stage truncation preserves the primary gradient descent trajectory without disrupting the residual relationships between successive boosting iterations.

### 5.3 Tabular Multi-Layer Perceptron (`models/neural_network_model.py`)
- **Architecture**:
  $$\text{Input}(d) \xrightarrow{\text{Linear}} 128 \xrightarrow{\text{BN, ReLU, Drop}(0.3)} 64 \xrightarrow{\text{BN, ReLU, Drop}(0.3)} 32 \xrightarrow{\text{BN, ReLU, Drop}(0.3)} 1 \xrightarrow{\text{Sigmoid}} \hat{p}$$
- **Training**: Adam optimizer ($\text{lr}=10^{-3}$), batch size 256, binary cross-entropy loss, early stopping with patience 10.
- **Compression Modes**:
  1. `prune_only`: Global $L_1$ unstructured weight pruning applied to hidden linear projections at sparsity level $\lambda$:
     $$\text{mask}_{ij} = \mathbb{I}\left(|W_{ij}| \ge \tau_\lambda\right)$$
     Pruning hooks are made permanent via `torch.nn.utils.prune.remove`.
  2. `quantize_only`: Dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic`) converting 32-bit floating point weights into signed 8-bit integers (`torch.qint8`) with INT8 matrix multiplication on CPU.
  3. `prune+quantize`: Compound pipeline: $L_1$ weight pruning followed by INT8 dynamic quantization.
  4. `fp16`: 16-bit half-precision floating point weight conversion.

---

## 6. Calibration Metrics & Proper Scoring Rules

For true labels $y_i \in \{0, 1\}$ and model predicted probabilities $\hat{p}_i \in [0, 1]$ over sample size $N$:

### 6.1 Expected Calibration Error (ECE)
Partitions predicted probabilities into $M=15$ equal-width bins $B_m = (\frac{m-1}{M}, \frac{m}{M}]$ with right-inclusive upper bound:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where empirical accuracy and confidence are:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i, \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$
Empty bins ($|B_m| = 0$) contribute 0 to the sum.

### 6.2 Maximum Calibration Error (MCE)
Identifies the worst-case bin deviation:
$$\text{MCE} = \max_{m: |B_m| > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 6.3 Brier Score
Proper scoring rule calculating mean squared error in probability space:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### 6.4 Log Loss (Binary Cross-Entropy)
Strictly proper scoring rule with extreme penalty for overconfident incorrect predictions ($\epsilon = 10^{-15}$):
$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i + \epsilon) + (1 - y_i) \ln(1 - \hat{p}_i + \epsilon) \right]$$

### 6.5 Group-Wise Disparity Metrics
For sensitive attribute $A$ with discrete subgroups $\{g_1, g_2, \dots, g_K\}$:
- **Maximum Calibration Gap**:
  $$\Delta \text{ECE}_A = \max_{k} \text{ECE}(g_k) - \min_{k} \text{ECE}(g_k)$$
- **Maximum Brier Gap**:
  $$\Delta \text{Brier}_A = \max_{k} \text{Brier}(g_k) - \min_{k} \text{Brier}(g_k)$$

---

## 7. Post-Hoc Recalibration Protocols

### 7.1 Global Recalibration
1. **Platt Scaling**:
   Fits a univariate unregularized logistic regression ($C=10^5$, L-BFGS) mapping uncalibrated probabilities to empirical class posteriors:
   $$\hat{p}_{\text{cal}} = \sigma(w \cdot \hat{p} + b)$$
2. **Isotonic Regression**:
   Fits a non-parametric, piecewise constant monotonic step function:
   $$\hat{p}_{\text{cal}} = \arg\min_f \sum_{i \in \text{Cal}} (y_i - f(\hat{p}_i))^2 \quad \text{s.t. } f(p_a) \le f(p_b) \text{ for } p_a \le p_b$$
   Boundary predictions are clipped to $[0, 1]$.

### 7.2 Group-Specific Recalibration
Fits dedicated calibrators independently per demographic slice:
$$f_{g}: \hat{p}_{\text{cal}}^{(g)} \to y_{\text{cal}}^{(g)}$$
- **Degenerate Slice Fallback**: If subgroup calibration size $|B_{\text{cal}}^{(g)}| < 10$ or if the calibration slice contains only a single class ($y \in \{0\}$ or $y \in \{1\}$), the algorithm automatically reverts to the global calibrator, preventing runtime failures and severe variance.

---

## 8. Hardware Efficiency Metrics

For every compressed model, [`src/efficiency.py`](file:///home/nquzet/Codes/Projects/ml_project26/src/efficiency.py) records:
1. **Total Parameters / Node Count**: Total tree nodes across all estimators (Random Forest / Gradient Boosting) or total trainable and quantized weights (MLP).
2. **Sparsity**: Fraction of zeroed parameters:
   $$\text{Sparsity} = 1.0 - \frac{\text{nonzero\_parameters}}{\text{total\_parameters}}$$
3. **Model Footprint (Bytes)**: Exact memory footprint upon serialization via `pickle`.
4. **Inference Latency (ms)**: CPU execution duration over the test set, measured over 5 iterations after a warm-up phase ($\text{mean} \pm \text{std}$).

---

## 9. Statistical Significance Testing

To verify whether observed calibration degradation or gap widening is statistically significant rather than stochastic variation:
1. **Paired Student's t-test**:
   Tests whether paired difference in metric values across matching seeds between baseline ($\lambda=0.0$) and compressed condition ($\lambda > 0.0$) has a mean significantly different from zero.
2. **Wilcoxon Signed-Rank Test**:
   Non-parametric paired test evaluating signed ranks of differences across seeds.
   *Sample Size Consideration*: Expanding to $N=10$ random seeds ensures the minimum achievable two-sided $p$-value is $2 \times 2^{-10} \approx 0.00195$, enabling rejection of $H_0$ at $\alpha = 0.05$.
3. **Multiple Comparison Correction**:
   Applies Bonferroni correction:
   $$\alpha_{\text{adjusted}} = \frac{\alpha_{\text{nominal}}}{K}$$
   controlling the Family-Wise Error Rate (FWER) across tested hypotheses.

---

## 10. Explainability & Attribution Shift

To analyze whether compression alters which features the model relies upon:
1. **SHAP (SHapley Additive exPlanations)**:
   - Evaluated at baseline ($\lambda=0.0$) and maximum compression ($\lambda=0.90$).
   - `shap.TreeExplainer` applied to Random Forest and Gradient Boosting.
   - Sampled background evaluation with fallback to `sklearn.inspection.permutation_importance` for PyTorch MLP or when SHAP is not installed.
2. **Reliability Diagrams**:
   Calibration curves plotting empirical accuracy vs. mean predicted confidence across bins, faceted by demographic groups.
