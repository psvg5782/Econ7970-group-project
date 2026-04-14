# Bank Telemarketing Prediction — Final Report
## Team 10 | ECON 7970 Applied Predictive Modeling

**Member 4: Joey | Evaluation, ROC Analysis, Interpretation & Business Recommendation**

---

## 1. Project Overview

This project builds machine learning models to predict whether a bank customer will subscribe to a term deposit following a telemarketing call, using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) (`bank-full.csv`, 45,211 records, 17 features).

The dataset is **heavily imbalanced**: only ~11% of customers subscribed ("yes"), making standard accuracy misleading. We therefore evaluate all models using **ROC-AUC, Recall, Precision, and F1-score**, and apply **SMOTE oversampling** to improve recall.

---

## 2. Team Division Summary

| Phase | Member | Key Outputs |
|-------|--------|-------------|
| 1 – Data Preparation & EDA | Member 1 (Gonghan) | `bankclean.csv`, train/test splits, 4 scenario datasets, EDA plots |
| 2 – Baseline Models + SMOTE | Member 2 (Habiba) | Logistic Reg, Lasso, Ridge, Decision Tree on all 4 scenarios |
| 3 – Advanced Models + SMOTE | Member 3 (Zaheer) | KNN, Random Forest, SVM, XGBoost on all 4 scenarios |
| 4 – Evaluation & Report | Member 4 (Joey) | ROC/PR curves, confusion matrices, threshold optimization, business recommendation, final report |

---

## 3. The 4 Evaluation Scenarios

To understand what actually drives predictions, all models were tested under 4 different feature sets:

| Scenario | Features Used | Purpose |
|----------|--------------|---------|
| **S1** | All features including `duration` | Upper-bound benchmark (leaky — duration unknown before call) |
| **S2** | All features **except** `duration` | Realistic pre-call prediction |
| **S3** | Demographics only (age, job, marital, etc.) | Can customer profile alone predict? |
| **S4** | Previous campaign only (pdays, previous, poutcome) | Does past behavior predict future? |

> **Why S2 is the most important scenario**: `duration` (call length) is only known *after* the call ends — so including it in a real-world model is cheating. S2 is the realistic, deployable scenario.

---

## 4. Model Performance Summary

### 4.1 Best Models per Scenario (with SMOTE)

| Scenario | Best Model | Accuracy | Recall | F1 | ROC-AUC |
|----------|-----------|----------|--------|----|---------|
| S1 (all features) | **XGBoost** | 0.8988 | 0.701 | 0.619 | **0.927** |
| S2 (realistic) | **XGBoost** | 0.8974 | 0.582 | 0.570 | **0.902** |
| S3 (demographics) | Random Forest | 0.7128 | 0.576 | 0.319 | 0.693 |
| S4 (previous only) | KNN | 0.8580 | 0.282 | 0.317 | 0.624 |

### 4.2 Recommended Final Model

> **XGBoost + SMOTE trained on S2 (Realistic, no duration)**
> - ROC-AUC = **0.902** | Recall = **0.582** | F1 = **0.570**
> - Optimal decision threshold = **0.44** (maximizes F1)
> - At threshold 0.44: Precision = 0.514, Recall = 0.660, F1 = 0.578

---

## 5. ROC Curve Analysis

ROC curves plot the True Positive Rate (recall) against the False Positive Rate at all possible thresholds. A model with AUC = 1.0 is perfect; AUC = 0.5 means random guessing.

**Key Findings:**
- In **S1**, all models perform very well (AUC 0.85–0.93), but this is partly because `duration` leaks information.
- In **S2** (realistic), XGBoost still achieves AUC **0.902**, showing it can genuinely discriminate subscribers before the call.
- In **S3** (demographics only), performance drops to AUC ~0.62–0.69, confirming that demographic data alone is insufficient.
- In **S4** (previous campaign only), AUC ~0.62, showing past behavior has limited but non-zero predictive value.

---

## 6. Duration Inflation Analysis

| Model | AUC (S1, with duration) | AUC (S2, no duration) | Inflation % |
|-------|------------------------|----------------------|-------------|
| KNN | 0.847 | 0.746 | **13.5%** |
| Logistic Reg | 0.905 | 0.815 | **11.1%** |
| Random Forest | 0.920 | 0.846 | **8.8%** |
| XGBoost | 0.932 | 0.917 | **1.7%** |

**Interpretation**: Adding `duration` artificially inflates ROC-AUC by 1.7–13.5% depending on model. XGBoost is least affected (only 1.7%), meaning it does not rely heavily on `duration` — making it more robust and realistic.

---

## 7. Precision-Recall Curves

For imbalanced datasets, Precision-Recall (PR) curves are more informative than ROC curves.

- **Precision** = of all customers we predict will subscribe, what % actually do? *(Don't waste calls)*
- **Recall** = of all customers who actually subscribe, what % do we catch? *(Don't miss buyers)*

In S2 without SMOTE, most models have low recall (<0.4) — they miss too many actual subscribers. **SMOTE significantly improves recall** by balancing the training data, though at a slight precision cost.

---

## 8. SMOTE Impact on Recall

SMOTE (Synthetic Minority Oversampling TEchnique) creates synthetic "yes" examples in the training set to balance the 89:11 class ratio.

| Model | Recall (S2, No SMOTE) | Recall (S2, With SMOTE) | Improvement |
|-------|----------------------|------------------------|-------------|
| KNN | 0.212 | 0.572 | **+0.360** |
| Logistic Reg | 0.181 | 0.632 | **+0.451** |
| Random Forest | 0.164 | 0.537 | **+0.373** |
| XGBoost | 0.228 | 0.582 | **+0.354** |

SMOTE dramatically improves the ability to catch actual subscribers, at the cost of some precision.

---

## 9. Confusion Matrix Analysis (Top 3 Models in S2)

A confusion matrix shows the breakdown of predictions:

```
               Predicted NO    Predicted YES
Actual NO      True Negative   False Positive (wasted calls)
Actual YES     False Negative  True Positive  (caught buyers!)
```

In a telemarketing context:
- **False Negatives** are costly — these are subscribers we missed.
- **False Positives** waste call center resources but are less harmful.

With SMOTE, XGBoost in S2 achieves a much higher True Positive rate while keeping False Positives manageable.

---

## 10. Scenario Comparison: What Feature Set Matters Most?

| Scenario | XGBoost ROC-AUC | XGBoost Recall |
|----------|----------------|----------------|
| S2 (Realistic) | **0.902** | 0.582 |
| S3 (Demographics) | 0.693 | 0.576 |
| S4 (Previous Campaign) | 0.624 | 0.282 |

**Conclusions:**
- Demographics alone (S3) give moderate discrimination (AUC ~0.69) but low recall without SMOTE
- Previous campaign data alone (S4) is weaker — past campaign outcomes are not a strong standalone predictor
- The **full feature set minus duration (S2)** is clearly the best realistic scenario

---

## 11. Top 15 Most Important Features

Based on combined importance from Random Forest and XGBoost (S2 scenario):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | **housing_yes** (has housing loan) | 0.161 |
| 2 | **poutcome_success** (previous campaign success) | 0.092 |
| 3 | **campaign** (number of contacts this campaign) | 0.076 |
| 4 | contact_missing | 0.068 |
| 5 | housing_no | 0.064 |
| 6 | **balance_per_call** (balance / campaign contacts) | 0.061 |
| 7 | **balance** (account balance) | 0.051 |
| 8 | contact_cellular | 0.047 |
| 9 | loan_yes | 0.036 |
| 10 | marital_married | 0.026 |

**Key Insight**: Customers with **no housing loan**, **higher balance**, **fewer contacts this campaign**, and a **previous successful campaign** are most likely to subscribe.

---

## 12. Threshold Optimization

By default, models predict "yes" when probability > 0.50. But we can optimize this threshold.

**For XGBoost + SMOTE in S2:**
- Default threshold (0.50): Higher precision, lower recall
- **Optimal threshold (0.44)**: Precision=0.514, Recall=0.660, F1=0.578

Lowering the threshold to 0.44 means the bank calls more people (higher recall) while maintaining reasonable precision — a better business trade-off when the goal is not to miss subscribers.

---

## 13. Business Recommendation

**Question: If the bank targets the customers with the highest predicted subscription probability, how efficiently can they run their campaign?**

Using XGBoost + SMOTE (S2, threshold = 0.44):

| % of Customers Called | Customers Called | Buyers Caught | Calls Saved | Lift |
|----------------------|-----------------|---------------|-------------|------|
| Top 10% | 904 | **50.9%** of all buyers | 90% | 5.09x |
| Top 20% | 1,808 | **75.8%** of all buyers | 80% | 3.79x |
| Top 30% | 2,712 | **88.5%** of all buyers | 70% | 2.95x |
| Top 40% | 3,617 | **92.8%** of all buyers | 60% | 2.32x |

**Recommendation**: **Call the top 20% of customers ranked by model score.**
- This catches 75.8% of all buyers while reducing calls by 80%.
- Lift of 3.79x over random calling means the bank is nearly 4× more efficient.
- If budget is tight, targeting the **top 10%** still captures over half of all buyers.

---

## 14. Interpretability Trade-off

| Model | Interpretable? | ROC-AUC (S2, SMOTE) | Recall (S2, SMOTE) |
|-------|---------------|--------------------|--------------------|
| **Logistic Regression** | Yes (coefficients visible) | 0.767 | 0.632 |
| **XGBoost** | No (Black Box) | **0.902** | 0.582 |

**Assessment:**
- XGBoost outperforms Logistic Regression by **13.5 AUC points** (0.902 vs 0.767).
- Logistic Regression is more interpretable (bank regulators can audit coefficients) but significantly weaker.
- **Recommendation**: Use XGBoost for prediction, use Logistic Regression for stakeholder explanation.
- Feature importance and SHAP values (if implemented) can partially explain XGBoost decisions.

---

## 15. Answers to Key Questions

| Question | Answer |
|----------|--------|
| **What is the final recommended model?** | XGBoost + SMOTE, trained on S2 (no duration) |
| **At what probability threshold should the bank call?** | **0.44** (maximizes F1, balances precision/recall) |
| **If bank calls top 20%, what % of buyers do they catch?** | **75.8%** of all actual subscribers |
| **Which 3 variables matter most?** | **housing_yes, poutcome_success, campaign** |
| **Is the most accurate model interpretable? Is it worth it?** | XGBoost is a black box but 13.5% more accurate — worth it for targeting, but supplement with LR for explanations |
| **How much does duration inflate performance?** | **1.7–13.5%** depending on model (XGBoost least affected) |
| **Does demographics-only work?** | Moderately (AUC ~0.69) but not well enough for deployment alone |
| **Does previous campaign help?** | Weakly (AUC ~0.62); `poutcome_success` is the most useful previous-campaign feature |

---

## 16. Conclusions

1. **XGBoost is the best model** for this bank telemarketing prediction task, achieving ROC-AUC of 0.902 in the realistic (S2) scenario.
2. **SMOTE is essential** — it dramatically improves recall from ~0.23 to ~0.58, making the model practically useful.
3. **The duration variable inflates performance** by up to 13.5% and should be excluded for real-world deployment.
4. **The bank should target the top 20%** of customers by model score, achieving 3.79× efficiency over random calling.
5. **Demographics alone are insufficient** — the full feature set is needed for good discrimination.

---

## 17. Files Produced by Member 4

### Figures (`results/figures/`)
| File | Description |
|------|-------------|
| `roc_curves.png` | ROC curves for all models, all scenarios (no SMOTE) |
| `roc_curves_smote.png` | ROC curves for all models, all scenarios (with SMOTE) |
| `roc_curves_S1.png` to `roc_curves_S4.png` | Individual scenario ROC curves (solid=no SMOTE, dashed=SMOTE) |
| `pr_curves.png` | Precision-Recall curves for all models and scenarios |
| `duration_inflation.png` | Bar chart: AUC inflation from including duration (S1 vs S2) |
| `scenario_comparison.png` | Bar chart: ROC-AUC and Recall across S2/S3/S4 |
| `smote_impact.png` | Bar chart: recall before vs after SMOTE |
| `feature_importance_combined.png` | Top 15 features (RF + XGBoost combined) |
| `threshold_optimization.png` | F1/Precision/Recall vs threshold curve |
| `business_recommendation.png` | Cumulative gains curve |
| `confusion_matrices/cm_top3_no_smote.png` | Confusion matrices for top 3 models (no SMOTE) |
| `confusion_matrices/cm_top3_smote.png` | Confusion matrices for top 3 models (with SMOTE) |

### Tables (`results/tables/`)
| File | Description |
|------|-------------|
| `final_comparison.csv` | All models, all scenarios, all metrics combined |
| `duration_inflation.csv` | AUC inflation values per model |
| `business_recommendation.csv` | Buyer catch rate at different call volumes |
| `feature_importance_combined.csv` | Top 15 features with importance scores |
| `interpretability_tradeoff.csv` | XGBoost vs Logistic Regression comparison |

### Code (`scripts/`)
| File | Description |
|------|-------------|
| `evaluation_fast.py` | Full evaluation script (all figures + tables) |

---

*Report prepared by Member 4 (Joey) | ECON 7970 Team 10 | April 2026*
