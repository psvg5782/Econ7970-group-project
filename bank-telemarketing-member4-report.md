# Bank Telemarketing Prediction: Model Evaluation & Business Recommendation
### Team 10 — Econ 7970 Applied Predictive Modeling — Member 4 Report

---

## Executive Summary

This report presents the evaluation of eight classification models — four baseline and four advanced — applied to the Bank of Portugal telemarketing dataset to predict whether a client will subscribe to a term deposit. A total of 64 model configurations are assessed across four feature scenarios and two SMOTE conditions.

The central finding is that **XGBoost without SMOTE, trained on all features excluding call duration (Scenario S2), is the recommended production model**, achieving an ROC-AUC of 0.7940. Applying this model with a probability threshold of approximately 0.196 yields an F1 score of 0.486. A top-20% risk-ranked call list built from this model captures **60.2% of all potential subscribers**, representing roughly a 3× improvement over undirected random dialing.

A critical methodological finding is that the feature `duration` — the length of the last phone call — causes severe metric inflation. Models trained with this feature show AUC values 16–36 percentage points higher than those trained without it. Because call duration is unknown before a call is initiated, its inclusion constitutes data leakage and renders Scenario S1 results unsuitable for deployment evaluation.

---

## 1. Dataset and Methodology

### 1.1 Dataset Overview

The dataset is the UCI Bank Marketing dataset (`bank-full.csv`), containing 45,211 records of direct phone marketing campaigns conducted by a Portuguese bank. The binary target variable indicates whether the client subscribed to a term deposit (`yes` / `no`).

| Attribute | Value |
|---|---|
| Total records | 45,211 |
| Class balance (no / yes) | 88.3% / 11.7% |
| Train set (80%) | 36,168 records |
| Test set (20%) | 9,043 records |
| Train/test split | Stratified, `random_state=42` |

The strong class imbalance — roughly 7.5 non-subscribers for every subscriber — is a defining challenge for all models evaluated. Standard accuracy is therefore a misleading metric; ROC-AUC, precision, recall, and F1 are the primary evaluation criteria.

### 1.2 Feature Set

Following the decisions made by Members 2 and 3, all models use the original 15 features from `bankclean.csv`. Member 1's engineered features (`isretired`, `isstudent`, `balance_per_call`, pre-encoded month dummies) are deliberately excluded to ensure cross-member comparability.

The 15 original features are:

| Category | Features |
|---|---|
| Client demographics | `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan` |
| Last contact | `contact`, `month`, `duration` |
| Campaign | `campaign`, `pdays`, `previous`, `poutcome` |

Categorical features are one-hot encoded; `duration` is withheld in Scenarios S2–S4 to simulate realistic deployment conditions.

### 1.3 Four Evaluation Scenarios

Four feature scenarios are constructed to isolate the contribution of different information groups and to quantify the effect of the leaky `duration` variable:

| Scenario | Features Used | Purpose |
|---|---|---|
| **S1** — All Features (With Duration) | All 15 features | Upper-bound benchmark; reveals data leakage |
| **S2** — Realistic (No Duration) | All features except `duration` | **Primary deployment scenario** |
| **S3** — Demographics Only | `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan` | What client profile alone predicts |
| **S4** — Previous Campaign Only | `pdays`, `previous`, `poutcome` | What prior campaign history alone predicts |

### 1.4 Models Evaluated

**Baseline models (Member 2):** Logistic Regression, Lasso (L1 regularisation), Ridge (L2 regularisation), Decision Tree.

**Advanced models (Member 3):** K-Nearest Neighbours (KNN), Random Forest, Support Vector Machine with RBF kernel (SVM), XGBoost.

Each of the eight models is evaluated with and without SMOTE (Synthetic Minority Over-sampling Technique) across all four scenarios, producing 64 total evaluation rows. All SMOTE operations use `random_state=42`.

---

## 2. Exploratory Data Analysis

Key distributional findings from Member 1's EDA inform the modelling choices throughout this report:

- **Call duration dominates outcomes.** Clients who eventually subscribed had significantly longer average call durations than non-subscribers. This correlation is the source of the severe data leakage quantified in Section 5.
- **Previous campaign success is the strongest deployable predictor.** Clients who subscribed in a prior campaign (`poutcome = success`) exhibit a substantially higher subscription rate in the current campaign — the most actionable signal available at call time.
- **Seasonality matters.** Campaigns run in March, September, October, and December show markedly higher conversion rates than those run in May or June.
- **Job and balance segment clients.** Students and retired clients subscribe at above-average rates; clients in blue-collar jobs and with outstanding loans subscribe at below-average rates.

---

## 3. Baseline Model Results (Member 2)

Logistic Regression, Lasso, Ridge, and Decision Tree were trained by Member 2 using stratified 80/20 splits with `random_state=42`.

### 3.1 S1 — With Duration (Inflated)

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9058 | 0.9015 | 0.647 | 0.348 | 0.452 |
| Lasso (L1) | 0.9058 | 0.9016 | 0.647 | 0.349 | 0.453 |
| Ridge (L2) | 0.9058 | 0.9015 | 0.647 | 0.348 | 0.452 |
| Decision Tree | 0.8862 | 0.9008 | 0.607 | 0.430 | 0.504 |

### 3.2 S2 — Realistic (No Duration)

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7724 | 0.8933 | 0.661 | 0.181 | 0.284 |
| Lasso (L1) | 0.7724 | 0.8935 | 0.664 | 0.181 | 0.285 |
| Ridge (L2) | 0.7724 | 0.8933 | 0.661 | 0.181 | 0.284 |
| Decision Tree | 0.7518 | 0.8935 | 0.639 | 0.206 | 0.312 |

The three regularised linear models produce near-identical results, suggesting that the L1/L2 penalty has minimal differentiation effect at this feature set size. The Decision Tree underperforms in AUC but achieves a somewhat higher recall, reflecting its tendency to produce more granular probability estimates.

### 3.3 S2 — With SMOTE

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7669 | 0.7467 | 0.260 | 0.632 | 0.369 |
| Lasso (L1) | 0.7670 | 0.7461 | 0.259 | 0.629 | 0.367 |
| Ridge (L2) | 0.7669 | 0.7467 | 0.260 | 0.632 | 0.369 |
| Decision Tree | 0.7478 | 0.8545 | 0.396 | 0.462 | 0.426 |

SMOTE increases recall substantially for linear models (from ~0.18 to ~0.63) at the cost of a sharp drop in precision (from ~0.66 to ~0.26) and overall accuracy. This trade-off is examined further in Section 6.

---

## 4. Advanced Model Results (Member 3)

Random Forest, KNN, XGBoost, and SVM (RBF kernel) were trained by Member 3 under identical train/test splits and SMOTE conditions.

### 4.1 S1 — With Duration (Inflated)

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | **0.9279** | 0.9097 | 0.664 | 0.460 | 0.544 |
| Random Forest | 0.9196 | 0.8970 | 0.749 | 0.181 | 0.291 |
| KNN | 0.8257 | 0.8947 | 0.583 | 0.353 | 0.440 |
| SVM (RBF) | 0.7244 | 0.8017 | 0.278 | 0.437 | 0.340 |

XGBoost achieves the highest overall AUC of any model across the entire study (0.9279), though this figure is substantially inflated by the `duration` feature.

### 4.2 S2 — Realistic (No Duration)

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | **0.7940** | 0.8965 | 0.669 | 0.228 | 0.340 |
| Random Forest | 0.7912 | 0.8943 | 0.709 | 0.164 | 0.266 |
| KNN | 0.7000 | 0.8865 | 0.538 | 0.212 | 0.304 |
| SVM (RBF) | 0.5330 | 0.6195 | 0.128 | 0.388 | 0.192 |

![ROC Curves — S2 Realistic (No Duration)](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/24a8692e-3e51-4fc9-9f23-8d90434008de/results/figures/roc/roc_curves_S2.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvMjRhODY5MmUtM2U1MS00ZmM5LTlmMjMtOGQ5MDQzNDAwOGRlL3Jlc3VsdHMvZmlndXJlcy9yb2Mvcm9jX2N1cnZlc19TMi5wbmc~KiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3Njc5MjIwOX19fV19&Signature=de4qC0S-urmZrtJRFDRPYBhew2Yi359GJeBMVKtiTyEliFulYpEw89hogZACaIyZq7udJCdE0jhI77-c1pGzoADoIKT-ytesKYo8lsRzGc7WdiNxTeuFN8Ou9PNoy-dkj1TZnLw7eLgvfPoQFTRmX6e7kra6hX2fXwOwJpVkD2OgqF-uhFuhB3KRMW42RXaQj09VsFuwRC45VKLr6C8Lm1NsjcV1Weo45M6CeXHsp5OUe-YzaF5-sIElRZMNCPQVYwxU~BI099PJ4nSCsfu6wfDmUMx8BCRlU9ejyFTNtv8uXA8QOrlO8AqQ-rJorJVGYzaPEFAASoU5CTgEdj9ceg__&Key-Pair-Id=K1BF7XGXAIMYNX)

In the realistic scenario, XGBoost and Random Forest are tightly competitive, separated by only 0.003 AUC. Both substantially outperform KNN. SVM (RBF) collapses to near-random performance (AUC 0.533), suggesting the RBF kernel is poorly suited to this feature space without duration.

### 4.3 S2 — With SMOTE

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest | **0.7794** | 0.8402 | 0.373 | 0.537 | 0.440 |
| XGBoost | 0.7712 | 0.8875 | 0.528 | 0.370 | 0.435 |
| KNN | 0.7035 | 0.7367 | 0.239 | 0.572 | 0.337 |
| SVM (RBF) | 0.4387 | 0.2242 | 0.107 | 0.768 | 0.188 |

SMOTE narrows the AUC gap between Random Forest and XGBoost while improving recall for both. However, precision drops substantially. The choice between SMOTE and no-SMOTE configurations is operationally driven and is addressed in Section 6.

### 4.4 Precision-Recall Curve — S2 Realistic

![Precision-Recall Curves — S2 Realistic](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/2cab68a5-2b68-4133-b50f-959e8f42b371/results/figures/pr/pr_curves_S2.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvMmNhYjY4YTUtMmI2OC00MTMzLWI1MGYtOTU5ZThmNDJiMzcxL3Jlc3VsdHMvZmlndXJlcy9wci9wcl9jdXJ2ZXNfUzIucG5nPyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY3OTIyMDl9fX1dfQ__&Signature=S~fFEqCPOWNMcCr1itXd9235b~hoVh2TdgciPbWXHf0DcOIjFyXkKAa8BagD9SsFwXavGyrQAdQj4PXUl8AfzsFyLJkkC8XRW1tRB5~FIyJUc0kgHwpSVTRzcQVDUHjSJfWfgpdogl9X-DPb5q4NPI-vsYrmV8ENZ7BawhnlN98kglXfMzCQqJns2Y-foh7LTgfq9CKec~lO9QeE5vaXmTrjMZ-kN35kKMncnZeE-4uU0miWkdrc~mCyyAvLJk-nryZInybb6z3A97MUOZorJ7yFFOlCzDUg3jxVfEBvq3gvemWwZOGOZWmlhywzwbqqweU7vgaM~2zS7VNOpFHD2w__&Key-Pair-Id=K1BF7XGXAIMYNX)

The PR curve provides a complementary view to ROC-AUC under class imbalance. XGBoost and Random Forest maintain the highest average precision across the recall range, confirming their superiority in the realistic deployment scenario. The SVM curve falls close to the random baseline (a horizontal line at the positive class rate of 0.117).

### 4.5 S3 — Demographics Only

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | **0.7050** | 0.333 | 0.006 | 0.011 |
| XGBoost | 0.7030 | 0.448 | 0.028 | 0.053 |
| Decision Tree | 0.6719 | 0.363 | 0.039 | 0.070 |
| Lasso/LR/Ridge | ~0.666 | 0.000 | 0.000 | 0.000 |
| KNN | 0.6461 | 0.366 | 0.131 | 0.193 |
| SVM (RBF) | 0.4043 | 0.091 | 0.439 | 0.151 |

![ROC Curves — S3 Demographics Only](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/79b0f7e9-b545-4b89-83f2-b598fff075a1/results/figures/roc/roc_curves_S3.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvNzliMGY3ZTktYjU0NS00Yjg5LTgzZjItYjU5OGZmZjA3NWExL3Jlc3VsdHMvZmlndXJlcy9yb2Mvcm9jX2N1cnZlc19TMy5wbmc~KiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3Njc5MjIwOX19fV19&Signature=cu98L40wB-7QZ--TNmURU3OIuRCVdTC8pj65OkXD-wGZVykJqSg4gxcdPvCeLwU2TcWXUwvx3pIgh~yXM3w9BgMd-f58PAAegHTIWeXHoLg-5RjpWJC3I5jaLrGVctpHDOftMs4X5joNdMl56YkBpWD3~gQDM3MSb5~brFu7NIW1djglJ8Nof5c9R6AKh34qTiDCVHIvXXT3Puf2lr8RzCkKkIlSDDQg4I5ihjpnPl1lTw~IIypcX4DE0umKlw2Z1fHjGJ2oMSzNaUEkOcAOXVPk~7~RTPyIgmT96zXFhC5POnW6VNTGN6uNzba84rbntgYt33eiTq0TB3GncySgAQ__&Key-Pair-Id=K1BF7XGXAIMYNX)

When restricted to demographic features alone, all models degrade substantially. Notably, Logistic Regression, Lasso, and Ridge all default to predicting the majority class exclusively (Precision = Recall = F1 = 0), indicating that linear separability breaks down entirely without campaign history. Random Forest and XGBoost retain partial discriminatory power (AUC ≈ 0.70) through non-linear interactions among demographic variables.

### 4.6 S4 — Previous Campaign Only

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Ridge (L2) | **0.6223** | 0.650 | 0.188 | 0.292 |
| Logistic Regression | 0.6223 | 0.650 | 0.188 | 0.292 |
| Lasso (L1) | 0.6223 | 0.650 | 0.188 | 0.292 |
| Random Forest | 0.6207 | 0.668 | 0.173 | 0.275 |
| XGBoost | 0.6199 | 0.674 | 0.166 | 0.267 |
| KNN | 0.5895 | 0.594 | 0.164 | 0.258 |
| SVM (RBF) | 0.5271 | 0.110 | 0.794 | 0.193 |

![ROC Curves — S4 Previous Campaign Only](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/1702c555-e556-4259-90ef-fe66404b3a00/results/figures/roc/roc_curves_S4.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvMTcwMmM1NTUtZTU1Ni00MjU5LTkwZWYtZmU2NjQwNGIzYTAwL3Jlc3VsdHMvZmlndXJlcy9yb2Mvcm9jX2N1cnZlc19TNC5wbmc~KiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3Njc5MjIwOX19fV19&Signature=SJuaKuot4jCjHzgYNBrOaBVPDn-exR4u2MK5Nzij5fALtkHUJ9N6wGmyAxmyZsxS3Cj9dQdwRPltSEARsBMuinN-kN6wg30Q2XOs8VTjg00I3mOhRYYaRd~gx9HwcLMbXuzZj49~RT0Hg8Ltd8XXGzif8UZryoA~8OsmbAlVrETzA2YBasLWlthuQU-xzXUZvzjprir-jcy3wca5vrO9UP2U73kF3OY2tIicIEkfgJ5n9zfsW2PuIsgJS5u5lCu6cQsEeykrYSQxKkaTVdAWW2N0emmsBm8uKZ5MmH8miOq-zpXeg6txjawGZ2qxgC5I8Fu0eDFQmVuEi~tTbZ8Ekw__&Key-Pair-Id=K1BF7XGXAIMYNX)

Prior campaign history alone (days since last contact, number of previous contacts, and prior outcome) yields AUC values in the range 0.59–0.62 for most models. Linear models perform on par with tree ensembles in this scenario, confirming that the relationship between previous campaign features and subscription is largely linear. The `poutcome_success` flag is the dominant signal in this scenario.

---

## 5. Duration Inflation Analysis

`duration` — the length of the last phone call in seconds — is the most powerful predictor in the dataset, but it is also leaky: its value is not known before a call is made. Including it artificially inflates model performance and produces an overly optimistic picture of real-world predictive power.

![Duration Inflation: S1 vs S2 AUC Comparison](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/03fcc62d-e47a-420a-badc-0494f86ab5c7/results/figures/analysis/duration_inflation.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvMDNmY2M2MmQtZTQ3YS00MjBhLWJhZGMtMDQ5NGY4NmFiNWM3L3Jlc3VsdHMvZmlndXJlcy9hbmFseXNpcy9kdXJhdGlvbl9pbmZsYXRpb24ucG5nPyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY3OTIyMDl9fX1dfQ__&Signature=PwMAAdWd9wLVKDJbGBM3kJvPQm--dIoS1mUDFvcG8Es3tu~M5obhtnzT19Gm18cTydoZGAdg8DkTV6HN8IF8Qr~Gw6bKJD4LQp~noJ9y~EQMYK7HmDCETNAeH-bFMu80F8nCop3j9jcA7p-I3sP3N1cA~BMuaB-N7CiIjLBQiybd8dOnBrjrFfqivagQRVphut6Mu1cKASUy1OMEhIzO~PcJHlYWHDkclf9EZw1E-1EG1YzXy5gKv6jEV1ZU8AqgoiuY812-8sEn9UFD7ZMboBxX5726osYSAh-8ILzRtPjUdzkFRR3ox8cnOkxueVmAkOWhOAa9Tyoo6kLVhkaZMg__&Key-Pair-Id=K1BF7XGXAIMYNX)

### 5.1 Inflation Quantified

| Model | AUC S1 (With Duration) | AUC S2 (No Duration) | Inflation |
|---|---|---|---|
| XGBoost | 0.9279 | 0.7940 | **16.9%** |
| Random Forest | 0.9196 | 0.7912 | **16.2%** |
| Logistic Regression | 0.9058 | 0.7724 | **17.3%** |
| Ridge (L2) | 0.9058 | 0.7724 | **17.3%** |
| Lasso (L1) | 0.9058 | 0.7724 | **17.3%** |
| Decision Tree | 0.8862 | 0.7518 | **17.9%** |
| KNN | 0.8257 | 0.7000 | **18.0%** |
| **SVM (RBF)** | **0.7244** | **0.5330** | **35.9%** |

All models show 16–18 percentage point inflation. SVM (RBF) is an outlier, with 35.9% inflation — indicating that the kernel relies disproportionately on the duration feature for non-linear separation, rendering its S2 AUC (0.533) essentially uninformative.

### 5.2 Interpretation

The inflation pattern is consistent with the underlying data generating process: once a client is on the phone for an extended time, they have already decided (or been persuaded) to subscribe. Including duration therefore encodes the outcome into a predictor, violating the temporal logic required for a real scoring system. All deployment-oriented conclusions in this report are drawn exclusively from S2 (no duration) results.

---

## 6. Scenario Comparison

![Scenario Comparison: S2 vs S3 vs S4 (Best Model per Scenario)](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/ff70dee3-66f3-4538-8cd2-4c1eae88d4f8/results/figures/analysis/scenario_comparison.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvZmY3MGRlZTMtNjZmMy00NTM4LThjZDItNGMxZWFlODhkNGY4L3Jlc3VsdHMvZmlndXJlcy9hbmFseXNpcy9zY2VuYXJpb19jb21wYXJpc29uLnBuZz8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzc2NzkyMjA5fX19XX0_&Signature=SDNSNcKrAayInxtE509JPRu8WDSsCwFU-wfHt7dcNR1zUI8Pv2PsNMcFA4dWVXI6cpgMehClCzujgGNmMf2kOA6T3AE6T7R8AZdQLTv9ouPJTbgIguH-h70YDeWlU7i4C5jz-XbUZFt5nXp6BZ5X50A1g6gR3s4T2VFK6YGm-BvwSQSeK8d5iVVnT9boZxhl~r1K57oYMfG02iUjjMsezbx9TmyxuOvSz7vnrQMVTFDUtzaxX5lYoTPF8bSJ0bw35pXGRe2rceRXhypkMjYt3-DnkHrLg~LkIr1mpI~02ZK7amskTLc1PrCrbewxeDbOuUAITb3xkySv-eXbAuXQVQ__&Key-Pair-Id=K1BF7XGXAIMYNX)

Comparing the best-performing model in each realistic scenario illuminates the marginal value of different information groups:

| Scenario | Best Model | AUC | Information Available |
|---|---|---|---|
| S2 — Realistic | XGBoost | **0.7940** | All features except duration |
| S3 — Demographics | Random Forest | 0.7050 | Client profile only |
| S4 — Previous Campaign | Ridge (L2) | 0.6223 | Campaign history only |

The gap between S2 and S3 (0.089 AUC) reflects the value added by campaign-related features — contact method, timing (month), and prior campaign history — beyond demographics alone. The gap between S3 and S4 (0.083 AUC) shows that demographic context contributes roughly as much as campaign history in isolation. Together, the two information groups are complementary and mutually reinforcing.

---

## 7. SMOTE Impact Analysis

![SMOTE Impact on AUC, Recall, and F1 — S2 Scenario](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/33855673-9bdc-4e42-8623-a7768e10d408/results/figures/analysis/smote_impact.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvMzM4NTU2NzMtOWJkYy00ZTQyLTg2MjMtYTc3NjhlMTBkNDA4L3Jlc3VsdHMvZmlndXJlcy9hbmFseXNpcy9zbW90ZV9pbXBhY3QucG5nPyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY3OTIyMDl9fX1dfQ__&Signature=NIaGUYrkVGrgZuR4AluuidgRhe7wnpbi1NGFrICDWLeUF3gyCoGNmrpHRRyGi4bOErnpuRLJFRvIm1pD3P~VyJgQNsKT-Y~ONE57UH2UlkzjtrHdIF-442D9ur1maUQsQTah396lc~wK3iRjGOhRlfUxy2p65nC-cLSCROxty2VtRF7AZYUJ~j5xJnHj9iSg4jZKerZkOI5m3CDq~RtY3AEKXPZp3sVhUsVpCwS4oo8yDyA6rFepSPy~p2spmln~HAnaUubWZ0PA49Ir~HBMEr0qlCjWrtm4QHsejUgMae9sCblUBvgcXwMp6qKc5hcNV6c0a3PCVpUqbWys~wIj~A__&Key-Pair-Id=K1BF7XGXAIMYNX)

SMOTE oversamples the minority class during training to address class imbalance. Its effects on model performance are nuanced:

| Model | AUC (No SMOTE) | AUC (SMOTE) | Recall (No SMOTE) | Recall (SMOTE) |
|---|---|---|---|---|
| XGBoost | 0.7940 | 0.7712 | 0.228 | 0.370 |
| Random Forest | 0.7912 | 0.7794 | 0.164 | 0.537 |
| Logistic Regression | 0.7724 | 0.7669 | 0.181 | 0.632 |
| Decision Tree | 0.7518 | 0.7478 | 0.206 | 0.462 |
| KNN | 0.7000 | 0.7035 | 0.212 | 0.572 |
| SVM (RBF) | 0.5330 | 0.4387 | 0.388 | 0.768 |

**Key observations:**

- **AUC falls slightly with SMOTE** for almost all models. Synthetic oversampling does not add new information; it shifts the decision boundary toward higher recall at the cost of precision.
- **Recall improves dramatically with SMOTE**, particularly for linear models and Random Forest. Linear Regression recall jumps from 0.181 to 0.632.
- **The optimal SMOTE choice depends on the operational objective.** If call-centre capacity is constrained and high precision matters (e.g., limited budget per campaign), No-SMOTE configurations are preferred. If recall matters more (e.g., management requires capturing as many buyers as possible), SMOTE configurations are preferred.
- **SVM (RBF) with SMOTE is harmful**, reducing AUC from an already poor 0.533 to 0.439 while pushing recall to 0.768 at the cost of near-total precision collapse.

For the recommended business use case (top-20% risk ranking), No-SMOTE XGBoost is preferred because ranking quality (AUC) matters more than absolute recall at a fixed threshold.

---

## 8. Feature Importance

![Feature Importance — Random Forest and XGBoost Combined (S2)](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/acec2b7d-8f4e-460e-a67e-ac5cab359a4a/results/figures/analysis/feature_importance_combined.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvYWNlYzJiN2QtOGY0ZS00NjBlLWE2N2UtYWM1Y2FiMzU5YTRhL3Jlc3VsdHMvZmlndXJlcy9hbmFseXNpcy9mZWF0dXJlX2ltcG9ydGFuY2VfY29tYmluZWQucG5nPyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY3OTIyMDl9fX1dfQ__&Signature=H0JWKtdpN1QCohxe01YAzO3tVhHrA1yVowKemi0eK4hkaGLG8CsYPhgjlsb5CM88PikVItY1PovaN-JK-t32CQ7TAmOs8ZELqdui~Po7R9AnPNjP0dFF-~klRxCDHuNxtih0a3GiclcADedl2Hb4K42tWjkU9AQ7vU1JFIErvPaFlioUG2Kd4LFHnjY~k5-CHH5tZFfna~cVgMEDnFlF8-B~kwYsLoTJJ1AbzElUsL6A5I7ZHA-WY5H8BFphM3fcvnFPWgb9TBtmtIr-R23YaTgluT2gY0SxHcJTLHzvvcSoxBgzwcYwgIkLRtxaVljNdQsIcdOfzW75bavjYF4rYg__&Key-Pair-Id=K1BF7XGXAIMYNX)

Feature importance is computed from Random Forest (Gini impurity decrease) and XGBoost (gain-based), then averaged. The top ten features for the S2 scenario are:

| Rank | Feature | RF Importance | XGB Importance | Mean |
|---|---|---|---|---|
| 1 | `poutcome_success` | 0.1805 | 0.4364 | **0.3084** |
| 2 | `contact_missing` | 0.0331 | 0.0995 | **0.0663** |
| 3 | `age` | 0.0917 | 0.0084 | **0.0500** |
| 4 | `pdays` | 0.0834 | 0.0091 | **0.0463** |
| 5 | `month_mar` | 0.0376 | 0.0493 | **0.0435** |
| 6 | `housing_no` | 0.0389 | 0.0277 | **0.0333** |
| 7 | `balance` | 0.0581 | 0.0060 | **0.0321** |
| 8 | `month_apr` | 0.0237 | 0.0376 | **0.0307** |
| 9 | `month_oct` | 0.0303 | 0.0254 | **0.0278** |
| 10 | `month_jun` | 0.0210 | 0.0310 | **0.0260** |

**Interpretation:**

`poutcome_success` is by far the dominant feature, with a mean importance of 0.308 — more than four times the next feature. Clients who previously subscribed to a term deposit are highly likely to do so again, making prior outcome the most actionable targeting criterion. XGBoost assigns this feature an importance of 0.436, underscoring how heavily gradient boosting relies on it.

`contact_missing` (clients whose contact type is unknown) is the second most important feature by mean importance, though the direction of its effect requires interpretation against the training data distribution.

`age`, `pdays` (days since last contact), and calendar-month dummies (March, April, October) round out the top features, consistent with the EDA finding that seasonality and client demographics carry meaningful signals.

Notably, there is significant disagreement between RF and XGB on several features: `age` (RF=0.092, XGB=0.008) and `balance` (RF=0.058, XGB=0.006). XGBoost focuses more narrowly on `poutcome_success` and `contact_missing`, while Random Forest distributes importance more broadly. This divergence reflects the different learning dynamics of the two ensembles.

---

## 9. Threshold Optimisation

Standard classification defaults to a probability threshold of 0.5. Under severe class imbalance, this threshold is suboptimal. Threshold optimisation identifies the probability cutoff that maximises a chosen evaluation metric.

![Threshold Optimisation — XGBoost S2 No SMOTE](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/6c912a0c-4680-4a47-9151-cc9a5c595149/results/figures/analysis/threshold_optimisation.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvNmM5MTJhMGMtNDY4MC00YTQ3LTkxNTEtY2M5YTVjNTk1MTQ5L3Jlc3VsdHMvZmlndXJlcy9hbmFseXNpcy90aHJlc2hvbGRfb3B0aW1pc2F0aW9uLnBuZz8qIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzc2NzkyMjA5fX19XX0_&Signature=ja9Se3ul3aRc11-hfEpwnAFq60ESV2awtajdkFZtZTQUzZZJXo3KhviIbzxwXDw6aUbldU~8uRiCpKcHVCIiilV6QJDFq4pvRQm0e02WZ3VNgp97DT5xnza42NvYfLWw5N0meQJCZxfjioSCK81buLEgEq9VJjRlOQuvH3JbbI5v2zm1xVKMFgvmW8v7rROHF-gdk6JdUEKxQzCHVuDUisEIlISbXibeOwzizMnj2kb4HteDbCXsqMUgerMvAMvd40NiA3FC58N-obdTjnPO3gnLNoe1BjSGykH6svhrnQSt5DGx3nFMbJTGPnXm88EDl8vNHSjD9ywP0yGz2i85lg__&Key-Pair-Id=K1BF7XGXAIMYNX)

### 9.1 Threshold Selection for XGBoost S2

| Objective | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Maximise F1 | 0.196 | 0.458 | 0.516 | **0.486** |
| Recall ≥ 60% | 0.132 | 0.357 | 0.600 | 0.447 |
| Default (0.50) | 0.500 | 0.669 | 0.228 | 0.340 |

At the default threshold of 0.5, the model is highly conservative — it predicts "yes" only when it is quite confident, yielding high precision (0.669) but low recall (0.228) and a modest F1 of 0.340. Lowering the threshold to 0.196 raises F1 to 0.486 by balancing precision and recall more evenly.

For the business application (selecting whom to call), the appropriate threshold is determined by the call centre's budget and acceptable false positive rate, as discussed in Section 10.

### 9.2 Confusion Matrices — Top-3 Models (S2 No SMOTE)

![Confusion Matrices — XGBoost, Random Forest, Logistic Regression (S2 No SMOTE)](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/4443ab9c-c1cf-46b0-9c10-493a1918dc16/results/figures/confusion_matrices/confusion_matrices_top3_S2.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvNDQ0M2FiOWMtYzFjZi00NmIwLTljMTAtNDkzYTE5MThkYzE2L3Jlc3VsdHMvZmlndXJlcy9jb25mdXNpb25fbWF0cmljZXMvY29uZnVzaW9uX21hdHJpY2VzX3RvcDNfUzIucG5nPyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY3OTIyMDl9fX1dfQ__&Signature=kvKP5s0roAXz6PtEMvUxiHs~RUaci9Wq96vktZ~ztgmyfQ-tJObD2J6AK1t~sAxhh7fucEpKFoo1tyooJ0ylqhm4wwwETHoOlfNJYCv9JN4sD-qIh3scsEvCUrzwvvgB~vFK58sSeCyve2kdbP14cSOlPlP1c4w1oK60HrxSxcuoaT2PiV26b-6YAcdzL89W6P7t7~04QHxVcC4vr4d80WuhuFaMj5JXBEVgmD0~xSSqIPmErkVwAAYgfcw~n-EcjvE6Z5SGkhRahDOfTDD0RgLC0WIz4ri4EBgSPi5gLZZeLBLk5rQ2zhlgWtVgfLg5cQO77AGtzw8sZ5YywvJ6QA__&Key-Pair-Id=K1BF7XGXAIMYNX)

The confusion matrices confirm that all top-performing models are heavily biased toward predicting the negative class under the default threshold, consistent with the class imbalance. XGBoost correctly identifies the largest share of true positives while maintaining competitive precision.

---

## 10. Business Recommendation

### 10.1 Model Selection

**Recommended configuration: XGBoost, Scenario S2 (No Duration), No SMOTE.**

| Criterion | Recommendation | Rationale |
|---|---|---|
| Model | XGBoost | Highest AUC (0.794) in realistic scenario |
| Scenario | S2 — No Duration | Duration is unavailable pre-call; S1 results are misleading |
| SMOTE | No | Higher AUC and precision; ranking quality preferred over raw recall |
| Threshold | 0.196 (F1-max) or 0.132 (Recall≥60%) | Adjustable based on call-centre budget |

### 10.2 Top-20% Risk-Ranked Call List

The most practical operational output is a probability-ranked call list. Restricting outreach to the top 20% of clients by predicted probability transforms model performance into a concrete business outcome:

| Metric | Value |
|---|---|
| Total test records | 9,043 |
| Top-20% list size | 1,808 calls |
| True subscribers in top 20% | ~637 out of 1,058 total |
| Buyer capture rate (Recall) | **60.2%** |
| Precision in top 20% | 0.352 |
| Expected lift over random | **~3×** |

A random dialling strategy contacting 20% of clients would reach approximately 20% of subscribers by chance. The XGBoost model concentrates 60.2% of subscribers into the top-20% list — a lift of approximately 3×. This substantially reduces the number of unproductive calls and improves the cost-effectiveness of the campaign.

### 10.3 Comparison of Top Configurations

| Model | SMOTE | AUC | F1 (Optimal Thresh.) | Top-20% Buyer Capture |
|---|---|---|---|---|
| **XGBoost** | **No** | **0.7940** | **0.486** | **60.2%** |
| Random Forest | No | 0.7912 | 0.484 | 59.2% |
| Random Forest | Yes | 0.7794 | 0.456 | 57.3% |
| Logistic Regression | No | 0.7724 | 0.454 | 56.0% |
| XGBoost | Yes | 0.7704 | 0.452 | 56.2% |
| Logistic Regression | Yes | 0.7669 | 0.452 | 55.6% |
| Decision Tree | No | 0.7518 | 0.439 | 53.4% |
| Decision Tree | Yes | 0.7478 | 0.442 | 54.6% |

XGBoost (No SMOTE) leads on all three primary metrics. The gap between XGBoost and the next-best option (Random Forest No SMOTE) is small (0.003 AUC, 1.0 percentage point in buyer capture), suggesting that Random Forest is a reasonable fallback with nearly identical operational value.

### 10.4 Actionable Operational Recommendations

1. **Deploy XGBoost S2 as the primary scoring model.** Score all prospective clients on the available 14 features (excluding `duration`) and rank by predicted probability.

2. **Use threshold 0.196 as the default operating point.** This maximises F1. Adjust toward 0.132 if the bank prioritises not missing potential subscribers over call efficiency.

3. **Never use `duration` in production.** Models trained with duration cannot be meaningfully evaluated before deployment, as the feature is realised only after the call ends.

4. **Focus calls on clients with `poutcome = success`.** The single most predictive signal is a prior successful subscription. Even without a full model, targeting previous subscribers generates substantial lift.

5. **Prioritise March, September, October, and December campaigns.** The strong monthly seasonality effect identified in EDA and feature importance analysis suggests campaign timing significantly affects conversion rates.

6. **Revisit SMOTE if recall targets change.** If a future campaign demands capturing 60%+ of subscribers (e.g., targeting a small high-value segment), SMOTE configurations of Random Forest (Recall=0.537) or Logistic Regression (Recall=0.632) should be evaluated, accepting lower precision.

7. **Monitor model decay over time.** The model is trained on historical campaign data. If client behaviour or macroeconomic conditions shift, performance should be re-evaluated on fresh data.

---

## 11. Summary Dashboard

![Overall Model Evaluation Summary Dashboard](https://d2z0o16i8xm8ak.cloudfront.net/cc6e5c3c-0103-492d-ba09-47d97013c4b5/87399464-94bf-4c4d-a4a1-29929a448fcd/results/figures/summary_dashboard.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kMnowbzE2aTh4bThhay5jbG91ZGZyb250Lm5ldC9jYzZlNWMzYy0wMTAzLTQ5MmQtYmEwOS00N2Q5NzAxM2M0YjUvODczOTk0NjQtOTRiZi00YzRkLWE0YTEtMjk5MjlhNDQ4ZmNkL3Jlc3VsdHMvZmlndXJlcy9zdW1tYXJ5X2Rhc2hib2FyZC5wbmc~KiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3Njc5MjIwOX19fV19&Signature=XYvmcn1CAv9x7VYEEZuCqH1rEg7l4rSYwuzt40os441D71bnWqFk6u0oH1OSQ56W394--MPgkXmlk1pymZySiC6LMy~lhljAQjIo4oVrtK-ur2UbGuzMwZ5hqGWmRfglEdPCGP5JoXV~z0khTzEAIh6bXTWon9m12fuerLMxNUGsB8cG5DUOZDjvf3Ct-~yHQQwe5OahZn9IRa-uWnzeASINSd5JptiSIeIk~8oiggrAyGcuOzXDvcxH7lsKGuF6kY0WlFNrnlxcV4DNrRMjKAgD0eqTVhaodogjA4lxt8Lz-csZwbW2tPAhVhUGBDKLDghtwMSVa90HDoCCGwgLZQ__&Key-Pair-Id=K1BF7XGXAIMYNX)

---

## 12. Conclusion

Eight machine learning models were evaluated under four feature scenarios and two SMOTE conditions, yielding 64 configurations in total. The key findings are:

- **`Duration` causes 16–36% AUC inflation** and must be excluded from any production model. SVM (RBF) is the most severely affected, with AUC dropping from 0.724 to 0.533 when duration is removed.
- **XGBoost and Random Forest are the top-performing deployable models** (AUC 0.794 and 0.791, respectively), substantially outperforming linear baselines and KNN in the realistic S2 scenario.
- **Previous campaign outcome (`poutcome_success`) is the strongest pre-call predictor**, followed by contact type, client age, days since last contact, and campaign month.
- **SMOTE trades AUC and precision for recall.** It is beneficial when the operational goal is maximising subscriber capture regardless of call volume; it is counterproductive when the goal is efficient targeting under budget constraints.
- **Threshold optimisation substantially improves F1.** Moving from the default 0.50 threshold to the F1-optimal 0.196 threshold raises XGBoost S2 F1 from 0.340 to 0.486.
- **The recommended top-20% call list captures 60.2% of all subscribers** at approximately 3× the rate of random dialling, providing a clear and measurable business case for model deployment.

The analysis demonstrates that meaningful predictive performance is achievable without relying on the leaky `duration` feature, and that tree-based ensemble methods — particularly XGBoost — are well suited to the bank telemarketing prediction task under realistic deployment constraints.

---

*Report prepared by Member 4 — Team 10, Econ 7970 Applied Predictive Modeling. Data sources: UCI Bank Marketing Dataset (bank-full.csv). Models and results validated against Member 2 (`baseline_results_original_features.csv`) and Member 3 (`advanced_results_S*.csv`, `advanced_smote_results_S*.csv`) outputs.*
