# 🧠 DQ Intelligence System  
### Rule-Based, Human-in-the-Loop Data Quality Assessment, Cleaning & Scoring

---

## 🚀 Overview

This project implements an **end-to-end Data Quality (DQ) Intelligence Framework** that goes beyond traditional profiling tools by focusing on **what happens after data quality issues are detected**.

Instead of stopping at diagnostics, the system:

- Detects and explains data quality issues  
- Recommends **explicit, rule-based fixes**  
- Applies **controlled, user-approved cleaning actions**  
- Logs every intervention for auditability  
- Re-profiles the data after cleaning  
- Quantifies improvement using a **Data Quality (DQ) Score**

The framework is designed for **ML-ready pipelines**, where data quality must be **measurable, explainable, and attributable**.

---

## 🎯 Core Philosophy

### Data Quality ≠ Preprocessing

This system distinguishes between:

- **Intrinsic data quality issues**  
  (missing values, invalid entries, rare categories, duplicates, rule violations)

- **Model-dependent preprocessing concerns**  
  (e.g. skewness, transformations)

Only intrinsic data quality issues contribute to the **final DQ score**, while model-specific properties are used **only to guide cleaning decisions**.

---

## 🧱 System Architecture

Raw Data
│
▼
Data Profiling & Meta Metrics
(generate_meta)
│
▼
Cleaning Recommendations
(cleaning_recommendations)
│
▼
Rule-Based Cleaning (Logged)
(get_cleaned_data)
│
▼
Re-Profiling (Post-Cleaning)
(generate_meta)
│
▼
DQ Feature Table
(get_table_for_DQ_computation)
│
▼
DQ Scoring
(Compute_DQ_Score)


Each stage is modular, inspectable, and reproducible.


## 🔍 Data Quality Dimensions Evaluated

| Dimension | Description |
|---------|------------|
| Completeness | Missing count, missing percentage |
| Integrity | Semantic type errors, rule violations |
| Uniqueness | Duplicate rows |
| Consistency | Whitespace issues, formatting errors |
| Accuracy | Statistical outliers |
| Variety | Rare categorical values |
| Temporal validity | Date range anomalies |
| Stability / Drift | Jensen–Shannon divergence, PSI |
| Information richness | Entropy (used diagnostically) |

> **Note:** Skewness is treated as a *model-specific property*, not a data quality defect, and is excluded from the final DQ score.

---

## 🧾 Rule-Based Cleaning System

All cleaning actions are triggered using **stable `rule_id`s**, not string matching.  
This enables deterministic execution and precise attribution.

| Rule ID | Cleaning Action |
|-------|----------------|
| 1 | Drop column due to high missingness |
| 2 | Impute missing values |
| 3 | Handle outliers |
| 4 | Strip leading/trailing whitespace |
| 5 | Combine rare categories / resolve typos |
| 6 | Detect invalid or constant date columns |
| 7 | Resolve semantic type inconsistencies |
| 8 | Enforce business rules |
| 9 | Power transformation for high skewness |
| 11 | Remove duplicate rows (dataset-level) |

Cleaning is **recommended automatically** but **applied only with user approval** in interactive mode.

---

## 🔁 Human-in-the-Loop Design

The framework supports two execution modes:

- **Automated mode**  
  All recommended rules are applied deterministically.

- **Interactive mode**  
  The user explicitly approves or rejects each recommended fix.

This ensures:
- transparency
- trust
- safety for sensitive datasets

---

## 📈 Data Quality Scoring Engine

The framework computes a **dataset-level DQ score (0–100)** using the following approach:

1. **Metric Normalization**  
   Each DQ metric is mapped to a `[0, 1]` quality score using metric-specific penalty functions.

2. **Metric-Specific Semantics**  
   - Distribution drift (JS, PSI) uses percentile-based penalties  
   - Structural issues use threshold-based penalties  
   - Non-applicable metrics are treated as non-penalizing

3. **Weighted Aggregation**  
   Users can assign weights to DQ dimensions based on their priorities  
   (default: equal weights).

4. **Column → Dataset Aggregation**  
   Column-level scores are averaged to produce a dataset-level DQ score.

This makes quality improvement **quantifiable, comparable, and explainable**.

---

## 📥 Pipeline Outputs

The pipeline returns:

| Output | Description |
|------|------------|
| Meta tables | Column-wise DQ metrics (before & after cleaning) |
| Recommendations | Suggested rules per column |
| Cleaned datasets | User-approved cleaned data |
| Change log | Rule-wise affected rows |
| DQ scores | Before vs after quality scores |
| Health summaries | Dataset-level quality overview |

---

## 🧪 Datasets Evaluated

| Dataset | Domain |
|--------|--------|
| Adult Census | Workforce Analytics |
| NYC Yellow Taxi | Transportation |
| UCI Credit Card | Finance |
| Customer Churn | Telecom |
| CC GENERAL | Customer Segmentation / Banking |

The same pipeline is applied across datasets for fair comparison.

---

## 🔍 How This Is Different

Most data quality tools stop at **telling you what is wrong**.

This framework focuses on the harder problem:

- What should be fixed?
- Why should it be fixed?
- What exactly changed?
- Did data quality *actually* improve?

It bridges the gap between **diagnosis and action**, while preserving auditability and control.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-Learn  
- NLTK  
- Statistical metrics: JS divergence, PSI, Entropy  

---

## 📌 Project Status

- Modular pipeline implemented  
- Notebook & programmatic usage supported  
- Designed for extension into:
  - ML performance attribution
  - automated monitoring
  - production pipelines

---

## 🧠 Key Takeaway

This project treats **data quality as a system**, not a checklist.

Quality is:
- measurable  
- attributable  
- explainable  
- and improvable  

rather than assumed.

---


