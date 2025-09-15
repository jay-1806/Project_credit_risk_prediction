# Assignment 1 — Credit Card Fraud Detection (CRISP-DM)

**Author:** Jay Vithlani  
**Medium article:** <https://medium.com/@jvithlani4/imbalanced-learning-that-pays-calibrated-credit-card-fraud-detection-with-decision-curves-f74faae666b2>  
**Transcript (PDF):** `https://docs.google.com/document/d/1wguY-80IXSL_9RUXMq0ZHGF7GAkaG_cXG0N4_TpDH7Y/edit?usp=sharing`  



> This repo contains all artifacts for my Assignment-1. The project follows **CRISP-DM** end-to-end and focuses on **calibrated, cost-aware** fraud detection on the ULB/Worldline dataset.

---

## 0) Contents
- [1) Problem & Goal](#1-problem--goal)
- [2) Dataset](#2-dataset)
- [3) Method (CRISP-DM)](#3-method-crispdm)
- [4) Key Results](#4-key-results)
- [5) Figures](#5-figures)
- [6) Tables / CSV Artifacts](#6-tables--csv-artifacts)
- [7) Reproduce / Regenerate](#7-reproduce--regenerate)
- [8) Folder Structure](#8-folder-structure)
- [9) Notes & Acknowledgments](#9-notes--acknowledgments)

---

## 1) Problem & Goal
Detect likely fraudulent transactions under a **limited review budget** to maximize **fraud \$ saved** and minimize analyst workload.  
Metrics emphasized: **PR-AUC**, **Precision/Recall @k** (alert budgets), **Brier score** (calibration), and **Net Savings** via a **decision curve**.

## 2) Dataset
- **ULB/Worldline Credit Card Fraud** (Kaggle mirror).  
- ~284,807 txns over ~2 days; 492 frauds (~0.17%).  
- Features `V1–V28` (PCA-anonymized), `Time`, `Amount`; label `Class` (1 = fraud).

> ⚠️ **Data file not included** (size/license). Download from Kaggle and place locally when regenerating.

## 3) Method (CRISP-DM)
- **Business Understanding:** triage with cost/benefit; **Net Savings** used for threshold selection.  
- **Data Understanding:** extreme imbalance; heavy-tailed `Amount`.  
- **Preparation:** de-dupe; time-aware split (60/20/20); `LogAmount`, `hour_sin/cos`; train-only scaling; median imputation; drop zero-variance cols.  
- **Modeling:** `LogisticRegression(class_weight="balanced")`; compared to RUS and dummy.  
- **Calibration:** **Isotonic** + **Platt** (manual) → better **Brier** and reliable scores.  
- **Thresholding:** sweep **decision curve** on validation at \$5/alert → lock threshold → evaluate on test.  
- **Bonus:** feature selection (univariate AUC + L1 + RF consensus) and **KMeans clustering** personas.

## 4) Key Results (Test)
- **LR_isotonic:** PR-AUC ≈ **0.765**, ROC-AUC ≈ **0.980**, Brier ≈ **0.000443**  
- **Final policy (threshold from validation, \$5/alert):**  
  - Threshold (isotonic scale) ≈ **0.01143**  
  - Alerts = **349**, TP=64, FP=285, FN=10, TN=56,387  
  - Precision ≈ **18.3%**, Recall ≈ **86.5%**, **Amount-Recall ≈ 69.6%**  
  - **Net Savings ≈ \$3,632.78** (2-day test window)

See: `assets/tables/metrics_calibrated_test.csv` & `final_policy_test.csv`.

## 5) Figures
> Open images in `assets/figures/` or view directly below (GitHub displays PNGs):

- Validation PR: `pr_curve_baselines.png`  
- Validation ROC: `roc_curve_baselines.png`  
- Calibration (test): `calibration_curves_test.png`  
- Decision curve (validation): `decision_curve_val_isotonic.png`  
- Test PR: `pr_roc_test_pr.png`  
- Test ROC: `pr_roc_test_roc.png`  
- Clusters — fraud rate (val): `cluster_fraudrate_val.png`  
- Clusters — PCA 2D (val): `cluster_pca2_val.png`  
- EDA — V14 density: `class_conditional_V14.png`  
- (Plus: class imbalance, amount hist, time hist, fraud rate by hour, correlation heatmap if available.)

## 6) Tables / CSV Artifacts
- Validation metrics (baselines): `metrics_summary_val.csv`  
- Calibration comparisons (test): `metrics_calibrated_test.csv`  
- Final policy (test): `final_policy_test.csv`  
- Alert budgets @k (val/test): `topline_at_k_val.csv`, `topline_at_k_test.csv`  
- Decision curve points (val, isotonic): `val_cost_curve_isotonic.csv`  
- Feature ranking consensus: `feature_rank_consensus.csv`  
- Subset eval (LR on top-K): `subset_eval_lr_val.csv`  
- Cluster profiles: `cluster_profiles.csv`

## 7) Reproduce / Regenerate
**Colab (recommended):**
1. Open `scripts/generate_missing_figures.py` in Colab.  
2. Upload `creditcard.csv` (Kaggle) to the Colab working directory.  
3. Run the script → it writes the PNGs/CSVs to your current dir.

**Environment notes:** scikit-learn ≥ 1.6, NumPy ≥ 2.0. We use **median imputation**, drop zero-variance features, and **manual calibration** (Isotonic + 1D Platt).

## 8) Folder Structure
assignment-1/
README.md
assets/
figures/ # all .png figures
tables/ # all .csv metrics/tables
transcript/
chatgpt_transcript.pdf
medium/
link.txt # paste your Medium article URL
scripts/
generate_missing_figures.py
notebooks/


## 9) Notes & Acknowledgments
- Dataset: ULB/Worldline Credit Card Fraud (Kaggle mirror).  
- Thanks to course staff & classmates for feedback.  
- Contact: <jvithlani4@gmail.com>


