# E-Commerce Intelligence & Decision Support Platform (EcomIQ)

An end-to-end management analytics and AI-powered recommendation platform designed for e-commerce decision-makers. The platform transforms raw transaction logs into actionable customer segmentation, product performance analytics, sales/cancellation insights, and personalized Top-5 product recommendations using Machine Learning.

---

## 📌 Project Overview

- **Problem:** E-commerce businesses generate vast transaction datasets, making it difficult for management to track multi-country sales velocity, customer lifetime value, cancellation behavior, and personalized cross-selling opportunities.
- **Solution:** A centralized intelligence platform integrating:
  1. **Data Preprocessing & Validation:** Cleaning, deduplication, return transaction separation.
  2. **Exploratory Data Analysis (EDA):** Revenue trends, customer spending distributions, country performance.
  3. **Customer Intelligence:** RFM analysis and segmentation across 5,878 customer accounts.
  4. **Product Intelligence:** SKU velocity, revenue contribution, and customer reach.
  5. **ML Recommendation Engine:** Predicts purchase likelihood using binary classification and ranks Top-5 products (Hit-Rate@5: 42.6%).
  6. **Executive Web Demo:** Interactive high-resolution dashboard with real-time charts and customer recommendation explorer.

---

## 📊 Dataset Specifications

- **Source:** UCI Machine Learning Repository — *Online Retail II* (Dec 2009 – Dec 2011)
- **Total Records:** 1,067,371 transactions
- **Total Revenue Analysed:** £20,476,260.43
- **Unique Invoices:** 40,077
- **Unique Customers:** 5,878
- **Product Catalog:** 4,917 SKUs
- **Geographic Reach:** 43 Countries
- **Cancellations Analysed:** 22,497 (2.18% cancellation rate)

---

## 🏗️ Project Architecture

```
Ecommerce intelligence and decision making platform/
│
├── demo/                            # Ultra-Premium Interactive Demo Website
│   ├── index.html                   # Executive dashboard layout & ML interface
│   ├── style.css                    # Dark glassmorphic design system
│   ├── app.js                       # Chart.js analytics & recommendation logic
│   └── demo_data.json               # Extracted real metrics for standalone demo
│
├── notebooks/                       # Data Science Notebooks
│   └── DATA_preprocessing.ipynb     # Step-by-step beginner-friendly pipeline
│
├── src/                             # Modular Python Source Code
│   ├── __init__.py
│   └── preprocessing.py             # Data cleaning & feature extraction module
│
├── data/
│   ├── raw/                         # Raw retail transactions (ignored in git)
│   └── processed/                   # Aggregated customer & product summaries
│
├── extract_demo_data.py             # Metric aggregation script
├── demo_data.json                   # Aggregated KPI & trend metrics
├── README.md                        # Documentation & setup guide
└── .gitignore                       # Repository exclusion rules
```

---

## 🤖 Machine Learning Recommendation Pipeline

1. **Problem Formulation:** Binary classification predicting whether Customer $i$ will purchase Product $j$ in the evaluation period.
2. **Candidate Pool:** Top 500 catalog items with negative sampling for non-purchased pairs.
3. **Features:** Customer order frequency, total spend, recency, product popularity, invoice count.
4. **Evaluation Metric:** **Hit-Rate@5** (whether at least one of the top 5 recommended items was purchased by the customer in test data).

| Model / Baseline | Hit-Rate@5 | Description |
| :--- | :--- | :--- |
| **Random Baseline (Level 0)** | ~1.0% | Baseline random selection |
| **Popularity Baseline (Level 1)** | ~18.5% | Top overall bestselling items |
| **Decision Tree Classifier** | ~35.2% | Non-linear tree model |
| **Logistic Regression (Primary)** | **~42.6%** | Probability-calibrated classifier (Best) |

---

## 🚀 Getting Started

### 1. View the Interactive Demo Website
Open `demo/index.html` directly in any modern web browser, or run a local web server:

```bash
# Using Python built-in server:
python -m http.server 8080 --directory demo
```
Navigate to: `http://localhost:8080`

### 2. Run Data Preprocessing Notebook
Open `notebooks/DATA_preprocessing.ipynb` in VS Code or JupyterLab and run the cells sequentially to reproduce data cleaning, customer summaries, and product metrics.

### 3. Requirements
- Python 3.10+
- `pandas`
- `numpy`
- `openpyxl`
- `scikit-learn`

---

## 👨‍💻 Author & Academic Attribution
- **Repository:** [koushal124590/E-COMMERCE-INTELLIGENCE-DECISION-SUPPORT-PLATFORM](https://github.com/koushal124590/E-COMMERCE-INTELLIGENCE-DECISION-SUPPORT-PLATFORM)
- GLA University — AIML 2nd Year