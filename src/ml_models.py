"""
Machine Learning Model Development
E-Commerce Intelligence & Decision Support Platform

Purpose:
--------
Train ML models to generate PERSONALIZED product recommendations.
This goes beyond the baselines (Random & Popularity) by learning
patterns from each customer's behaviour.

Models:
  1. Logistic Regression - Simple, interpretable, fast
  2. Decision Tree       - Non-linear patterns, easy to explain

Approach:
  - Frame recommendation as a binary classification problem:
    "Will customer X buy product Y in the future?"
  - Create (customer, product) pairs with features from HISTORICAL data
  - Labels come from EVALUATION data (did they actually buy it?)
  - Split eligible customers into TRAIN (70%) and TEST (30%)
  - Score all candidate products per test customer, recommend Top-5
  - Evaluate with Hit-Rate@5 (same metric as baselines)

Data Leakage Prevention:
  - ALL features computed from historical period only (Dec 2009 - May 2011)
  - Labels from evaluation period (Jun - Dec 2011)
  - Train/test split by CUSTOMER (not by row) to prevent leakage

Author: ML Model Pipeline
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ==============================================================
# CONFIGURATION
# ==============================================================

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models", "trained_models")

HISTORICAL_FILE = os.path.join(PROCESSED_DIR, "historical_transactions.csv")
EVALUATION_FILE = os.path.join(PROCESSED_DIR, "evaluation_transactions.csv")
CUSTOMER_FEATURES_FILE = os.path.join(PROCESSED_DIR, "customer_features.csv")
PRODUCT_FEATURES_FILE = os.path.join(PROCESSED_DIR, "product_features.csv")
CP_FEATURES_FILE = os.path.join(PROCESSED_DIR, "customer_product_features.csv")

TOP_K = 5           # Number of products to recommend
POOL_SIZE = 500     # Product pool (same as baselines for fair comparison)
TEST_SIZE = 0.3     # 30% of customers for testing
RANDOM_SEED = 42    # Reproducibility

# Feature columns to use from each table
CUSTOMER_NUMERIC_FEATURES = [
    "total_orders",
    "total_spending",
    "total_quantity",
    "unique_products",
    "avg_order_value",
    "recency_days",
    "tenure_days",
    "avg_days_between_orders",
]

PRODUCT_NUMERIC_FEATURES = [
    "total_quantity_sold",
    "total_revenue",
    "total_orders",
    "unique_customers",
    "avg_unit_price",
    "popularity_score",
]

INTERACTION_FEATURES = [
    "purchase_count",
    "total_qty_purchased",
    "total_spent_on_product",
    "days_since_last_purchase",
    "has_previous_purchase",
]


# ==============================================================
# STEP 1: Load All Data
# ==============================================================

def load_all_data():
    """
    Load the feature tables and transaction files.
    All features were computed from historical data only (by features.py).
    """
    print("=" * 60)
    print("STEP 1: Loading Feature Data")
    print("=" * 60)

    customer_features = pd.read_csv(CUSTOMER_FEATURES_FILE)
    print(f"  Customer features : {customer_features.shape}")

    product_features = pd.read_csv(PRODUCT_FEATURES_FILE)
    print(f"  Product features  : {product_features.shape}")

    cp_features = pd.read_csv(CP_FEATURES_FILE)
    print(f"  Cust-Prod features: {cp_features.shape}")

    historical = pd.read_csv(
        HISTORICAL_FILE,
        dtype={"InvoiceNo": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    evaluation = pd.read_csv(
        EVALUATION_FILE,
        dtype={"InvoiceNo": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    print(f"  Historical trans  : {historical.shape}")
    print(f"  Evaluation trans  : {evaluation.shape}")

    return customer_features, product_features, cp_features, historical, evaluation


# ==============================================================
# STEP 2: Identify Eligible Customers & Product Pool
# ==============================================================

def setup_eligible_pool(historical, evaluation):
    """
    Eligible customers = appeared in BOTH historical and evaluation.
    Product pool = top 500 products by historical popularity.
    (Same as baselines so comparison is fair.)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Setting Up Eligible Customers & Product Pool")
    print("=" * 60)

    hist_custs = set(historical["CustomerID"].unique())
    eval_custs = set(evaluation["CustomerID"].unique())
    eligible = sorted(hist_custs & eval_custs)

    # Top 500 products by historical order count
    product_pop = (
        historical.groupby("StockCode")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )
    product_pool = product_pop.head(POOL_SIZE).index.tolist()

    print(f"  Eligible customers : {len(eligible):,}")
    print(f"  Product pool       : {len(product_pool)} products")

    return eligible, product_pool


# ==============================================================
# STEP 3: Build the Training Dataset
# ==============================================================

def build_ml_dataset(
    eligible, product_pool, customer_features, product_features,
    cp_features, evaluation
):
    """
    Create labeled (customer, product) pairs for ML training.

    For each eligible customer and each product in the pool:
      - Features = customer features + product features + interaction features
      - Label = 1 if customer bought the product in evaluation, 0 otherwise

    This creates a table where each row is a (customer, product) pair
    with all relevant features and a binary target.

    KEY: Features come from HISTORICAL data, labels from EVALUATION data.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building ML Dataset (Feature Matrix + Labels)")
    print("=" * 60)

    t0 = time.time()

    # ── 3a. Create all (customer, product) candidate pairs ──
    # This is like saying: "For each customer, consider every product
    # in the pool and ask: will they buy it?"
    print("  Creating candidate pairs...")
    pairs = pd.DataFrame(
        [(c, p) for c in eligible for p in product_pool],
        columns=["CustomerID", "StockCode"],
    )
    print(f"    Total candidate pairs: {len(pairs):,}")

    # ── 3b. Create labels from evaluation data ──
    # Label = 1 if the customer actually bought this product in Jun-Dec 2011
    print("  Creating labels from evaluation purchases...")
    eval_purchases = (
        evaluation[evaluation["CustomerID"].isin(eligible)]
        .drop_duplicates(subset=["CustomerID", "StockCode"])[["CustomerID", "StockCode"]]
        .copy()
    )
    eval_purchases["purchased"] = 1

    pairs = pairs.merge(
        eval_purchases, on=["CustomerID", "StockCode"], how="left"
    )
    pairs["purchased"] = pairs["purchased"].fillna(0).astype(int)

    positives = pairs["purchased"].sum()
    negatives = len(pairs) - positives
    print(f"    Positives (bought)    : {positives:,} ({positives/len(pairs)*100:.2f}%)")
    print(f"    Negatives (not bought): {negatives:,} ({negatives/len(pairs)*100:.2f}%)")

    # ── 3c. Merge customer features ──
    # Each customer gets their historical behaviour profile
    print("  Merging customer features...")
    cust_cols = ["CustomerID"] + CUSTOMER_NUMERIC_FEATURES
    pairs = pairs.merge(
        customer_features[cust_cols], on="CustomerID", how="left"
    )

    # ── 3d. Merge product features ──
    # Each product gets its historical performance profile
    print("  Merging product features...")

    # Rename product feature columns to avoid clashes with customer features
    prod_renamed = product_features.copy()
    rename_map = {}
    for col in PRODUCT_NUMERIC_FEATURES:
        rename_map[col] = "prod_" + col
    prod_renamed = prod_renamed.rename(columns=rename_map)

    prod_cols = ["StockCode"] + ["prod_" + c for c in PRODUCT_NUMERIC_FEATURES]
    pairs = pairs.merge(
        prod_renamed[prod_cols], on="StockCode", how="left"
    )

    # ── 3e. Merge customer-product interaction features ──
    # If a customer has previously bought this product, these features
    # capture that history. If not, they'll be 0 (no prior interaction).
    print("  Merging customer-product interaction features...")
    cp_cols = ["CustomerID", "StockCode"] + INTERACTION_FEATURES
    pairs = pairs.merge(
        cp_features[cp_cols], on=["CustomerID", "StockCode"], how="left"
    )

    # Fill missing interaction features with 0
    # (No prior interaction = all zeros, which is correct)
    for col in INTERACTION_FEATURES:
        pairs[col] = pairs[col].fillna(0)

    # Fill any remaining NaNs in customer/product features
    # (avg_days_between_orders can be NaN for single-order customers)
    pairs = pairs.fillna(0)

    elapsed = round(time.time() - t0, 1)
    print(f"\n  Dataset built in {elapsed}s")
    print(f"  Final shape: {pairs.shape}")
    print(f"  Feature columns: {len(pairs.columns) - 3} "
          f"(excluding CustomerID, StockCode, purchased)")

    return pairs


# ==============================================================
# STEP 4: Train/Test Split by Customer
# ==============================================================

def split_by_customer(dataset, eligible):
    """
    Split the dataset by CUSTOMER, not by row.

    Why? If we split by row, some rows from the same customer could
    end up in both train and test, which would leak information.
    By splitting by customer, the model has never seen any data
    from test customers during training.

    Train customers (70%): Used to train the ML models
    Test customers  (30%): Used to evaluate Hit-Rate@5
    """
    print("\n" + "=" * 60)
    print("STEP 4: Train/Test Split (by Customer)")
    print("=" * 60)

    # Split customer IDs into train and test groups
    train_customers, test_customers = train_test_split(
        eligible,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # Split the dataset based on customer groups
    train_data = dataset[dataset["CustomerID"].isin(train_customers)].copy()
    test_data = dataset[dataset["CustomerID"].isin(test_customers)].copy()

    print(f"  Train customers: {len(train_customers):,} "
          f"({len(train_customers)/len(eligible)*100:.0f}%)")
    print(f"  Test customers : {len(test_customers):,} "
          f"({len(test_customers)/len(eligible)*100:.0f}%)")
    print(f"  Train samples  : {len(train_data):,} "
          f"(positives: {train_data['purchased'].sum():,})")
    print(f"  Test samples   : {len(test_data):,} "
          f"(positives: {test_data['purchased'].sum():,})")

    return train_data, test_data, train_customers, test_customers


# ==============================================================
# STEP 5: Prepare Features for Training
# ==============================================================

def prepare_features(train_data, test_data):
    """
    Prepare the feature matrix (X) and labels (y) for training.

    Steps:
      1. Separate features from labels and identifiers.
      2. Scale numeric features using StandardScaler.
         - Why? Logistic Regression is sensitive to feature scales.
           A feature ranging 0-100 would dominate one ranging 0-1.
         - StandardScaler transforms each feature to mean=0, std=1.
      3. The scaler is fit ONLY on training data to prevent leakage.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Preparing Features (Scaling)")
    print("=" * 60)

    # Columns that are NOT features (identifiers + target)
    non_feature_cols = ["CustomerID", "StockCode", "purchased"]

    # All remaining columns are features
    feature_cols = [c for c in train_data.columns if c not in non_feature_cols]

    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")

    # Separate features and labels
    X_train = train_data[feature_cols].values
    y_train = train_data["purchased"].values
    X_test = test_data[feature_cols].values
    y_test = test_data["purchased"].values

    # Scale features
    # IMPORTANT: fit on training data ONLY, then transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled = scaler.transform(X_test)          # transform only (no fit!)

    print(f"\n  X_train shape: {X_train_scaled.shape}")
    print(f"  X_test shape : {X_test_scaled.shape}")
    print(f"  y_train: {y_train.sum():,} positives / "
          f"{len(y_train):,} total ({y_train.mean()*100:.2f}%)")
    print(f"  y_test : {y_test.sum():,} positives / "
          f"{len(y_test):,} total ({y_test.mean()*100:.2f}%)")

    return X_train_scaled, y_train, X_test_scaled, y_test, feature_cols, scaler


# ==============================================================
# STEP 6: Train Logistic Regression
# ==============================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train a Logistic Regression model.

    Why Logistic Regression?
      - Simple and interpretable
      - Outputs probabilities (not just 0/1), which we need for ranking
      - Handles imbalanced classes with class_weight='balanced'
      - Fast to train even on 900K+ samples

    Parameters:
      - class_weight='balanced': Automatically adjusts weights to handle
        the imbalanced dataset (~4% positive). Without this, the model
        would just predict 0 for everything and get 96% accuracy!
      - max_iter=1000: Allow enough iterations to converge
      - C=1.0: Default regularization strength
    """
    print("\n" + "=" * 60)
    print("STEP 6: Training Logistic Regression")
    print("=" * 60)

    t0 = time.time()

    lr_model = LogisticRegression(
        class_weight="balanced",  # Handle class imbalance
        max_iter=1000,            # Enough iterations to converge
        C=1.0,                    # Default regularization
        random_state=RANDOM_SEED,
        solver="lbfgs",           # Good for medium-sized datasets
        n_jobs=-1,                # Use all CPU cores
    )

    print("  Training Logistic Regression...")
    lr_model.fit(X_train, y_train)

    elapsed = round(time.time() - t0, 1)
    print(f"  Training completed in {elapsed}s")

    # Predictions
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Classification metrics
    print(f"\n  --- Classification Metrics (Test Set) ---")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

    return lr_model, y_prob


# ==============================================================
# STEP 7: Train Decision Tree
# ==============================================================

def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train a Decision Tree classifier.

    Why Decision Tree?
      - Captures non-linear patterns that Logistic Regression misses
      - Very interpretable (you can visualize the tree)
      - No feature scaling needed (but we use scaled data for consistency)
      - Good comparison point against Logistic Regression

    Parameters:
      - max_depth=10: Limits tree depth to prevent overfitting.
        Without a limit, the tree would memorize the training data.
      - min_samples_leaf=50: Each leaf needs at least 50 samples.
        Prevents the tree from creating tiny, noisy leaves.
      - class_weight='balanced': Handle class imbalance (same as LR)
    """
    print("\n" + "=" * 60)
    print("STEP 7: Training Decision Tree")
    print("=" * 60)

    t0 = time.time()

    dt_model = DecisionTreeClassifier(
        max_depth=10,               # Prevent overfitting
        min_samples_leaf=50,        # Minimum samples per leaf
        class_weight="balanced",    # Handle class imbalance
        random_state=RANDOM_SEED,
    )

    print("  Training Decision Tree...")
    dt_model.fit(X_train, y_train)

    elapsed = round(time.time() - t0, 1)
    print(f"  Training completed in {elapsed}s")
    print(f"  Tree depth: {dt_model.get_depth()}")
    print(f"  Number of leaves: {dt_model.get_n_leaves()}")

    # Predictions
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1]

    # Classification metrics
    print(f"\n  --- Classification Metrics (Test Set) ---")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

    return dt_model, y_prob


# ==============================================================
# STEP 8: Generate Top-5 Recommendations
# ==============================================================

def generate_recommendations(
    test_data, lr_probs, dt_probs, test_customers, evaluation
):
    """
    Use model probabilities to create personalized recommendations.

    For each test customer:
      1. Get the predicted purchase probability for each product.
      2. Sort products by probability (highest first).
      3. Pick the Top-5 products as recommendations.

    This is the key difference from baselines:
      - Random baseline: picks randomly (no intelligence)
      - Popularity baseline: same 5 products for everyone (no personalization)
      - ML models: different products for each customer based on THEIR behaviour!
    """
    print("\n" + "=" * 60)
    print("STEP 8: Generating Top-5 Personalized Recommendations")
    print("=" * 60)

    # Add probabilities to test data
    test_df = test_data[["CustomerID", "StockCode", "purchased"]].copy()
    test_df["lr_probability"] = lr_probs
    test_df["dt_probability"] = dt_probs

    # Build ground truth for test customers
    eval_test = evaluation[evaluation["CustomerID"].isin(test_customers)]
    ground_truth = (
        eval_test.groupby("CustomerID")["StockCode"]
        .apply(set)
        .to_dict()
    )

    # Generate recommendations for each model
    results = {}

    for model_name, prob_col in [
        ("Logistic Regression", "lr_probability"),
        ("Decision Tree", "dt_probability"),
    ]:
        print(f"\n  --- {model_name} ---")
        recommendations = {}
        rec_rows = []

        for customer_id in test_customers:
            # Get this customer's product scores
            cust_mask = test_df["CustomerID"] == customer_id
            cust_scores = test_df[cust_mask].sort_values(
                prob_col, ascending=False
            )

            # Pick top-5 products with highest predicted probability
            top_5 = cust_scores.head(TOP_K)
            top_5_products = top_5["StockCode"].tolist()
            top_5_probs = top_5[prob_col].tolist()

            recommendations[customer_id] = top_5_products

            # Save detailed results
            actual = ground_truth.get(customer_id, set())
            for rank, (prod, prob) in enumerate(
                zip(top_5_products, top_5_probs), 1
            ):
                rec_rows.append({
                    "CustomerID": int(customer_id),
                    "rank": rank,
                    "recommended_product": prod,
                    "probability": round(prob, 6),
                    "was_hit": 1 if prod in actual else 0,
                })

        # Calculate Hit-Rate@5
        hits = 0
        total = 0
        for cust_id, recs in recommendations.items():
            actual = ground_truth.get(cust_id, set())
            if not actual:
                continue
            total += 1
            if set(recs) & actual:
                hits += 1

        hit_rate = hits / total if total > 0 else 0
        print(f"  Customers evaluated : {total:,}")
        print(f"  Hits (>= 1 correct) : {hits:,}")
        print(f"  Hit-Rate@{TOP_K}         : {hit_rate:.4f} ({hit_rate*100:.2f}%)")

        rec_df = pd.DataFrame(rec_rows)
        results[model_name] = {
            "recommendations": recommendations,
            "hit_rate": hit_rate,
            "hits": hits,
            "total": total,
            "rec_df": rec_df,
        }

    return results, test_df


# ==============================================================
# STEP 9: Save Models & Results
# ==============================================================

def save_outputs(
    lr_model, dt_model, scaler, feature_cols, results, test_df
):
    """
    Save trained models, predictions, and recommendations.
    """
    print("\n" + "=" * 60)
    print("STEP 9: Saving Models & Results")
    print("=" * 60)

    # ── Save Models ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    lr_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
    joblib.dump(lr_model, lr_path)
    print(f"  [OK] Saved: {lr_path}")

    dt_path = os.path.join(MODEL_DIR, "decision_tree.joblib")
    joblib.dump(dt_model, dt_path)
    print(f"  [OK] Saved: {dt_path}")

    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  [OK] Saved: {scaler_path}")

    # Save feature column names for future reference
    feat_path = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"  [OK] Saved: {feat_path}")

    # ── Save Predictions ──
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    pred_path = os.path.join(PROCESSED_DIR, "ml_predictions.csv")
    test_df.to_csv(pred_path, index=False)
    size_mb = os.path.getsize(pred_path) / (1024 * 1024)
    print(f"  [OK] Saved: {pred_path} ({len(test_df):,} rows, {size_mb:.1f} MB)")

    # ── Save Recommendations ──
    for model_name, data in results.items():
        safe_name = model_name.lower().replace(" ", "_")
        rec_path = os.path.join(PROCESSED_DIR, f"{safe_name}_recommendations.csv")
        data["rec_df"].to_csv(rec_path, index=False)
        print(f"  [OK] Saved: {rec_path} ({len(data['rec_df']):,} rows)")

    # ── Save Metrics Summary ──
    metrics_rows = []
    for model_name, data in results.items():
        metrics_rows.append({
            "model": model_name,
            "hit_rate_at_5": round(data["hit_rate"], 4),
            "hits": data["hits"],
            "total_customers": data["total"],
            "hit_rate_pct": round(data["hit_rate"] * 100, 2),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(PROCESSED_DIR, "ml_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  [OK] Saved: {metrics_path}")


# ==============================================================
# STEP 10: Validate Outputs
# ==============================================================

def validate_outputs(lr_model, dt_model, results, test_customers, feature_cols):
    """
    Run sanity checks on model outputs.
    """
    print("\n" + "=" * 60)
    print("STEP 10: Validating Model Outputs")
    print("=" * 60)

    errors = []

    # 10a. Models are fitted
    has_coef = hasattr(lr_model, "coef_")
    has_tree = hasattr(dt_model, "tree_")
    print(f"  [a] LR model fitted  : {'PASS' if has_coef else 'FAIL'}")
    print(f"      DT model fitted  : {'PASS' if has_tree else 'FAIL'}")
    if not has_coef:
        errors.append("LR model not fitted!")
    if not has_tree:
        errors.append("DT model not fitted!")

    # 10b. Feature count matches
    lr_features = lr_model.coef_.shape[1]
    print(f"  [b] LR expects {lr_features} features, "
          f"we have {len(feature_cols)}: "
          f"{'PASS' if lr_features == len(feature_cols) else 'FAIL'}")
    if lr_features != len(feature_cols):
        errors.append(f"Feature mismatch: LR={lr_features}, actual={len(feature_cols)}")

    # 10c. Each test customer got exactly TOP_K recommendations
    for model_name, data in results.items():
        wrong = sum(
            1 for recs in data["recommendations"].values()
            if len(recs) != TOP_K
        )
        print(f"  [c] {model_name}: all customers got {TOP_K} recs: "
              f"{'PASS' if wrong == 0 else 'FAIL'}")
        if wrong > 0:
            errors.append(f"{model_name}: {wrong} customers got wrong # of recs")

    # 10d. No duplicate recommendations per customer
    for model_name, data in results.items():
        dups = sum(
            1 for recs in data["recommendations"].values()
            if len(recs) != len(set(recs))
        )
        print(f"  [d] {model_name}: no duplicate recs: "
              f"{'PASS' if dups == 0 else 'FAIL'}")
        if dups > 0:
            errors.append(f"{model_name}: {dups} customers have duplicate recs")

    # 10e. Hit rates are valid
    for model_name, data in results.items():
        valid = 0 <= data["hit_rate"] <= 1
        print(f"  [e] {model_name}: hit rate in [0,1]: "
              f"{'PASS' if valid else 'FAIL'}")
        if not valid:
            errors.append(f"{model_name}: hit rate out of range")

    # 10f. Saved model files exist
    for fname in ["logistic_regression.joblib", "decision_tree.joblib",
                   "feature_scaler.joblib", "feature_columns.txt"]:
        path = os.path.join(MODEL_DIR, fname)
        exists = os.path.isfile(path)
        print(f"  [f] File exists: {fname}: {'PASS' if exists else 'FAIL'}")
        if not exists:
            errors.append(f"Missing file: {fname}")

    # Summary
    print("\n" + "-" * 60)
    if errors:
        print("  VALIDATION FAILED:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  ALL VALIDATIONS PASSED")
    print("-" * 60)

    return errors


# ==============================================================
# STEP 11: Final Comparison (All Models vs Baselines)
# ==============================================================

def print_final_comparison(results):
    """
    Compare ML models against the baselines.
    Load baseline metrics for a complete picture.
    """
    print("\n" + "=" * 60)
    print("STEP 11: Final Comparison - All Models")
    print("=" * 60)

    # Load baseline metrics
    baseline_path = os.path.join(PROCESSED_DIR, "baseline_metrics.csv")
    if os.path.exists(baseline_path):
        baselines = pd.read_csv(baseline_path)
    else:
        baselines = pd.DataFrame()

    print("\n  +--------------------------------------+----------+")
    print("  | Model                                | Hit@5    |")
    print("  +--------------------------------------+----------+")

    # Baselines
    if len(baselines) > 0:
        for _, row in baselines.iterrows():
            name = row["baseline"]
            hr = row["hit_rate_pct"]
            print(f"  | {name:<36} | {hr:>5.2f}%  |")

    # ML Models
    for model_name, data in results.items():
        hr = data["hit_rate"] * 100
        print(f"  | {model_name:<36} | {hr:>5.2f}%  |")

    print("  +--------------------------------------+----------+")

    # Note about comparison fairness
    print("\n  NOTE: Baselines were evaluated on ALL 2,602 eligible customers.")
    print("  ML models are evaluated on 30% test customers only (~781).")
    print("  This is standard practice - ML models need a train/test split.")


# ==============================================================
# MAIN PIPELINE
# ==============================================================

def run_ml_pipeline():
    """
    Run the complete ML model development pipeline.
    """
    print("\n+" + "=" * 58 + "+")
    print("|  ML MODEL DEVELOPMENT                                    |")
    print("|  E-Commerce Intelligence & Decision Support Platform     |")
    print("+" + "=" * 58 + "+\n")

    t_start = time.time()

    # Step 1: Load data
    customer_features, product_features, cp_features, historical, evaluation = (
        load_all_data()
    )

    # Step 2: Setup eligible customers & product pool
    eligible, product_pool = setup_eligible_pool(historical, evaluation)

    # Step 3: Build ML dataset
    dataset = build_ml_dataset(
        eligible, product_pool,
        customer_features, product_features, cp_features,
        evaluation,
    )

    # Step 4: Train/test split by customer
    train_data, test_data, train_custs, test_custs = split_by_customer(
        dataset, eligible
    )

    # Step 5: Prepare features
    X_train, y_train, X_test, y_test, feature_cols, scaler = prepare_features(
        train_data, test_data
    )

    # Step 6: Train Logistic Regression
    lr_model, lr_probs = train_logistic_regression(X_train, y_train, X_test, y_test)

    # Step 7: Train Decision Tree
    dt_model, dt_probs = train_decision_tree(X_train, y_train, X_test, y_test)

    # Step 8: Generate recommendations
    results, test_df = generate_recommendations(
        test_data, lr_probs, dt_probs, test_custs, evaluation
    )

    # Step 9: Save models & results
    save_outputs(lr_model, dt_model, scaler, feature_cols, results, test_df)

    # Step 10: Validate
    errors = validate_outputs(lr_model, dt_model, results, test_custs, feature_cols)

    # Step 11: Final comparison
    print_final_comparison(results)

    elapsed = round(time.time() - t_start, 1)
    print(f"\nPipeline completed in {elapsed}s")

    return {
        "lr_model": lr_model,
        "dt_model": dt_model,
        "results": results,
        "validation_errors": errors,
    }


# ==============================================================
# Run when executed directly
# ==============================================================
if __name__ == "__main__":
    run_ml_pipeline()

