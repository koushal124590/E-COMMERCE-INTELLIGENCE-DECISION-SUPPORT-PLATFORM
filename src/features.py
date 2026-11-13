"""
Feature Engineering Module
E-Commerce Intelligence & Decision Support Platform

Purpose:
--------
Generate ML-ready features from the cleaned transaction data.
All features are computed ONLY from historical data (Dec 2009 – May 2011)
to prevent data leakage into the evaluation period (Jun – Dec 2011).

Feature Groups:
  1. Customer Features    → customer_features.csv
  2. Product Features     → product_features.csv
  3. Customer-Product     → customer_product_features.csv

Author: Feature Engineering Pipeline
"""

import os
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Temporal split boundary
SPLIT_DATE = pd.Timestamp("2011-06-01")

# Paths
PROCESSED_DIR = os.path.join("data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_transactions.csv")


# ──────────────────────────────────────────────
# STEP 1: Load & Split Data
# ──────────────────────────────────────────────

def load_and_split(filepath: str = INPUT_FILE):
    """
    Load the cleaned transactions CSV and split it into:
      - historical : Dec 2009 – May 2011  (for building features)
      - evaluation : Jun 2011 – Dec 2011  (held out, no peeking!)

    We also drop rows where CustomerID is missing because
    customer-level features require a valid customer identifier.
    """
    print("=" * 60)
    print("STEP 1: Loading & Splitting Data")
    print("=" * 60)

    # Load with explicit types to avoid mixed-type warnings
    df = pd.read_csv(
        filepath,
        dtype={"InvoiceNo": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} rows from {filepath}")

    # Keep only rows with a valid CustomerID
    before = len(df)
    df = df[df["CustomerID"].notnull()].copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    print(f"  Dropped {before - len(df):,} rows with missing CustomerID")
    print(f"  Remaining: {len(df):,} rows")

    # ── Temporal split ──
    historical = df[df["InvoiceDate"] < SPLIT_DATE].copy()
    evaluation = df[df["InvoiceDate"] >= SPLIT_DATE].copy()

    print(f"\n  Historical period : {historical['InvoiceDate'].min()} → "
          f"{historical['InvoiceDate'].max()}")
    print(f"    Rows            : {len(historical):,}")
    print(f"    Unique customers: {historical['CustomerID'].nunique():,}")
    print(f"    Unique products : {historical['StockCode'].nunique():,}")

    print(f"\n  Evaluation period : {evaluation['InvoiceDate'].min()} → "
          f"{evaluation['InvoiceDate'].max()}")
    print(f"    Rows            : {len(evaluation):,}")
    print(f"    Unique customers: {evaluation['CustomerID'].nunique():,}")

    return historical, evaluation


# ──────────────────────────────────────────────
# STEP 2: Customer Features
# ──────────────────────────────────────────────

def build_customer_features(historical: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer in the HISTORICAL period, compute:

      - total_orders       : Number of distinct invoices
      - total_spending     : Sum of TotalAmount (Quantity × UnitPrice)
      - avg_order_value    : total_spending / total_orders
      - total_quantity     : Total items purchased
      - unique_products    : How many different products they bought
      - first_purchase     : Date of earliest transaction
      - last_purchase      : Date of most recent transaction
      - recency_days       : Days between last purchase and the split date
      - tenure_days        : Days between first purchase and the split date
      - avg_days_between   : Average gap (in days) between consecutive orders
      - country            : Most frequent country for the customer
    """
    print("\n" + "=" * 60)
    print("STEP 2: Building Customer Features (from historical only)")
    print("=" * 60)

    # ── Basic aggregations ──
    customer_agg = historical.groupby("CustomerID").agg(
        total_orders=("InvoiceNo", "nunique"),
        total_spending=("TotalAmount", "sum"),
        total_quantity=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        country=("Country", lambda s: s.mode().iloc[0] if len(s) > 0 else "Unknown"),
    ).reset_index()

    # ── Derived features ──
    # Average order value
    customer_agg["avg_order_value"] = (
        customer_agg["total_spending"] / customer_agg["total_orders"]
    ).round(2)

    # Recency: how many days before the split date was the last purchase?
    customer_agg["recency_days"] = (
        SPLIT_DATE - customer_agg["last_purchase"]
    ).dt.days

    # Tenure: how long has this customer been known?
    customer_agg["tenure_days"] = (
        SPLIT_DATE - customer_agg["first_purchase"]
    ).dt.days

    # Round monetary values
    customer_agg["total_spending"] = customer_agg["total_spending"].round(2)

    # ── Average days between orders ──
    # For customers with only 1 order, this will be NaN (which is correct).
    def calc_avg_gap(group):
        """Calculate average gap between unique order dates."""
        order_dates = group["InvoiceDate"].drop_duplicates().sort_values()
        if len(order_dates) < 2:
            return np.nan
        gaps = order_dates.diff().dropna().dt.days
        return round(gaps.mean(), 1)

    avg_gaps = historical.groupby("CustomerID").apply(
        calc_avg_gap, include_groups=False
    ).reset_index()
    avg_gaps.columns = ["CustomerID", "avg_days_between_orders"]

    # Merge the gap feature in
    customer_agg = customer_agg.merge(avg_gaps, on="CustomerID", how="left")

    print(f"  Generated features for {len(customer_agg):,} customers")
    print(f"  Columns: {list(customer_agg.columns)}")

    return customer_agg


# ──────────────────────────────────────────────
# STEP 3: Product Features
# ──────────────────────────────────────────────

def build_product_features(historical: pd.DataFrame) -> pd.DataFrame:
    """
    For each product (StockCode) in the HISTORICAL period, compute:

      - description        : Most common product name
      - total_quantity_sold : Total units sold
      - total_revenue      : Sum of TotalAmount
      - total_orders       : Number of distinct invoices containing this product
      - unique_customers   : Number of distinct buyers
      - avg_unit_price     : Average selling price
      - popularity_score   : Percentile rank based on total_orders
                             (0 = least popular, 100 = most popular)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building Product Features (from historical only)")
    print("=" * 60)

    product_agg = historical.groupby("StockCode").agg(
        description=(
            "Description",
            lambda s: s.dropna().mode().iloc[0]
            if not s.dropna().empty
            else "Unknown Product",
        ),
        total_quantity_sold=("Quantity", "sum"),
        total_revenue=("TotalAmount", "sum"),
        total_orders=("InvoiceNo", "nunique"),
        unique_customers=("CustomerID", "nunique"),
        avg_unit_price=("UnitPrice", "mean"),
    ).reset_index()

    # Round monetary columns
    product_agg["total_revenue"] = product_agg["total_revenue"].round(2)
    product_agg["avg_unit_price"] = product_agg["avg_unit_price"].round(2)

    # Popularity score: percentile rank of total_orders
    # rank(pct=True) gives a value between 0 and 1; multiply by 100 for readability
    product_agg["popularity_score"] = (
        product_agg["total_orders"].rank(pct=True) * 100
    ).round(1)

    print(f"  Generated features for {len(product_agg):,} products")
    print(f"  Columns: {list(product_agg.columns)}")

    return product_agg


# ──────────────────────────────────────────────
# STEP 4: Customer-Product Interaction Features
# ──────────────────────────────────────────────

def build_customer_product_features(historical: pd.DataFrame) -> pd.DataFrame:
    """
    For each (CustomerID, StockCode) pair in the HISTORICAL period, compute:

      - purchase_count       : How many times the customer bought this product
      - total_qty_purchased  : Total quantity of this product the customer bought
      - total_spent_on_product: Total money spent on this product
      - first_purchase_date  : When the customer first bought this product
      - last_purchase_date   : When the customer last bought this product
      - days_since_last_purchase: Days between last purchase of this product
                                  and the split date
      - has_previous_purchase : 1 if purchase_count > 1, else 0
                                (indicates repeat purchase behaviour)
    """
    print("\n" + "=" * 60)
    print("STEP 4: Building Customer-Product Interaction Features")
    print("=" * 60)

    cp_agg = historical.groupby(["CustomerID", "StockCode"]).agg(
        purchase_count=("InvoiceNo", "nunique"),
        total_qty_purchased=("Quantity", "sum"),
        total_spent_on_product=("TotalAmount", "sum"),
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
    ).reset_index()

    # Days since last purchase of this specific product
    cp_agg["days_since_last_purchase"] = (
        SPLIT_DATE - cp_agg["last_purchase_date"]
    ).dt.days

    # Binary flag: has the customer purchased this product more than once?
    cp_agg["has_previous_purchase"] = (cp_agg["purchase_count"] > 1).astype(int)

    # Round monetary
    cp_agg["total_spent_on_product"] = cp_agg["total_spent_on_product"].round(2)

    print(f"  Generated {len(cp_agg):,} customer-product interaction rows")
    print(f"  Columns: {list(cp_agg.columns)}")

    return cp_agg


# ──────────────────────────────────────────────
# STEP 5: Save Outputs
# ──────────────────────────────────────────────

def save_features(
    customer_features: pd.DataFrame,
    product_features: pd.DataFrame,
    customer_product_features: pd.DataFrame,
    historical: pd.DataFrame,
    evaluation: pd.DataFrame,
    output_dir: str = PROCESSED_DIR,
):
    """
    Save all feature tables and the temporal splits as CSVs.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Saving Feature Files")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    outputs = {
        "customer_features": customer_features,
        "product_features": product_features,
        "customer_product_features": customer_product_features,
        "historical_transactions": historical,
        "evaluation_transactions": evaluation,
    }

    for name, df in outputs.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"  [OK] {name}.csv → {len(df):,} rows, {size_kb:.0f} KB")


# ──────────────────────────────────────────────
# STEP 6: Validate Outputs
# ──────────────────────────────────────────────

def validate_features(
    customer_features: pd.DataFrame,
    product_features: pd.DataFrame,
    customer_product_features: pd.DataFrame,
    historical: pd.DataFrame,
):
    """
    Run sanity checks on the generated feature tables:

      1. No duplicate primary keys
      2. No unexpected nulls in key columns
      3. Monetary totals match the source data
      4. Recency values are non-negative
      5. Counts are positive
    """
    print("\n" + "=" * 60)
    print("STEP 6: Validating Feature Quality")
    print("=" * 60)

    errors = []

    # ── 6a. Customer Features ──
    print("\n  [Customer Features]")

    # Duplicate customer IDs?
    dup_customers = customer_features["CustomerID"].duplicated().sum()
    print(f"    Duplicate CustomerIDs       : {dup_customers}")
    if dup_customers > 0:
        errors.append("Duplicate CustomerIDs found!")

    # Missing values in critical columns
    critical_cols = ["total_orders", "total_spending", "avg_order_value", "recency_days"]
    for col in critical_cols:
        nulls = customer_features[col].isnull().sum()
        print(f"    Nulls in {col:25s}: {nulls}")
        if nulls > 0 and col != "avg_days_between_orders":
            errors.append(f"Nulls in customer_features.{col}")

    # Recency should be >= 0
    neg_recency = (customer_features["recency_days"] < 0).sum()
    print(f"    Negative recency_days       : {neg_recency}")
    if neg_recency > 0:
        errors.append("Negative recency_days found!")

    # Total spending should match historical total
    hist_total = round(historical["TotalAmount"].sum(), 2)
    feat_total = round(customer_features["total_spending"].sum(), 2)
    pct_diff = abs(hist_total - feat_total) / hist_total * 100 if hist_total != 0 else 0
    print(f"    Historical total spending    : {hist_total:,.2f}")
    print(f"    Customer features total      : {feat_total:,.2f}")
    print(f"    Difference                   : {pct_diff:.4f}%")
    if pct_diff > 0.01:
        errors.append(
            f"Spending mismatch: historical={hist_total}, features={feat_total}"
        )

    # ── 6b. Product Features ──
    print("\n  [Product Features]")

    dup_products = product_features["StockCode"].duplicated().sum()
    print(f"    Duplicate StockCodes        : {dup_products}")
    if dup_products > 0:
        errors.append("Duplicate StockCodes found!")

    neg_revenue = (product_features["total_revenue"] < 0).sum()
    print(f"    Negative revenue products   : {neg_revenue}")

    null_desc = product_features["description"].isnull().sum()
    print(f"    Null descriptions           : {null_desc}")

    # popularity_score range check
    pop_min = product_features["popularity_score"].min()
    pop_max = product_features["popularity_score"].max()
    print(f"    Popularity score range      : {pop_min} – {pop_max}")
    if pop_min < 0 or pop_max > 100:
        errors.append("Popularity score out of [0, 100] range!")

    # ── 6c. Customer-Product Features ──
    print("\n  [Customer-Product Features]")

    dup_cp = customer_product_features.duplicated(
        subset=["CustomerID", "StockCode"]
    ).sum()
    print(f"    Duplicate (Cust, Product)   : {dup_cp}")
    if dup_cp > 0:
        errors.append("Duplicate customer-product pairs found!")

    neg_days = (customer_product_features["days_since_last_purchase"] < 0).sum()
    print(f"    Negative days_since_last    : {neg_days}")
    if neg_days > 0:
        errors.append("Negative days_since_last_purchase found!")

    zero_count = (customer_product_features["purchase_count"] <= 0).sum()
    print(f"    Zero/negative purchase_count: {zero_count}")
    if zero_count > 0:
        errors.append("Zero or negative purchase_count found!")

    # ── 6d. Data Leakage Check ──
    print("\n  [Data Leakage Check]")
    max_date_in_features = customer_product_features["last_purchase_date"].max()
    print(f"    Latest date in features     : {max_date_in_features}")
    print(f"    Split date                  : {SPLIT_DATE}")
    if pd.Timestamp(max_date_in_features) >= SPLIT_DATE:
        errors.append("DATA LEAKAGE: features contain data from evaluation period!")
    else:
        print("    [OK] No leakage detected — all feature dates are before the split.")

    # ── Summary ──
    print("\n" + "-" * 60)
    if errors:
        print("  [FAIL] VALIDATION FAILED — Issues found:")
        for e in errors:
            print(f"    • {e}")
    else:
        print("  [OK] ALL VALIDATIONS PASSED")
    print("-" * 60)

    return errors


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run_feature_pipeline():
    """
    Run the complete feature engineering pipeline end-to-end.
    """
    print("\n" + "+" + "=" * 58 + "+")
    print("|  FEATURE ENGINEERING PIPELINE                            |")
    print("|  E-Commerce Intelligence & Decision Support Platform     |")
    print("+" + "=" * 58 + "+\n")

    # Step 1: Load & split
    historical, evaluation = load_and_split()

    # Step 2: Customer features (historical only!)
    customer_features = build_customer_features(historical)

    # Step 3: Product features (historical only!)
    product_features = build_product_features(historical)

    # Step 4: Customer-product interaction features (historical only!)
    customer_product_features = build_customer_product_features(historical)

    # Step 5: Save everything
    save_features(
        customer_features,
        product_features,
        customer_product_features,
        historical,
        evaluation,
    )

    # Step 6: Validate
    errors = validate_features(
        customer_features,
        product_features,
        customer_product_features,
        historical,
    )

    return {
        "customer_features": customer_features,
        "product_features": product_features,
        "customer_product_features": customer_product_features,
        "historical": historical,
        "evaluation": evaluation,
        "validation_errors": errors,
    }


# ──────────────────────────────────────────────
# Run when executed directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    results = run_feature_pipeline()
