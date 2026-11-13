"""
Baseline Recommendation System
E-Commerce Intelligence & Decision Support Platform

Purpose:
--------
Implement two simple recommendation baselines to establish performance
benchmarks BEFORE building any ML models.

  Level 0 - Random Baseline    : Randomly recommend 5 products
  Level 1 - Popularity Baseline: Recommend the 5 most popular products

Metric:
  Hit-Rate@5 = % of customers who received at least 1 correct
               recommendation (i.e., a product they actually bought
               in the evaluation period).

Data Leakage Prevention:
  - Product popularity is calculated ONLY from historical data
    (Dec 2009 - May 2011).
  - Evaluation uses ONLY June - December 2011 transactions.

Author: Baseline Recommendation Pipeline
"""

import os
import time
import pandas as pd
import numpy as np


# ==============================================================
# CONFIGURATION
# ==============================================================

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("outputs", "figures")

HISTORICAL_FILE = os.path.join(PROCESSED_DIR, "historical_transactions.csv")
EVALUATION_FILE = os.path.join(PROCESSED_DIR, "evaluation_transactions.csv")

# Recommendation settings
TOP_K = 5             # Number of products to recommend
POOL_SIZE = 500       # Product pool size for random baseline
RANDOM_SEED = 42      # For reproducibility


# ==============================================================
# STEP 1: Load Data
# ==============================================================

def load_data():
    """
    Load the historical and evaluation transaction files.
    These were created by the feature engineering pipeline (features.py)
    and are already split at June 1, 2011.
    """
    print("=" * 60)
    print("STEP 1: Loading Historical & Evaluation Data")
    print("=" * 60)

    historical = pd.read_csv(
        HISTORICAL_FILE,
        dtype={"InvoiceNo": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    print(f"  Historical : {len(historical):,} rows, "
          f"{historical['CustomerID'].nunique():,} customers, "
          f"{historical['StockCode'].nunique():,} products")
    print(f"  Date range : {historical['InvoiceDate'].min()} to "
          f"{historical['InvoiceDate'].max()}")

    evaluation = pd.read_csv(
        EVALUATION_FILE,
        dtype={"InvoiceNo": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
        low_memory=False,
    )
    print(f"\n  Evaluation : {len(evaluation):,} rows, "
          f"{evaluation['CustomerID'].nunique():,} customers, "
          f"{evaluation['StockCode'].nunique():,} products")
    print(f"  Date range : {evaluation['InvoiceDate'].min()} to "
          f"{evaluation['InvoiceDate'].max()}")

    return historical, evaluation


# ==============================================================
# STEP 2: Identify Eligible Customers
# ==============================================================

def get_eligible_customers(historical, evaluation):
    """
    Eligible customers = customers who appear in BOTH historical AND
    evaluation periods.

    Why? We can only fairly evaluate a recommendation system on
    customers we have history for AND who actually purchased something
    in the evaluation period (so we have ground truth to check against).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Identifying Eligible Customers")
    print("=" * 60)

    hist_customers = set(historical["CustomerID"].unique())
    eval_customers = set(evaluation["CustomerID"].unique())

    # Customers in BOTH periods
    eligible_customers = sorted(hist_customers & eval_customers)

    print(f"  Customers in historical only : {len(hist_customers - eval_customers):,}")
    print(f"  Customers in evaluation only : {len(eval_customers - hist_customers):,}")
    print(f"  Customers in BOTH (eligible) : {len(eligible_customers):,}")

    return eligible_customers


# ==============================================================
# STEP 3: Build Ground Truth
# ==============================================================

def build_ground_truth(evaluation, eligible_customers):
    """
    For each eligible customer, find the set of products they
    ACTUALLY purchased in the evaluation period (Jun-Dec 2011).

    This is our "answer key" - what a perfect recommender would predict.

    Returns:
        dict: {CustomerID: set(StockCodes they bought)}
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building Ground Truth (Evaluation Purchases)")
    print("=" * 60)

    # Filter evaluation data to eligible customers only
    eval_eligible = evaluation[
        evaluation["CustomerID"].isin(eligible_customers)
    ]

    # For each customer, collect the set of products they bought
    ground_truth = (
        eval_eligible
        .groupby("CustomerID")["StockCode"]
        .apply(set)
        .to_dict()
    )

    # Statistics
    total_products = sum(len(v) for v in ground_truth.values())
    avg_products = total_products / len(ground_truth) if ground_truth else 0

    print(f"  Eligible customers with eval purchases : {len(ground_truth):,}")
    print(f"  Average products per customer in eval  : {avg_products:.1f}")
    print(f"  Total product purchases to predict     : {total_products:,}")

    return ground_truth


# ==============================================================
# STEP 4: Calculate Hit-Rate@K
# ==============================================================

def calculate_hit_rate(recommendations, ground_truth, k=TOP_K):
    """
    Hit-Rate@K measures: "For what fraction of customers did we
    recommend at least 1 product they actually bought?"

    Formula:
        Hit-Rate@K = (Number of customers with >= 1 hit) / (Total customers)

    A "hit" means one of our K recommended products is in the
    customer's actual evaluation purchases.

    Args:
        recommendations: dict {CustomerID: [list of recommended StockCodes]}
        ground_truth:    dict {CustomerID: set(actual StockCodes bought)}
        k:               Number of recommendations (for logging)

    Returns:
        float: Hit-rate between 0.0 and 1.0
    """
    hits = 0
    total = 0

    for customer_id, recommended_products in recommendations.items():
        # What did this customer actually buy?
        actual_products = ground_truth.get(customer_id, set())

        if not actual_products:
            continue  # Skip customers with no eval purchases

        total += 1

        # Check if ANY of our recommendations match an actual purchase
        recommended_set = set(recommended_products[:k])
        if recommended_set & actual_products:  # Set intersection
            hits += 1

    hit_rate = hits / total if total > 0 else 0.0
    return hit_rate, hits, total


# ==============================================================
# STEP 5: Level 0 - Random Baseline
# ==============================================================

def random_baseline(historical, eligible_customers, ground_truth):
    """
    LEVEL 0 - Random Baseline
    -------------------------
    Strategy: For each customer, randomly pick 5 products from a pool
    of 500 eligible products.

    This is the simplest possible "recommender" - it has no intelligence
    at all. Any real recommendation system should beat this easily.

    The product pool = top 500 products by historical order count
    (so we don't recommend obscure products nobody has heard of).
    """
    print("\n" + "=" * 60)
    print("STEP 5: Level 0 - Random Baseline")
    print("=" * 60)

    # Build product pool: top 500 products by number of orders in historical data
    product_popularity = (
        historical
        .groupby("StockCode")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
    )
    product_pool = product_popularity.head(POOL_SIZE).index.tolist()
    print(f"  Product pool size: {len(product_pool)} products")
    print(f"  Pool covers orders from {product_popularity.iloc[POOL_SIZE-1]} "
          f"to {product_popularity.iloc[0]} historical orders")

    # Set random seed for reproducibility
    rng = np.random.RandomState(RANDOM_SEED)

    # For each eligible customer, randomly pick 5 products from the pool
    recommendations = {}
    for customer_id in eligible_customers:
        # Randomly sample 5 products (without replacement)
        random_picks = rng.choice(product_pool, size=TOP_K, replace=False)
        recommendations[customer_id] = list(random_picks)

    print(f"  Generated random recommendations for {len(recommendations):,} customers")

    # Calculate Hit-Rate@5
    hit_rate, hits, total = calculate_hit_rate(recommendations, ground_truth)

    print(f"\n  --- Random Baseline Results ---")
    print(f"  Customers evaluated : {total:,}")
    print(f"  Hits (>= 1 correct) : {hits:,}")
    print(f"  Hit-Rate@{TOP_K}         : {hit_rate:.4f} ({hit_rate*100:.2f}%)")

    return recommendations, hit_rate, hits, total


# ==============================================================
# STEP 6: Level 1 - Popularity Baseline
# ==============================================================

def popularity_baseline(historical, eligible_customers, ground_truth):
    """
    LEVEL 1 - Popularity Baseline
    -----------------------------
    Strategy: Find the 5 most popular products from HISTORICAL data,
    then recommend those same 5 products to EVERY customer.

    This is smarter than random because popular products are popular
    for a reason - many people buy them, so there's a good chance
    any given customer will buy one too.

    Popularity = number of distinct invoices (orders) containing
    the product in the historical period.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Level 1 - Popularity Baseline")
    print("=" * 60)

    # Calculate product popularity from HISTORICAL data only
    product_popularity = (
        historical
        .groupby("StockCode")
        .agg(
            order_count=("InvoiceNo", "nunique"),
            description=("Description", lambda s: s.mode().iloc[0] if len(s) > 0 else "Unknown"),
        )
        .sort_values("order_count", ascending=False)
    )

    # Pick top 5 most popular products
    top_5 = product_popularity.head(TOP_K)

    print(f"  Top {TOP_K} most popular products (historical):")
    print(f"  {'Rank':<5} {'StockCode':<10} {'Orders':<8} {'Description'}")
    print(f"  {'-'*4:<5} {'-'*9:<10} {'-'*6:<8} {'-'*30}")
    for rank, (code, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {rank:<5} {code:<10} {row['order_count']:<8} {row['description']}")

    top_5_products = top_5.index.tolist()

    # Recommend the SAME top-5 products to every eligible customer
    recommendations = {}
    for customer_id in eligible_customers:
        recommendations[customer_id] = top_5_products.copy()

    print(f"\n  Recommended same {TOP_K} products to all "
          f"{len(recommendations):,} eligible customers")

    # Calculate Hit-Rate@5
    hit_rate, hits, total = calculate_hit_rate(recommendations, ground_truth)

    print(f"\n  --- Popularity Baseline Results ---")
    print(f"  Customers evaluated : {total:,}")
    print(f"  Hits (>= 1 correct) : {hits:,}")
    print(f"  Hit-Rate@{TOP_K}         : {hit_rate:.4f} ({hit_rate*100:.2f}%)")

    return recommendations, hit_rate, hits, total, top_5_products


# ==============================================================
# STEP 7: Compare & Save Results
# ==============================================================

def compare_and_save(
    random_recs, random_hit_rate, random_hits, random_total,
    pop_recs, pop_hit_rate, pop_hits, pop_total,
    top_5_products, historical, ground_truth,
):
    """
    Compare the two baselines side by side and save all outputs.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Comparison & Saving Results")
    print("=" * 60)

    # ── Comparison Table ──
    print("\n  +------------------------------+----------+------------+")
    print("  | Baseline                     | Hit-Rate | Hits/Total |")
    print("  +------------------------------+----------+------------+")
    print(f"  | Level 0 - Random             |  {random_hit_rate*100:5.2f}%  | "
          f"{random_hits:>4}/{random_total:<5} |")
    print(f"  | Level 1 - Popularity         |  {pop_hit_rate*100:5.2f}%  | "
          f"{pop_hits:>4}/{pop_total:<5} |")
    print("  +------------------------------+----------+------------+")

    improvement = ((pop_hit_rate - random_hit_rate) / random_hit_rate * 100
                   if random_hit_rate > 0 else float('inf'))
    print(f"\n  Popularity vs Random improvement: {improvement:+.1f}%")

    # ── Save Evaluation Metrics ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_df = pd.DataFrame([
        {
            "baseline": "Level 0 - Random",
            "hit_rate_at_5": round(random_hit_rate, 4),
            "hits": random_hits,
            "total_customers": random_total,
            "hit_rate_pct": round(random_hit_rate * 100, 2),
        },
        {
            "baseline": "Level 1 - Popularity",
            "hit_rate_at_5": round(pop_hit_rate, 4),
            "hits": pop_hits,
            "total_customers": pop_total,
            "hit_rate_pct": round(pop_hit_rate * 100, 2),
        },
    ])

    metrics_path = os.path.join(OUTPUT_DIR, "baseline_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  [OK] Saved: {metrics_path}")

    # ── Save Random Recommendations (sample) ──
    random_rows = []
    for cust_id, prods in random_recs.items():
        for rank, prod in enumerate(prods, 1):
            actual = ground_truth.get(cust_id, set())
            random_rows.append({
                "CustomerID": int(cust_id),
                "rank": rank,
                "recommended_product": prod,
                "was_hit": 1 if prod in actual else 0,
            })
    random_df = pd.DataFrame(random_rows)
    random_path = os.path.join(OUTPUT_DIR, "random_recommendations.csv")
    random_df.to_csv(random_path, index=False)
    print(f"  [OK] Saved: {random_path} ({len(random_df):,} rows)")

    # ── Save Popularity Recommendations ──
    pop_rows = []
    for cust_id, prods in pop_recs.items():
        for rank, prod in enumerate(prods, 1):
            actual = ground_truth.get(cust_id, set())
            pop_rows.append({
                "CustomerID": int(cust_id),
                "rank": rank,
                "recommended_product": prod,
                "was_hit": 1 if prod in actual else 0,
            })
    pop_df = pd.DataFrame(pop_rows)
    pop_path = os.path.join(OUTPUT_DIR, "popularity_recommendations.csv")
    pop_df.to_csv(pop_path, index=False)
    print(f"  [OK] Saved: {pop_path} ({len(pop_df):,} rows)")

    # ── Save Top-5 Popular Products Reference ──
    top5_info = []
    for rank, code in enumerate(top_5_products, 1):
        desc = historical[historical["StockCode"] == code]["Description"].mode()
        desc = desc.iloc[0] if len(desc) > 0 else "Unknown"
        orders = historical[historical["StockCode"] == code]["InvoiceNo"].nunique()
        top5_info.append({
            "rank": rank,
            "StockCode": code,
            "description": desc,
            "historical_orders": orders,
        })
    top5_df = pd.DataFrame(top5_info)
    top5_path = os.path.join(OUTPUT_DIR, "top5_popular_products.csv")
    top5_df.to_csv(top5_path, index=False)
    print(f"  [OK] Saved: {top5_path}")

    return metrics_df


# ==============================================================
# STEP 8: Validate Results
# ==============================================================

def validate_results(
    random_recs, pop_recs, ground_truth, historical, evaluation,
    random_hit_rate, pop_hit_rate,
):
    """
    Run sanity checks on the baseline results.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Validating Results")
    print("=" * 60)

    errors = []

    # 8a. Every customer got exactly TOP_K recommendations
    for cust_id, prods in random_recs.items():
        if len(prods) != TOP_K:
            errors.append(f"Random: Customer {cust_id} got {len(prods)} recs, expected {TOP_K}")
            break
    for cust_id, prods in pop_recs.items():
        if len(prods) != TOP_K:
            errors.append(f"Popularity: Customer {cust_id} got {len(prods)} recs, expected {TOP_K}")
            break
    print(f"  [a] All customers got exactly {TOP_K} recommendations: "
          f"{'PASS' if not errors else 'FAIL'}")

    # 8b. Popularity recommendations are the same for every customer
    first_pop_rec = list(pop_recs.values())[0]
    all_same = all(prods == first_pop_rec for prods in pop_recs.values())
    print(f"  [b] Popularity recs identical for all customers: "
          f"{'PASS' if all_same else 'FAIL'}")
    if not all_same:
        errors.append("Popularity recommendations differ across customers!")

    # 8c. Hit rates are in valid range [0, 1]
    valid_range = 0 <= random_hit_rate <= 1 and 0 <= pop_hit_rate <= 1
    print(f"  [c] Hit rates in [0, 1] range: {'PASS' if valid_range else 'FAIL'}")
    if not valid_range:
        errors.append("Hit rate out of [0,1] range!")

    # 8d. Popularity should beat random (expected, not guaranteed, but very likely)
    print(f"  [d] Popularity > Random (expected): "
          f"{'PASS' if pop_hit_rate > random_hit_rate else 'WARNING'}")

    # 8e. Data leakage check - all recommended products exist in historical
    hist_products = set(historical["StockCode"].unique())
    for prods in random_recs.values():
        for p in prods:
            if p not in hist_products:
                errors.append(f"Random rec product {p} not in historical data!")
                break
    for p in first_pop_rec:
        if p not in hist_products:
            errors.append(f"Popular product {p} not in historical data!")
    print(f"  [e] All recommended products from historical: "
          f"{'PASS' if not any('not in historical' in e for e in errors) else 'FAIL'}")

    # 8f. No duplicate recommendations per customer
    random_dups = sum(1 for prods in random_recs.values() if len(prods) != len(set(prods)))
    pop_dups = sum(1 for prods in pop_recs.values() if len(prods) != len(set(prods)))
    print(f"  [f] No duplicate recs per customer: "
          f"{'PASS' if random_dups == 0 and pop_dups == 0 else 'FAIL'}")
    if random_dups > 0:
        errors.append(f"{random_dups} customers have duplicate random recs")

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
# MAIN PIPELINE
# ==============================================================

def run_baseline_pipeline():
    """
    Run the complete baseline recommendation pipeline.
    """
    print("\n" + "+" + "=" * 58 + "+")
    print("|  BASELINE RECOMMENDATION SYSTEM                          |")
    print("|  E-Commerce Intelligence & Decision Support Platform     |")
    print("+" + "=" * 58 + "+\n")

    t_start = time.time()

    # Step 1: Load data
    historical, evaluation = load_data()

    # Step 2: Identify eligible customers
    eligible_customers = get_eligible_customers(historical, evaluation)

    # Step 3: Build ground truth from evaluation data
    ground_truth = build_ground_truth(evaluation, eligible_customers)

    # Step 5: Level 0 - Random baseline
    random_recs, random_hr, random_hits, random_total = random_baseline(
        historical, eligible_customers, ground_truth
    )

    # Step 6: Level 1 - Popularity baseline
    pop_recs, pop_hr, pop_hits, pop_total, top_5 = popularity_baseline(
        historical, eligible_customers, ground_truth
    )

    # Step 7: Compare & save
    metrics = compare_and_save(
        random_recs, random_hr, random_hits, random_total,
        pop_recs, pop_hr, pop_hits, pop_total,
        top_5, historical, ground_truth,
    )

    # Step 8: Validate
    errors = validate_results(
        random_recs, pop_recs, ground_truth, historical, evaluation,
        random_hr, pop_hr,
    )

    elapsed = round(time.time() - t_start, 1)
    print(f"\nPipeline completed in {elapsed}s")

    return {
        "metrics": metrics,
        "random_hit_rate": random_hr,
        "popularity_hit_rate": pop_hr,
        "validation_errors": errors,
    }


# ==============================================================
# Run when executed directly
# ==============================================================
if __name__ == "__main__":
    results = run_baseline_pipeline()

