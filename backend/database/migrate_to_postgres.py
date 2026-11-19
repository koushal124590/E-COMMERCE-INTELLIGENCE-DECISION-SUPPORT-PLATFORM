"""
PostgreSQL Migration & Data Ingestion Script
E-Commerce Intelligence & Decision Support Platform

Usage:
------
1. Ensure your PostgreSQL server is running and database exists:
     CREATE DATABASE ecommerce_db;
2. Set your DATABASE_URL in .env:
     DATABASE_URL=postgresql://postgres:your_password@localhost:5432/ecommerce_db
3. Run this script:
     python backend/database/migrate_to_postgres.py

What it does:
-------------
- Creates tables: customers, products, recommendations
- Ingests data from data/processed/*.csv files
- Verifies row counts and constraints
"""

import os
import sys
import time
import pandas as pd
from sqlalchemy import text

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.database.config import get_engine, check_db_connection, DATABASE_URL
from backend.database.models import Base, Customer, Product, Recommendation
from backend.database.config import get_session_factory

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def run_migration():
    print("=" * 65)
    print("  POSTGRESQL MIGRATION & DATA INGESTION PIPELINE")
    print("=" * 65)
    print(f"Target Database URL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")

    # Step 1: Check Connection
    print("\n[1/4] Checking PostgreSQL connection...")
    if not check_db_connection():
        print("\n  [ERROR] Cannot connect to PostgreSQL!")
        print("  Please verify:")
        print("    1. PostgreSQL server is running (locally or on cloud like Neon/Supabase/RDS).")
        print("    2. DATABASE_URL in .env has correct user, password, host, port, and database name.")
        print(f"    Current URL: {DATABASE_URL}\n")
        return False

    engine = get_engine()
    print("  [OK] Successfully connected to PostgreSQL!")

    # Step 2: Create Tables
    print("\n[2/4] Creating database tables (customers, products, recommendations)...")
    Base.metadata.create_all(bind=engine)
    print("  [OK] Schema created successfully.")

    SessionFactory = get_session_factory()
    session = SessionFactory()

    t0 = time.time()

    try:
        # Step 3a: Ingest Customers
        print("\n[3/4] Ingesting data from data/processed/...")
        cust_csv = os.path.join(DATA_DIR, "customer_features.csv")
        if os.path.exists(cust_csv):
            print(f"  Reading {cust_csv}...")
            df_cust = pd.read_csv(cust_csv)

            # Assign customer segments
            def get_segment(row):
                spend = row.get("total_spending", 0)
                recency = row.get("recency_days", 999)
                if spend >= 2000 and recency <= 60:
                    return "VIP Champion (High Spend, High Activity)"
                elif spend >= 1000:
                    return "Loyal Customer (Steady Spender)"
                elif recency > 180:
                    return "At-Risk / Lapsed Customer"
                return "Regular Shopper"

            df_cust["segment"] = df_cust.apply(get_segment, axis=1)

            # Clear existing
            session.execute(text("DELETE FROM recommendations;"))
            session.execute(text("DELETE FROM customers;"))
            session.commit()

            customers_to_insert = []
            for _, row in df_cust.iterrows():
                customers_to_insert.append(Customer(
                    customer_id=int(row["CustomerID"]),
                    country=str(row.get("country", "Unknown")),
                    total_spending=float(row.get("total_spending", 0.0)),
                    total_orders=int(row.get("total_orders", 0)),
                    total_quantity=int(row.get("total_quantity", 0)),
                    unique_products=int(row.get("unique_products", 0)),
                    avg_order_value=float(row.get("avg_order_value", 0.0)),
                    recency_days=int(row.get("recency_days", 0)),
                    tenure_days=int(row.get("tenure_days", 0)),
                    segment=str(row.get("segment", "Regular Shopper")),
                    first_purchase=str(row.get("first_purchase", "")),
                    last_purchase=str(row.get("last_purchase", ""))
                ))
            session.bulk_save_objects(customers_to_insert)
            session.commit()
            print(f"  -> Inserted {len(customers_to_insert):,} rows into 'customers'.")

        # Step 3b: Ingest Products
        prod_csv = os.path.join(DATA_DIR, "product_features.csv")
        if os.path.exists(prod_csv):
            print(f"  Reading {prod_csv}...")
            df_prod = pd.read_csv(prod_csv)

            session.execute(text("DELETE FROM products;"))
            session.commit()

            products_to_insert = []
            for _, row in df_prod.iterrows():
                products_to_insert.append(Product(
                    stock_code=str(row["StockCode"]),
                    description=str(row.get("description", "Unknown Product")),
                    total_quantity_sold=int(row.get("total_quantity_sold", 0)),
                    total_revenue=float(row.get("total_revenue", 0.0)),
                    total_orders=int(row.get("total_orders", 0)),
                    unique_customers=int(row.get("unique_customers", 0)),
                    avg_unit_price=float(row.get("avg_unit_price", 0.0)),
                    popularity_score=float(row.get("popularity_score", 0.0))
                ))
            session.bulk_save_objects(products_to_insert)
            session.commit()
            print(f"  -> Inserted {len(products_to_insert):,} rows into 'products'.")

        # Step 3c: Ingest Recommendations
        recs_to_insert = []

        # Decision Tree recommendations
        dt_csv = os.path.join(DATA_DIR, "decision_tree_recommendations.csv")
        if os.path.exists(dt_csv):
            df_dt = pd.read_csv(dt_csv)
            for _, row in df_dt.iterrows():
                recs_to_insert.append(Recommendation(
                    customer_id=int(row["CustomerID"]),
                    stock_code=str(row["recommended_product"]),
                    rank=int(row["rank"]),
                    model_name="decision_tree",
                    probability=float(row.get("probability", 1.0)),
                    was_hit=bool(row.get("was_hit", 0) == 1)
                ))

        # Logistic Regression recommendations
        lr_csv = os.path.join(DATA_DIR, "logistic_regression_recommendations.csv")
        if os.path.exists(lr_csv):
            df_lr = pd.read_csv(lr_csv)
            for _, row in df_lr.iterrows():
                recs_to_insert.append(Recommendation(
                    customer_id=int(row["CustomerID"]),
                    stock_code=str(row["recommended_product"]),
                    rank=int(row["rank"]),
                    model_name="logistic_regression",
                    probability=float(row.get("probability", 1.0)),
                    was_hit=bool(row.get("was_hit", 0) == 1)
                ))

        if recs_to_insert:
            session.bulk_save_objects(recs_to_insert)
            session.commit()
            print(f"  -> Inserted {len(recs_to_insert):,} rows into 'recommendations'.")

        # Step 4: Verification
        print("\n[4/4] Verifying record counts in PostgreSQL...")
        c_count = session.query(Customer).count()
        p_count = session.query(Product).count()
        r_count = session.query(Recommendation).count()

        print(f"  'customers' table       : {c_count:,} rows")
        print(f"  'products' table        : {p_count:,} rows")
        print(f"  'recommendations' table : {r_count:,} rows")

        elapsed = round(time.time() - t0, 2)
        print(f"\n[SUCCESS] Migration completed in {elapsed}s!")
        return True

    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Migration failed: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    run_migration()
