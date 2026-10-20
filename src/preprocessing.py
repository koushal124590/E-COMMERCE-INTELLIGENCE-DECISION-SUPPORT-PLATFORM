"""
Data Preprocessing & Cleaning Module
E-Commerce Intelligence & Decision Support Platform
"""

import os
import time
import pandas as pd
import numpy as np


COLUMN_MAPPING = {
    'Invoice': 'InvoiceNo',
    'Price': 'UnitPrice',
    'Customer ID': 'CustomerID'
}


def load_raw_data(file_path: str = os.path.join('data', 'raw', 'online_retail_II.xlsx')) -> pd.DataFrame:
    """
    Step 1: Load both Excel sheets from Online Retail II and combine them.
    Standardizes column names and casts object columns to string to avoid mixed types.
    """
    print(f"[1/5] Loading raw sheets from: {file_path}")
    t0 = time.time()
    
    df_2009 = pd.read_excel(file_path, sheet_name='Year 2009-2010')
    df_2009.rename(columns=COLUMN_MAPPING, inplace=True)
    print(f"  - Sheet 'Year 2009-2010' loaded: {df_2009.shape[0]:,} rows, {df_2009.shape[1]} columns")

    df_2010 = pd.read_excel(file_path, sheet_name='Year 2010-2011')
    df_2010.rename(columns=COLUMN_MAPPING, inplace=True)
    print(f"  - Sheet 'Year 2010-2011' loaded: {df_2010.shape[0]:,} rows, {df_2010.shape[1]} columns")

    df_combined = pd.concat([df_2009, df_2010], ignore_index=True)
    
    # Cast identifiers and descriptions to string to resolve mixed-type issues (e.g. C invoices and numeric stock codes)
    df_combined['InvoiceNo'] = df_combined['InvoiceNo'].astype(str).str.strip()
    df_combined['StockCode'] = df_combined['StockCode'].astype(str).str.strip()
    df_combined['Description'] = df_combined['Description'].astype(str).str.strip()
    df_combined['Country'] = df_combined['Country'].astype(str).str.strip()
    
    # Standardize missing description
    df_combined['Description'] = df_combined['Description'].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    elapsed = round(time.time() - t0, 2)
    print(f"  -> Combined total: {df_combined.shape[0]:,} rows, {df_combined.shape[1]} columns in {elapsed}s")
    return df_combined


def audit_raw_data(df: pd.DataFrame) -> dict:
    """
    Step 2: Inspect data types, nulls, duplicates, and cancellations.
    """
    print("\n[2/5] Auditing Raw Dataset Health...")
    total_rows = len(df)
    null_counts = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())
    cancellation_rows = int(df['InvoiceNo'].str.startswith('C').sum())
    negative_qty = int((df['Quantity'] < 0).sum())
    zero_or_neg_price = int((df['UnitPrice'] <= 0).sum())

    audit_report = {
        'total_rows': total_rows,
        'null_counts': null_counts,
        'duplicate_rows': duplicates,
        'cancellation_invoices': cancellation_rows,
        'negative_quantity_rows': negative_qty,
        'zero_or_negative_price_rows': zero_or_neg_price
    }

    print(f"  - Total Records: {total_rows:,}")
    print(f"  - Duplicate Rows: {duplicates:,}")
    print(f"  - Invoices starting with 'C': {cancellation_rows:,}")
    print(f"  - Negative Quantity rows: {negative_qty:,}")
    print(f"  - Zero or Negative UnitPrice rows: {zero_or_neg_price:,}")
    print(f"  - Missing CustomerID: {null_counts.get('CustomerID', 0):,} ({null_counts.get('CustomerID', 0)/total_rows*100:.2f}%)")
    print(f"  - Missing Description: {null_counts.get('Description', 0):,}")
    return audit_report


def clean_dataset(df: pd.DataFrame):
    """
    Step 3: Clean and partition the dataset.
    - Separates cancellations for dedicated cancellation analytics.
    - Filters valid purchases (Quantity > 0 and UnitPrice > 0).
    - Removes duplicate transaction lines.
    - Formats InvoiceDate to datetime.
    - Computes TotalAmount = Quantity * UnitPrice.
    """
    print("\n[3/5] Cleaning and Partitioning Transactions...")
    
    # 1. Deduplicate
    df_dedup = df.drop_duplicates().copy()
    print(f"  - Dropped duplicates: {len(df) - len(df_dedup):,} rows removed.")

    # 2. Format InvoiceDate
    df_dedup['InvoiceDate'] = pd.to_datetime(df_dedup['InvoiceDate'])

    # 3. Identify Cancellations
    is_cancellation = df_dedup['InvoiceNo'].str.startswith('C') | (df_dedup['Quantity'] < 0)
    df_cancellations = df_dedup[is_cancellation].copy()
    print(f"  - Isolated cancellations dataset: {len(df_cancellations):,} records.")

    # 4. Filter Active Valid Purchases
    valid_mask = (~is_cancellation) & (df_dedup['Quantity'] > 0) & (df_dedup['UnitPrice'] > 0)
    df_cleaned = df_dedup[valid_mask].copy()

    # 5. Compute TotalAmount
    df_cleaned['TotalAmount'] = (df_cleaned['Quantity'] * df_cleaned['UnitPrice']).round(2)
    df_cancellations['TotalAmount'] = (df_cancellations['Quantity'].abs() * df_cancellations['UnitPrice']).round(2)

    print(f"  - Valid Cleaned Purchases: {len(df_cleaned):,} records.")
    return df_cleaned, df_cancellations


def create_customer_dataset(df_cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 4: Prepare Customer-Specific Dataset and Customer Aggregations.
    - Retains records where CustomerID is available (for ML & Customer Intelligence).
    - Aggregates RFM metrics (Recency, Frequency, Monetary value).
    """
    print("\n[4/5] Building Customer Dataset & RFM Aggregations...")
    
    # Customer transaction lines
    df_cust_trans = df_cleaned[df_cleaned['CustomerID'].notnull()].copy()
    df_cust_trans['CustomerID'] = df_cust_trans['CustomerID'].astype(int)
    print(f"  - Customer Transaction Records: {len(df_cust_trans):,} lines.")

    # Customer Aggregations
    ref_date = df_cust_trans['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    customer_summary = df_cust_trans.groupby('CustomerID').agg(
        total_spend=('TotalAmount', 'sum'),
        total_quantity=('Quantity', 'sum'),
        total_invoices=('InvoiceNo', 'nunique'),
        unique_products=('StockCode', 'nunique'),
        first_purchase=('InvoiceDate', 'min'),
        last_purchase=('InvoiceDate', 'max'),
        country=('Country', lambda s: s.mode().iloc[0] if not s.empty else 'Unknown')
    ).reset_index()

    # RFM metrics
    customer_summary['recency_days'] = (ref_date - customer_summary['last_purchase']).dt.days
    customer_summary['tenure_days'] = (ref_date - customer_summary['first_purchase']).dt.days
    customer_summary['avg_order_value'] = (customer_summary['total_spend'] / customer_summary['total_invoices']).round(2)
    customer_summary['total_spend'] = customer_summary['total_spend'].round(2)

    print(f"  - Unique Identified Customers: {len(customer_summary):,} profiles generated.")
    return df_cust_trans, customer_summary


def create_product_dataset(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Prepare Product Aggregations.
    - Aggregates product popularity, total units, total revenue, distinct buyers.
    """
    print("\n[5/5] Building Product Performance Dataset...")
    
    product_summary = df_cleaned.groupby('StockCode').agg(
        description=('Description', lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else 'Unknown Product'),
        total_quantity=('Quantity', 'sum'),
        total_revenue=('TotalAmount', 'sum'),
        total_invoices=('InvoiceNo', 'nunique'),
        unique_customers=('CustomerID', lambda s: s.dropna().nunique()),
        avg_unit_price=('UnitPrice', 'mean'),
        min_unit_price=('UnitPrice', 'min'),
        max_unit_price=('UnitPrice', 'max')
    ).reset_index()

    product_summary['total_revenue'] = product_summary['total_revenue'].round(2)
    product_summary['avg_unit_price'] = product_summary['avg_unit_price'].round(2)
    
    # Filter out anomalous zero-sales products
    product_summary = product_summary.sort_values(by='total_revenue', ascending=False)
    print(f"  - Unique Catalog Products: {len(product_summary):,} items cataloged.")
    return product_summary


def save_processed_files(df_cleaned: pd.DataFrame,
                         df_cust_trans: pd.DataFrame,
                         df_cancellations: pd.DataFrame,
                         customer_summary: pd.DataFrame,
                         product_summary: pd.DataFrame,
                         processed_dir: str = os.path.join('data', 'processed')):
    """
    Save all processed datasets in CSV and fast Parquet formats.
    """
    os.makedirs(processed_dir, exist_ok=True)
    print(f"\n[Saving Datasets to {processed_dir}]...")

    files = {
        'cleaned_transactions': df_cleaned,
        'customer_transactions': df_cust_trans,
        'cancellations': df_cancellations,
        'customer_data': customer_summary,
        'product_data': product_summary
    }

    for name, data in files.items():
        csv_path = os.path.join(processed_dir, f"{name}.csv")
        parquet_path = os.path.join(processed_dir, f"{name}.parquet")
        
        # Save CSV
        data.to_csv(csv_path, index=False)
        # Save Parquet for high-speed app loading
        data.to_parquet(parquet_path, index=False)
        print(f"  [OK] Saved: {name}.csv & {name}.parquet ({len(data):,} rows)")

    print("\n[SUCCESS] All clean datasets successfully persisted.")
