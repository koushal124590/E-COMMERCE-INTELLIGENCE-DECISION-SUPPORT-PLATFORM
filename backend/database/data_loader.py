"""
Data Loader Module for Backend
E-Commerce Intelligence & Decision Support Platform

Purpose:
--------
Loads all necessary processed CSV datasets and trained machine learning
models into memory once at server startup for fast response times.
"""

import os
import joblib
import pandas as pd

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trained_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


class DataLoader:
    """Singleton class that holds in-memory cached datasets and models."""

    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.dt_recommendations_df = None
        self.lr_recommendations_df = None
        self.popularity_recommendations_df = None
        self.evaluation_metrics_df = None
        self.top5_popular_df = None
        self.dt_model = None
        self.lr_model = None
        self.scaler = None
        self.is_loaded = False

    def load_all(self):
        """Load datasets and models into memory."""
        if self.is_loaded:
            return

        print("[Backend] Loading data and machine learning models into memory...")

        # 1. Customer Profiles & Features
        cust_path = os.path.join(DATA_DIR, "customer_features.csv")
        if os.path.exists(cust_path):
            self.customers_df = pd.read_csv(cust_path)
            print(f"  [OK] Loaded {len(self.customers_df):,} customer profiles.")

        # 2. Product Catalog & Performance
        prod_path = os.path.join(DATA_DIR, "product_features.csv")
        if os.path.exists(prod_path):
            self.products_df = pd.read_csv(prod_path)
            print(f"  [OK] Loaded {len(self.products_df):,} catalog products.")

        # 3. Top-5 Recommendations Datasets
        dt_rec_path = os.path.join(DATA_DIR, "decision_tree_recommendations.csv")
        if os.path.exists(dt_rec_path):
            self.dt_recommendations_df = pd.read_csv(dt_rec_path)

        lr_rec_path = os.path.join(DATA_DIR, "logistic_regression_recommendations.csv")
        if os.path.exists(lr_rec_path):
            self.lr_recommendations_df = pd.read_csv(lr_rec_path)

        pop_rec_path = os.path.join(DATA_DIR, "popularity_recommendations.csv")
        if os.path.exists(pop_rec_path):
            self.popularity_recommendations_df = pd.read_csv(pop_rec_path)

        # 4. Top-5 Popular Products Reference
        top5_path = os.path.join(DATA_DIR, "top5_popular_products.csv")
        if os.path.exists(top5_path):
            self.top5_popular_df = pd.read_csv(top5_path)

        # 5. Model Evaluation Metrics
        metrics_path = os.path.join(OUTPUTS_DIR, "model_evaluation_metrics.csv")
        if os.path.exists(metrics_path):
            self.evaluation_metrics_df = pd.read_csv(metrics_path)

        # 6. Trained Models
        dt_model_path = os.path.join(MODEL_DIR, "decision_tree.joblib")
        if os.path.exists(dt_model_path):
            self.dt_model = joblib.load(dt_model_path)

        lr_model_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
        if os.path.exists(lr_model_path):
            self.lr_model = joblib.load(lr_model_path)

        scaler_path = os.path.join(MODEL_DIR, "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        self.is_loaded = True
        print("[Backend] All datasets and models ready in memory!\n")


# Global data loader instance
db = DataLoader()

