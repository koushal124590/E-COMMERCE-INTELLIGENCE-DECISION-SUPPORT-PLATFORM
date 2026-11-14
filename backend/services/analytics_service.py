"""
Analytics Service Module
E-Commerce Intelligence & Decision Support Platform

Purpose:
--------
Provides business intelligence KPIs, customer profile analytics,
product catalog browsing, and model benchmark comparisons.
"""

from typing import List, Dict, Any, Optional
from backend.database.data_loader import db


class AnalyticsService:
    """Service handling e-commerce intelligence, RFM analytics, and KPIs."""

    @staticmethod
    def get_platform_kpis() -> Dict[str, Any]:
        """Return overarching business intelligence platform KPIs."""
        if not db.is_loaded:
            db.load_all()

        total_customers = len(db.customers_df) if db.customers_df is not None else 0
        total_products = len(db.products_df) if db.products_df is not None else 0
        total_revenue = float(db.customers_df["total_spending"].sum()) if db.customers_df is not None else 0.0
        avg_order_val = float(db.customers_df["avg_order_value"].mean()) if db.customers_df is not None else 0.0
        total_orders = int(db.customers_df["total_orders"].sum()) if db.customers_df is not None else 0

        return {
            "platform_name": "E-Commerce Intelligence & Decision Support Platform",
            "total_customers": total_customers,
            "total_products": total_products,
            "total_orders": total_orders,
            "total_revenue_gbp": round(total_revenue, 2),
            "average_order_value_gbp": round(avg_order_val, 2),
            "status": "Online",
            "ml_models_active": ["Decision Tree", "Logistic Regression", "Popularity Baseline"]
        }

    @staticmethod
    def get_customer_profile(customer_id: int) -> Optional[Dict[str, Any]]:
        """
        Return the RFM (Recency, Frequency, Monetary) intelligence profile
        for a specific customer.
        """
        if not db.is_loaded:
            db.load_all()

        if db.customers_df is None:
            return None

        cust_match = db.customers_df[db.customers_df["CustomerID"] == customer_id]
        if len(cust_match) == 0:
            return None

        row = cust_match.iloc[0]
        spend = float(row["total_spending"])
        orders = int(row["total_orders"])
        recency = int(row["recency_days"])
        tenure = int(row["tenure_days"])

        # Simple segment heuristic
        if spend >= 2000 and recency <= 60:
            segment = "VIP Champion (High Spend, High Activity)"
        elif spend >= 1000:
            segment = "Loyal Customer (Steady Spender)"
        elif recency > 180:
            segment = "At-Risk / Lapsed Customer"
        else:
            segment = "Regular Shopper"

        return {
            "customer_id": int(row["CustomerID"]),
            "segment": segment,
            "country": str(row["country"]),
            "total_spending_gbp": round(spend, 2),
            "total_orders": orders,
            "avg_order_value_gbp": round(float(row["avg_order_value"]), 2),
            "total_quantity_items": int(row["total_quantity"]),
            "unique_products_bought": int(row["unique_products"]),
            "recency_days": recency,
            "tenure_days": tenure,
            "first_purchase_date": str(row["first_purchase"]),
            "last_purchase_date": str(row["last_purchase"]),
        }

    @staticmethod
    def get_model_benchmarks() -> List[Dict[str, Any]]:
        """Return the official model evaluation and performance metrics."""
        if not db.is_loaded:
            db.load_all()

        if db.evaluation_metrics_df is None:
            return []

        results = []
        for _, row in db.evaluation_metrics_df.iterrows():
            results.append({
                "model": str(row["model"]),
                "hits": int(row["hits"]),
                "total_customers": int(row["total_customers"]),
                "hit_rate_at_5_pct": float(row["hit_rate_pct"]),
                "ci_95_lower_pct": float(row["ci_95_lower_pct"]),
                "ci_95_upper_pct": float(row["ci_95_upper_pct"]),
                "improvement_vs_level1_pct": float(row["improvement_over_level1_pct"]),
                "status": str(row["status"])
            })
        return results

    @staticmethod
    def search_products(query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """Search products by description or StockCode."""
        if not db.is_loaded:
            db.load_all()

        if db.products_df is None:
            return []

        df = db.products_df
        if query:
            q = query.lower().strip()
            mask = df["description"].str.lower().str.contains(q, na=False) | df["StockCode"].str.lower().str.contains(q, na=False)
            filtered = df[mask]
        else:
            filtered = df.sort_values(by="total_orders", ascending=False)

        results = []
        for _, row in filtered.head(limit).iterrows():
            results.append({
                "stock_code": str(row["StockCode"]),
                "description": str(row["description"]),
                "total_orders": int(row["total_orders"]),
                "total_revenue_gbp": round(float(row["total_revenue"]), 2),
                "avg_unit_price_gbp": round(float(row["avg_unit_price"]), 2),
                "popularity_score": round(float(row["popularity_score"]), 1)
            })
        return results

