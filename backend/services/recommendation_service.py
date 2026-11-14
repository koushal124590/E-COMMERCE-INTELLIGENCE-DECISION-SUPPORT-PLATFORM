"""
Recommendation Service Module
E-Commerce Intelligence & Decision Support Platform

Purpose:
--------
Provides personalized product recommendations for customers using
either Machine Learning (Decision Tree or Logistic Regression)
or the Popularity Baseline.
"""

from typing import List, Dict, Any, Optional
from backend.database.data_loader import db


class RecommendationService:
    """Service handling product recommendations for customers."""

    @staticmethod
    def get_recommendations(
        customer_id: int,
        model: str = "decision_tree"
    ) -> Dict[str, Any]:
        """
        Get personalized Top-5 recommendations for a customer.

        Parameters:
        -----------
        customer_id: int
            The ID of the customer (e.g. 12347, 12348, etc.)
        model: str
            The model to use: 'decision_tree' (default, best model),
            'logistic_regression', or 'popularity'.

        Returns:
        --------
        Dictionary containing customer info, model used, and list of Top-5 products.
        """
        # Ensure database is loaded
        if not db.is_loaded:
            db.load_all()

        model_clean = model.lower().strip()

        # Select the appropriate recommendations dataframe
        if model_clean in ["decision_tree", "dt", "tree"]:
            df_recs = db.dt_recommendations_df
            model_display_name = "Decision Tree (Personalized ML)"
        elif model_clean in ["logistic_regression", "lr", "linear"]:
            df_recs = db.lr_recommendations_df
            model_display_name = "Logistic Regression (Personalized ML)"
        elif model_clean in ["popularity", "popular", "baseline"]:
            df_recs = db.popularity_recommendations_df
            model_display_name = "Popularity Baseline (Top 5 Bestsellers)"
        else:
            # Default to Decision Tree
            df_recs = db.dt_recommendations_df
            model_display_name = "Decision Tree (Personalized ML)"

        # Find recommendations for this customer
        cust_recs = df_recs[df_recs["CustomerID"] == customer_id] if df_recs is not None else None

        fallback_used = False

        # If customer not in test cohort (e.g. new customer or training customer),
        # gracefully fallback to Top-5 Popularity baseline!
        if cust_recs is None or len(cust_recs) == 0:
            fallback_used = True
            model_display_name = "Popularity Baseline (Cold-start Fallback)"
            if db.top5_popular_df is not None:
                cust_recs = db.top5_popular_df.copy()
                cust_recs["recommended_product"] = cust_recs["StockCode"]
                cust_recs["probability"] = 1.0
                cust_recs["was_hit"] = 0

        # Build detailed product list
        recommended_products: List[Dict[str, Any]] = []

        if cust_recs is not None:
            for _, row in cust_recs.head(5).iterrows():
                stock_code = str(row.get("recommended_product", row.get("StockCode", "")))
                prob = float(row.get("probability", 1.0))
                rank = int(row.get("rank", len(recommended_products) + 1))
                was_hit = bool(row.get("was_hit", 0) == 1)

                # Fetch extra product details from catalog
                desc = "Unknown Product"
                price = 0.0
                popularity = 0.0

                if db.products_df is not None:
                    prod_info = db.products_df[db.products_df["StockCode"] == stock_code]
                    if len(prod_info) > 0:
                        desc = str(prod_info["description"].values[0])
                        price = float(prod_info["avg_unit_price"].values[0])
                        popularity = float(prod_info["popularity_score"].values[0])

                recommended_products.append({
                    "rank": rank,
                    "stock_code": stock_code,
                    "description": desc,
                    "avg_unit_price": round(price, 2),
                    "popularity_score": round(popularity, 1),
                    "purchase_probability": round(prob, 4),
                    "bought_in_evaluation": was_hit
                })

        # Check if customer exists in profiles
        customer_exists = False
        customer_country = "Unknown"
        if db.customers_df is not None:
            c_info = db.customers_df[db.customers_df["CustomerID"] == customer_id]
            if len(c_info) > 0:
                customer_exists = True
                customer_country = str(c_info["country"].values[0])

        return {
            "customer_id": customer_id,
            "customer_registered": customer_exists,
            "country": customer_country,
            "model_requested": model,
            "model_used": model_display_name,
            "is_cold_start_fallback": fallback_used,
            "total_recommendations": len(recommended_products),
            "recommendations": recommended_products
        }

    @staticmethod
    def get_top_popular_products() -> List[Dict[str, Any]]:
        """Return the Top 5 Bestselling products from historical data."""
        if not db.is_loaded:
            db.load_all()

        results = []
        if db.top5_popular_df is not None:
            for _, row in db.top5_popular_df.iterrows():
                results.append({
                    "rank": int(row["rank"]),
                    "stock_code": str(row["StockCode"]),
                    "description": str(row["description"]),
                    "historical_orders": int(row["historical_orders"])
                })
        return results

