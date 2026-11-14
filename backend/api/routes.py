"""
FastAPI Routes Definition
E-Commerce Intelligence & Decision Support Platform

Endpoints:
----------
1. GET /api/health                      -> Service health check
2. GET /api/kpis                        -> Platform overview KPIs
3. GET /api/recommendations/{cust_id}   -> Top-5 recommendations (DT / LR / Popularity)
4. GET /api/customers/{cust_id}         -> Customer RFM profile
5. GET /api/products/popular            -> Top 5 popular bestsellers
6. GET /api/products/search             -> Search catalog
7. GET /api/models/benchmarks           -> Model comparison metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.services.recommendation_service import RecommendationService
from backend.services.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check():
    """Verify backend API is running and responsive."""
    return {
        "status": "healthy",
        "service": "E-Commerce Intelligence & Decision Support API",
        "version": "1.0.0"
    }


@router.get("/kpis", tags=["Analytics"])
def get_platform_kpis():
    """Get high-level business intelligence platform KPIs."""
    return AnalyticsService.get_platform_kpis()


@router.get("/recommendations/{customer_id}", tags=["Recommendations"])
def get_customer_recommendations(
    customer_id: int,
    model: str = Query(
        default="decision_tree",
        description="Recommendation model: 'decision_tree' (best), 'logistic_regression', or 'popularity'"
    )
):
    """
    Get personalized Top-5 product recommendations for a customer.
    Falls back gracefully to the Level 1 Popularity baseline for cold-start / new customers.
    """
    return RecommendationService.get_recommendations(customer_id=customer_id, model=model)


@router.get("/customers/{customer_id}", tags=["Customer Intelligence"])
def get_customer_profile(customer_id: int):
    """Get detailed RFM customer profile and intelligence metrics."""
    profile = AnalyticsService.get_customer_profile(customer_id=customer_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer ID {customer_id} was not found in historical records."
        )
    return profile


@router.get("/products/popular", tags=["Catalog"])
def get_popular_products():
    """Get the Top 5 Bestselling products from historical data."""
    return {
        "title": "Top 5 Popular Products (Historical Benchmark)",
        "products": RecommendationService.get_top_popular_products()
    }


@router.get("/products/search", tags=["Catalog"])
def search_products(
    query: str = Query(default="", description="Product description keyword or StockCode"),
    limit: int = Query(default=10, ge=1, le=50, description="Max products to return")
):
    """Search catalog products by name or SKU."""
    return {
        "query": query,
        "results": AnalyticsService.search_products(query=query, limit=limit)
    }


@router.get("/models/benchmarks", tags=["ML Evaluation"])
def get_model_benchmarks():
    """Get model performance comparison and 95% confidence intervals."""
    return {
        "metric": "Hit-Rate@5",
        "evaluation_cohort": "781 Test Customers (Jun-Dec 2011)",
        "models": AnalyticsService.get_model_benchmarks()
    }

