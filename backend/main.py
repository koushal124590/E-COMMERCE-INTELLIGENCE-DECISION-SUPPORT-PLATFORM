"""
FastAPI Backend Application Entrypoint
E-Commerce Intelligence & Decision Support Platform (EcomIQ)

Purpose:
--------
Runs the REST API backend server with interactive Swagger documentation,
CORS support, and in-memory data loading.

To start the server:
  uvicorn backend.main:app --reload --port 8000
"""

import sys
import os

# Ensure project root is in Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.database.data_loader import db
from backend.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and models on startup."""
    print("\n[FastAPI] Initializing server...")
    db.load_all()
    print("[FastAPI] Server startup complete! Interactive docs at: http://localhost:8000/docs\n")
    yield
    print("[FastAPI] Server shutting down.")


# Create FastAPI application
app = FastAPI(
    title="E-Commerce Intelligence & Decision Support API",
    description="""
    REST API providing real-time personalized product recommendations,
    customer RFM analytics, product catalog intelligence, and ML model evaluation metrics.
    
    ### Available Features:
    * **Personalized Recommendations:** Top-5 product recommendations powered by Decision Tree & Logistic Regression models.
    * **Customer Intelligence:** Customer RFM segmentation, spending metrics, and order history.
    * **Catalog Performance:** SKU velocity, popularity scores, and product search.
    * **Model Benchmarks:** Real Hit-Rate@5 comparisons and 95% confidence intervals.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows frontend demo and local web apps to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes with prefix /api
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["Root"])
def root():
    """Welcome route providing helpful links."""
    return {
        "message": "Welcome to the E-Commerce Intelligence & Decision Support API!",
        "interactive_docs": "http://localhost:8000/docs",
        "health_check": "http://localhost:8000/api/health",
        "sample_endpoints": {
            "platform_kpis": "/api/kpis",
            "customer_recommendations": "/api/recommendations/12347",
            "customer_profile": "/api/customers/12347",
            "popular_products": "/api/products/popular",
            "model_benchmarks": "/api/models/benchmarks"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

