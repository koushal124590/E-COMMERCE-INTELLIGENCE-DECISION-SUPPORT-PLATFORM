"""
Backend API Unit Tests
E-Commerce Intelligence & Decision Support Platform

Tests all endpoints using FastAPI's TestClient.
"""

import sys
import os
import unittest

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.data_loader import db


class TestBackendAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        # Load data once for all tests
        db.load_all()
        cls.client = TestClient(app)

    def test_01_root(self):
        """Test root / endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("interactive_docs", data)

    def test_02_health(self):
        """Test /api/health endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_03_kpis(self):
        """Test /api/kpis endpoint."""
        response = self.client.get("/api/kpis")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_customers", data)
        self.assertIn("total_revenue_gbp", data)
        self.assertGreater(data["total_customers"], 0)
        self.assertGreater(data["total_revenue_gbp"], 0)

    def test_04_recommendations_decision_tree(self):
        """Test /api/recommendations/{customer_id} with Decision Tree on test customer."""
        response = self.client.get("/api/recommendations/16128?model=decision_tree")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 16128)
        self.assertEqual(len(data["recommendations"]), 5)
        self.assertIn("Decision Tree", data["model_used"])

    def test_05_recommendations_logistic_regression(self):
        """Test /api/recommendations/{customer_id} with Logistic Regression on test customer."""
        response = self.client.get("/api/recommendations/16128?model=logistic_regression")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 16128)
        self.assertEqual(len(data["recommendations"]), 5)
        self.assertIn("Logistic Regression", data["model_used"])

    def test_06_recommendations_fallback(self):
        """Test /api/recommendations/{customer_id} fallback for cold-start customer."""
        response = self.client.get("/api/recommendations/999999?model=decision_tree")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 999999)
        self.assertEqual(len(data["recommendations"]), 5)
        self.assertTrue(data["is_cold_start_fallback"])
        self.assertIn("Popularity", data["model_used"])

    def test_07_customer_profile(self):
        """Test /api/customers/{customer_id} endpoint."""
        response = self.client.get("/api/customers/12347")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], 12347)
        self.assertIn("total_spending_gbp", data)
        self.assertIn("segment", data)

    def test_08_customer_not_found(self):
        """Test /api/customers/9999999 returns 404."""
        response = self.client.get("/api/customers/9999999")
        self.assertEqual(response.status_code, 404)

    def test_09_popular_products(self):
        """Test /api/products/popular endpoint."""
        response = self.client.get("/api/products/popular")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["products"]), 5)

    def test_10_search_products(self):
        """Test /api/products/search endpoint."""
        response = self.client.get("/api/products/search?query=HEART&limit=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data["results"]), 3)

    def test_11_model_benchmarks(self):
        """Test /api/models/benchmarks endpoint."""
        response = self.client.get("/api/models/benchmarks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 4)


if __name__ == "__main__":
    unittest.main()

