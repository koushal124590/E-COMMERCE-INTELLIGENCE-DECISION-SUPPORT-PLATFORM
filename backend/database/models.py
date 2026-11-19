"""
SQLAlchemy Database Models for PostgreSQL
E-Commerce Intelligence & Decision Support Platform

Tables:
-------
1. customers       -> Customer profiles, spending, and RFM metrics
2. products        -> Catalog items, revenue, and popularity scores
3. recommendations -> Top-5 recommendations from ML models (Decision Tree, Logistic Regression)
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Index
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class Customer(Base):
    """Stores customer RFM profiles and segmentation."""
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True, index=True)
    country = Column(String(100), default="Unknown")
    total_spending = Column(Float, default=0.0)
    total_orders = Column(Integer, default=0)
    total_quantity = Column(Integer, default=0)
    unique_products = Column(Integer, default=0)
    avg_order_value = Column(Float, default=0.0)
    recency_days = Column(Integer, default=0)
    tenure_days = Column(Integer, default=0)
    segment = Column(String(100), default="Regular Shopper")
    first_purchase = Column(String(50), nullable=True)
    last_purchase = Column(String(50), nullable=True)

    # Relationships
    recommendations = relationship("Recommendation", back_populates="customer", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Customer {self.customer_id} ({self.country}) - Spend: £{self.total_spending}>"


class Product(Base):
    """Stores catalog SKU intelligence and popularity metrics."""
    __tablename__ = "products"

    stock_code = Column(String(50), primary_key=True, index=True)
    description = Column(String(255), default="Unknown Product")
    total_quantity_sold = Column(Integer, default=0)
    total_revenue = Column(Float, default=0.0)
    total_orders = Column(Integer, default=0)
    unique_customers = Column(Integer, default=0)
    avg_unit_price = Column(Float, default=0.0)
    popularity_score = Column(Float, default=0.0)

    # Relationships
    recommendations = relationship("Recommendation", back_populates="product")

    def __repr__(self):
        return f"<Product {self.stock_code}: {self.description[:25]} - Popularity: {self.popularity_score}>"


class Recommendation(Base):
    """Stores precomputed and ranked Top-5 recommendations per customer."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), index=True, nullable=False)
    stock_code = Column(String(50), ForeignKey("products.stock_code"), index=True, nullable=False)
    rank = Column(Integer, nullable=False)  # 1 to 5
    model_name = Column(String(50), nullable=False, index=True)  # 'decision_tree' or 'logistic_regression'
    probability = Column(Float, default=1.0)
    was_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="recommendations")
    product = relationship("Product", back_populates="recommendations")

    # Composite Index for fast customer + model lookups
    __table_args__ = (
        Index("idx_cust_model", "customer_id", "model_name"),
    )

    def __repr__(self):
        return f"<Recommendation Cust {self.customer_id} -> Rank {self.rank}: {self.stock_code} ({self.model_name})>"
