"""
Database Connection & Engine Configuration
E-Commerce Intelligence & Decision Support Platform

Supports:
---------
- PostgreSQL via psycopg2 / SQLAlchemy
- Configurable via DATABASE_URL in .env or environment variables
- Connection pooling and session lifecycle management
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Default PostgreSQL Connection URL (using psycopg2 driver)
# Format: postgresql+psycopg2://[user]:[password]@[host]:[port]/[database]
DEFAULT_DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce_db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Normalize postgresql:// to postgresql+psycopg2:// if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = None
SessionLocal = None
_is_connected = None
_last_check_time = 0.0


def get_engine():
    """Create or return SQLAlchemy engine with connection pool."""
    global engine
    if engine is None:
        try:
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 1} if "postgresql" in DATABASE_URL else {}
            )
        except Exception as e:
            print(f"[Database] Engine creation warning: {e}")
            engine = None
    return engine


def get_session_factory():
    """Return sessionmaker bound to the engine."""
    global SessionLocal
    if SessionLocal is None:
        eng = get_engine()
        if eng:
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return SessionLocal


def check_db_connection() -> bool:
    """
    Check if PostgreSQL is currently reachable.
    Caches connection result for 30s to avoid latency when offline.
    """
    global _is_connected, _last_check_time
    import time
    now = time.time()
    if _is_connected is not None and (now - _last_check_time) < 30.0:
        return _is_connected

    _last_check_time = now
    try:
        eng = get_engine()
        if eng is None:
            _is_connected = False
            return False
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        _is_connected = True
        return True
    except Exception:
        _is_connected = False
        return False


def get_db():
    """FastAPI dependency for yielding database sessions."""
    session_factory = get_session_factory()
    if session_factory is None:
        yield None
        return

    db = session_factory()
    try:
        yield db
    finally:
        db.close()
