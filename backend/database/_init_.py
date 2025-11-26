# ============================================
# EchoMind - Database Package Initialization
# ============================================

"""
Database Package

This package contains database models and connection management for EchoMind.

Modules:
    - models: SQLAlchemy ORM models for users, usage history, etc.
    - database: Database connection and initialization utilities

The application uses SQLAlchemy as the ORM with support for:
    - SQLite (development)
    - PostgreSQL (production)
"""

# db is already initialized in __init__.py
from backend.database import db

# Import models after db initialization to avoid circular imports
from backend.database.models import User, UsageHistory, Preference, Prediction
from backend.database.database import init_db, get_db_session, close_db_session

# Package exports
__all__ = [
    "db",
    "init_db",
    "get_db_session",
    "close_db_session",
    "User",
    "UsageHistory",
    "Preference",
    "Prediction",
]


def reset_database():
    """
    Reset the database by dropping and recreating all tables.
    Use with caution - this will delete all data!
    """
    from backend.utils.logger import logger
    
    logger.warning("Resetting database - all data will be lost!")
    db.drop_all()
    db.create_all()
    logger.info("Database reset complete")


def get_table_names():
    """
    Get list of all table names in the database.
    
    Returns:
        List of table names
    """
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    return inspector.get_table_names()


def get_database_info():
    """
    Get information about the database.
    
    Returns:
        Dictionary with database information
    """
    return {
        "dialect": db.engine.dialect.name,
        "driver": db.engine.driver,
        "tables": get_table_names(),
        "url": str(db.engine.url).replace(
            db.engine.url.password or "", "***"
        ) if db.engine.url.password else str(db.engine.url),
    }