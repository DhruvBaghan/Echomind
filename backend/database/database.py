# ============================================
# EchoMind - Database Utilities
# ============================================

"""
Database connection and initialization utilities.

Provides functions for:
    - Database initialization
    - Session management
    - Connection testing
    - Database migrations
"""

from contextlib import contextmanager
from typing import Optional, Generator

from flask import Flask, current_app
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.database import db
from backend.utils.logger import logger


def init_db(app: Flask) -> None:
    """
    Initialize database with Flask application.
    
    Args:
        app: Flask application instance
    """
    try:
        # Initialize SQLAlchemy with app
        db.init_app(app)
        
        with app.app_context():
            # Import models to register them
            from backend.database.models import (
                User, UsageHistory, Preference, Prediction, Alert
            )
            
            # Create all tables
            db.create_all()
            
            # Verify connection
            if test_connection():
                logger.info("Database initialized successfully")
            else:
                logger.error("Database connection test failed")
                
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        db.session.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_db_session():
    """
    Get current database session.
    
    Returns:
        SQLAlchemy session
    """
    return db.session


def close_db_session(exception: Optional[Exception] = None) -> None:
    """
    Close database session.
    
    Args:
        exception: Optional exception that caused the close
    """
    try:
        if exception:
            db.session.rollback()
            logger.warning(f"Session rolled back due to: {exception}")
        db.session.remove()
    except Exception as e:
        logger.error(f"Error closing session: {e}")


@contextmanager
def db_transaction() -> Generator:
    """
    Context manager for database transactions.
    
    Automatically commits on success or rolls back on failure.
    
    Usage:
        with db_transaction():
            # database operations
            db.session.add(model)
    
    Yields:
        Database session
    """
    try:
        yield db.session
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Transaction rolled back: {e}")
        raise
    except Exception as e:
        db.session.rollback()
        logger.error(f"Transaction error: {e}")
        raise


def create_tables() -> None:
    """Create all database tables."""
    try:
        db.create_all()
        logger.info("Database tables created")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def drop_tables() -> None:
    """
    Drop all database tables.
    Use with caution - this will delete all data!
    """
    try:
        db.drop_all()
        logger.warning("All database tables dropped")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        raise


def reset_database() -> None:
    """
    Reset database by dropping and recreating all tables.
    Use with caution - this will delete all data!
    """
    logger.warning("Resetting database - all data will be lost!")
    drop_tables()
    create_tables()
    logger.info("Database reset complete")


def seed_demo_data() -> None:
    """
    Seed database with demo data for testing.
    """
    from datetime import datetime, timedelta
    from backend.database.models import User, UsageHistory, Preference
    
    try:
        # Check if demo user exists
        demo_user = User.query.filter_by(email="demo@echomind.io").first()
        
        if not demo_user:
            # Create demo user
            demo_user = User(
                email="demo@echomind.io",
                password="demo123",
                name="Demo User",
                household_size=4,
                location="San Francisco, CA"
            )
            db.session.add(demo_user)
            db.session.commit()
            
            # Create preferences
            prefs = Preference(user_id=demo_user.id)
            db.session.add(prefs)
            
            # Create sample usage history
            base_date = datetime.now() - timedelta(days=30)
            
            for i in range(30 * 24):  # 30 days of hourly data
                timestamp = base_date + timedelta(hours=i)
                hour = timestamp.hour
                
                # Electricity with daily pattern
                if 6 <= hour <= 9 or 17 <= hour <= 21:
                    elec_base = 2.0
                elif 0 <= hour <= 5:
                    elec_base = 0.5
                else:
                    elec_base = 1.2
                
                elec_usage = UsageHistory(
                    resource_type="electricity",
                    consumption=round(elec_base + (i % 10) * 0.1, 2),
                    recorded_at=timestamp,
                    user_id=demo_user.id,
                    source="demo"
                )
                db.session.add(elec_usage)
                
                # Water with daily pattern
                if 6 <= hour <= 9 or 18 <= hour <= 22:
                    water_base = 40
                elif 0 <= hour <= 5:
                    water_base = 5
                else:
                    water_base = 15
                
                water_usage = UsageHistory(
                    resource_type="water",
                    consumption=round(water_base + (i % 20), 2),
                    recorded_at=timestamp,
                    user_id=demo_user.id,
                    source="demo"
                )
                db.session.add(water_usage)
            
            db.session.commit()
            logger.info("Demo data seeded successfully")
        else:
            logger.info("Demo user already exists, skipping seed")
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error seeding demo data: {e}")
        raise


def get_database_stats() -> dict:
    """
    Get database statistics.
    
    Returns:
        Dictionary with database statistics
    """
    from backend.database.models import User, UsageHistory, Prediction, Alert
    
    try:
        stats = {
            "users": {
                "total": User.query.count(),
                "active": User.query.filter_by(is_active=True).count(),
            },
            "usage_history": {
                "total": UsageHistory.query.count(),
                "electricity": UsageHistory.query.filter_by(resource_type="electricity").count(),
                "water": UsageHistory.query.filter_by(resource_type="water").count(),
            },
            "predictions": {
                "total": Prediction.query.count(),
            },
            "alerts": {
                "total": Alert.query.count(),
                "unread": Alert.query.filter_by(is_read=False).count(),
            },
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}


def cleanup_old_data(days: int = 90) -> dict:
    """
    Clean up old data from the database.
    
    Args:
        days: Remove data older than this many days
        
    Returns:
        Dictionary with cleanup results
    """
    from datetime import datetime, timedelta
    from backend.database.models import UsageHistory, Prediction, Alert
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        # Clean up old usage history (keep user data longer)
        # Only clean anonymous data
        usage_deleted = UsageHistory.query.filter(
            UsageHistory.user_id.is_(None),
            UsageHistory.created_at < cutoff_date
        ).delete()
        
        # Clean up old predictions
        predictions_deleted = Prediction.query.filter(
            Prediction.created_at < cutoff_date
        ).delete()
        
        # Clean up dismissed alerts
        alerts_deleted = Alert.query.filter(
            Alert.is_dismissed == True,
            Alert.created_at < cutoff_date
        ).delete()
        
        db.session.commit()
        
        result = {
            "usage_history_deleted": usage_deleted,
            "predictions_deleted": predictions_deleted,
            "alerts_deleted": alerts_deleted,
            "cutoff_date": cutoff_date.isoformat(),
        }
        
        logger.info(f"Database cleanup complete: {result}")
        return result
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during cleanup: {e}")
        return {"error": str(e)}


def backup_database(backup_path: str) -> bool:
    """
    Create a backup of the database (SQLite only).
    
    Args:
        backup_path: Path for the backup file
        
    Returns:
        True if successful, False otherwise
    """
    import shutil
    from pathlib import Path
    
    try:
        # Get database path from config
        db_url = str(db.engine.url)
        
        if "sqlite" not in db_url:
            logger.warning("Backup only supported for SQLite databases")
            return False
        
        # Extract database file path
        db_path = db_url.replace("sqlite:///", "")
        
        if not Path(db_path).exists():
            logger.error(f"Database file not found: {db_path}")
            return False
        
        # Create backup
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Backup error: {e}")
        return False