#!/usr/bin/env python3
# ============================================
# EchoMind - Database Setup Script
# ============================================

"""
Script to setup and manage the EchoMind database.

This script can:
    1. Initialize the database schema
    2. Create required tables
    3. Seed demo data
    4. Reset the database
    5. Run migrations (if needed)
    6. Backup the database

Usage:
    python scripts/setup_database.py
    python scripts/setup_database.py --init
    python scripts/setup_database.py --seed
    python scripts/setup_database.py --reset
    python scripts/setup_database.py --backup
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def init_database(app):
    """
    Initialize database tables.
    
    Args:
        app: Flask application instance
    """
    from backend.database import db
    
    print("Initializing database...")
    
    with app.app_context():
        # Import all models to register them
        from backend.database.models import (
            User, UsageHistory, Preference, Prediction, Alert
        )
        
        # Create all tables
        db.create_all()
        
        print("✓ Database tables created")
        
        # List tables
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"\nCreated tables ({len(tables)}):")
        for table in tables:
            print(f"  - {table}")


def seed_demo_data(app):
    """
    Seed database with demo data.
    
    Args:
        app: Flask application instance
    """
    from backend.database import db
    from backend.database.models import User, UsageHistory, Preference, Alert
    
    print("\nSeeding demo data...")
    
    with app.app_context():
        # Check if demo user exists
        demo_user = User.query.filter_by(email='demo@echomind.io').first()
        
        if demo_user:
            print("Demo user already exists. Skipping user creation.")
        else:
            # Create demo user
            demo_user = User(
                email='demo@echomind.io',
                password='demo123',
                name='Demo User',
                household_size=4,
                location='San Francisco, CA'
            )
            demo_user.is_verified = True
            db.session.add(demo_user)
            db.session.commit()
            
            print("✓ Demo user created (demo@echomind.io / demo123)")
        
        # Create preferences
        if not Preference.query.filter_by(user_id=demo_user.id).first():
            prefs = Preference(
                user_id=demo_user.id,
                electricity_rate=0.12,
                water_rate=0.002
            )
            db.session.add(prefs)
            print("✓ User preferences created")
        
        # Generate sample usage history
        existing_count = UsageHistory.query.filter_by(user_id=demo_user.id).count()
        
        if existing_count < 100:
            print("Generating usage history...")
            
            import numpy as np
            
            base_date = datetime.now() - timedelta(days=30)
            
            for day in range(30):
                for hour in range(24):
                    timestamp = base_date + timedelta(days=day, hours=hour)
                    h = timestamp.hour
                    
                    # Electricity pattern
                    if 6 <= h <= 9:
                        elec_base = 2.0
                    elif 17 <= h <= 21:
                        elec_base = 2.5
                    elif 0 <= h <= 5:
                        elec_base = 0.5
                    else:
                        elec_base = 1.2
                    
                    elec_value = elec_base + np.random.uniform(-0.3, 0.3)
                    
                    db.session.add(UsageHistory(
                        user_id=demo_user.id,
                        resource_type='electricity',
                        consumption=round(max(0.1, elec_value), 2),
                        recorded_at=timestamp,
                        source='demo'
                    ))
                    
                    # Water pattern
                    if 6 <= h <= 9:
                        water_base = 45
                    elif 18 <= h <= 22:
                        water_base = 40
                    elif 0 <= h <= 5:
                        water_base = 5
                    else:
                        water_base = 15
                    
                    water_value = water_base + np.random.uniform(-5, 5)
                    
                    db.session.add(UsageHistory(
                        user_id=demo_user.id,
                        resource_type='water',
                        consumption=round(max(0, water_value), 1),
                        recorded_at=timestamp,
                        source='demo'
                    ))
            
            db.session.commit()
            print(f"✓ Generated {30 * 24 * 2} usage history entries")
        else:
            print(f"Usage history already exists ({existing_count} entries)")
        
        # Create sample alerts
        alert_count = Alert.query.filter_by(user_id=demo_user.id).count()
        
        if alert_count == 0:
            alerts = [
                Alert(
                    user_id=demo_user.id,
                    alert_type='info',
                    priority='low',
                    title='Welcome to EchoMind!',
                    message='Start tracking your consumption to get personalized insights.'
                ),
                Alert(
                    user_id=demo_user.id,
                    alert_type='tip',
                    resource_type='electricity',
                    priority='low',
                    title='Energy Saving Tip',
                    message='Consider using LED bulbs to reduce electricity consumption by up to 75%.'
                ),
            ]
            
            for alert in alerts:
                db.session.add(alert)
            
            db.session.commit()
            print(f"✓ Created {len(alerts)} sample alerts")
        
        print("\n✓ Demo data seeding complete!")


def reset_database(app, confirm: bool = False):
    """
    Reset the database (drop all tables and recreate).
    
    Args:
        app: Flask application instance
        confirm: Whether to skip confirmation
    """
    from backend.database import db
    
    if not confirm:
        response = input("\n⚠ WARNING: This will delete ALL data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    print("\nResetting database...")
    
    with app.app_context():
        # Drop all tables
        db.drop_all()
        print("✓ Dropped all tables")
        
        # Recreate tables
        db.create_all()
        print("✓ Recreated all tables")
    
    print("\n✓ Database reset complete!")


def backup_database(app):
    """
    Backup the database.
    
    Args:
        app: Flask application instance
    """
    from backend.config import Config
    
    print("\nBacking up database...")
    
    db_url = str(Config.SQLALCHEMY_DATABASE_URI)
    
    if 'sqlite' in db_url:
        # SQLite backup
        db_path = db_url.replace('sqlite:///', '')
        
        if not os.path.exists(db_path):
            print("✗ Database file not found")
            return
        
        backup_dir = PROJECT_ROOT / 'backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f'echomind_backup_{timestamp}.db'
        
        shutil.copy2(db_path, backup_path)
        
        print(f"✓ Database backed up to: {backup_path}")
        
        # Show backup size
        size = os.path.getsize(backup_path)
        if size > 1024 * 1024:
            print(f"  Size: {size / 1024 / 1024:.2f} MB")
        else:
            print(f"  Size: {size / 1024:.2f} KB")
    else:
        print("Database backup for non-SQLite databases requires pg_dump or similar tools.")
        print("Please use your database's native backup tools.")


def show_database_info(app):
    """
    Show database information and statistics.
    
    Args:
        app: Flask application instance
    """
    from backend.database import db, get_database_info
    from backend.database.database import get_database_stats
    
    print("\nDatabase Information")
    print("=" * 50)
    
    with app.app_context():
        info = get_database_info()
        
        print(f"\nDialect: {info['dialect']}")
        print(f"Driver: {info['driver']}")
        print(f"URL: {info['url']}")
        
        print(f"\nTables ({len(info['tables'])}):")
        for table in info['tables']:
            print(f"  - {table}")
        
        # Get statistics
        try:
            stats = get_database_stats()
            
            print("\nStatistics:")
            
            if 'users' in stats:
                print(f"\n  Users:")
                print(f"    - Total: {stats['users']['total']}")
                print(f"    - Active: {stats['users']['active']}")
            
            if 'usage_history' in stats:
                print(f"\n  Usage History:")
                print(f"    - Total entries: {stats['usage_history']['total']}")
                print(f"    - Electricity: {stats['usage_history']['electricity']}")
                print(f"    - Water: {stats['usage_history']['water']}")
            
            if 'predictions' in stats:
                print(f"\n  Predictions:")
                print(f"    - Total: {stats['predictions']['total']}")
            
            if 'alerts' in stats:
                print(f"\n  Alerts:")
                print(f"    - Total: {stats['alerts']['total']}")
                print(f"    - Unread: {stats['alerts']['unread']}")
                
        except Exception as e:
            print(f"\nCould not get statistics: {e}")


def run_cleanup(app, days: int = 90):
    """
    Clean up old data.
    
    Args:
        app: Flask application instance
        days: Remove data older than this many days
    """
    from backend.database.database import cleanup_old_data
    
    print(f"\nCleaning up data older than {days} days...")
    
    with app.app_context():
        result = cleanup_old_data(days=days)
        
        if 'error' in result:
            print(f"✗ Cleanup failed: {result['error']}")
        else:
            print("✓ Cleanup complete!")
            print(f"  - Usage history deleted: {result.get('usage_history_deleted', 0)}")
            print(f"  - Predictions deleted: {result.get('predictions_deleted', 0)}")
            print(f"  - Alerts deleted: {result.get('alerts_deleted', 0)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Setup and manage EchoMind database'
    )
    parser.add_argument(
        '--init', '-i',
        action='store_true',
        help='Initialize database tables'
    )
    parser.add_argument(
        '--seed', '-s',
        action='store_true',
        help='Seed demo data'
    )
    parser.add_argument(
        '--reset', '-r',
        action='store_true',
        help='Reset database (drops all data!)'
    )
    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        help='Backup database'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show database info'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old data'
    )
    parser.add_argument(
        '--cleanup-days',
        type=int,
        default=90,
        help='Days to keep when cleaning up (default: 90)'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EchoMind - Database Setup")
    print("=" * 60)
    
    # Create Flask app
    from backend.app import create_app
    from backend.config import get_config
    
    app = create_app(get_config())
    
    # Default action: init + seed
    if not any([args.init, args.seed, args.reset, args.backup, args.info, args.cleanup]):
        args.init = True
        args.seed = True
    
    # Execute requested actions
    if args.reset:
        reset_database(app, confirm=args.yes)
    
    if args.init:
        init_database(app)
    
    if args.seed:
        seed_demo_data(app)
    
    if args.backup:
        backup_database(app)
    
    if args.info:
        show_database_info(app)
    
    if args.cleanup:
        run_cleanup(app, days=args.cleanup_days)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()