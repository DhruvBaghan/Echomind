# Database package
# Import db first to avoid circular imports
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

# Now import the rest
from backend.database.models import User, UsageHistory, Preference, Prediction
from backend.database.database import init_db, get_db_session, close_db_session

__all__ = [
    'db',
    'init_db',
    'get_db_session',
    'close_db_session',
    'User',
    'UsageHistory',
    'Preference',
    'Prediction',
]


