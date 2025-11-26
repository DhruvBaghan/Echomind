# ============================================
# EchoMind - Database Models
# ============================================

"""
SQLAlchemy ORM models for EchoMind.

Models:
    - User: User accounts and profiles
    - UsageHistory: Historical consumption data
    - Preference: User preferences and settings
    - Prediction: Stored predictions for analysis
"""

from datetime import datetime
from typing import Optional, Dict, Any

from werkzeug.security import generate_password_hash, check_password_hash

from backend.database import db


class User(db.Model):
    """
    User model for authentication and profile management.
    """
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    household_size = db.Column(db.Integer, default=4)
    location = db.Column(db.String(255), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationships
    usage_history = db.relationship(
        "UsageHistory",
        backref="user",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )
    preferences = db.relationship(
        "Preference",
        backref="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    predictions = db.relationship(
        "Prediction",
        backref="user",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    def __init__(
        self,
        email: str,
        password: str,
        name: str,
        household_size: int = 4,
        location: Optional[str] = None
    ):
        """Initialize user with hashed password."""
        self.email = email.lower()
        self.set_password(password)
        self.name = name
        self.household_size = household_size
        self.location = location

    def set_password(self, password: str) -> None:
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify password against hash."""
        return check_password_hash(self.password_hash, password)

    def update_last_login(self) -> None:
        """Update last login timestamp."""
        self.last_login = datetime.utcnow()

    def to_dict(self, include_email: bool = True) -> Dict[str, Any]:
        """
        Convert user to dictionary.
        
        Args:
            include_email: Whether to include email in output
            
        Returns:
            Dictionary representation of user
        """
        data = {
            "id": self.id,
            "name": self.name,
            "household_size": self.household_size,
            "location": self.location,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_email:
            data["email"] = self.email
        return data

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class UsageHistory(db.Model):
    """
    Model for storing historical consumption data.
    """
    __tablename__ = "usage_history"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    resource_type = db.Column(
        db.String(50),
        nullable=False,
        index=True
    )  # 'electricity' or 'water'
    consumption = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=False)  # 'kWh' or 'liters'
    recorded_at = db.Column(db.DateTime, nullable=False, index=True)
    notes = db.Column(db.Text, nullable=True)
    source = db.Column(db.String(50), default="manual")  # 'manual', 'api', 'import'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Indexes for common queries
    __table_args__ = (
        db.Index("idx_usage_user_resource", "user_id", "resource_type"),
        db.Index("idx_usage_user_date", "user_id", "recorded_at"),
    )

    def __init__(
        self,
        resource_type: str,
        consumption: float,
        recorded_at: datetime,
        user_id: Optional[int] = None,
        notes: Optional[str] = None,
        source: str = "manual"
    ):
        """Initialize usage history entry."""
        self.user_id = user_id
        self.resource_type = resource_type.lower()
        self.consumption = consumption
        self.unit = "kWh" if resource_type.lower() == "electricity" else "liters"
        self.recorded_at = recorded_at
        self.notes = notes
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "resource_type": self.resource_type,
            "consumption": self.consumption,
            "unit": self.unit,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "notes": self.notes,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def get_by_user(
        cls,
        user_id: int,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ):
        """
        Get usage history for a user with optional filters.
        
        Args:
            user_id: User ID
            resource_type: Optional resource type filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of records
            
        Returns:
            Query result
        """
        query = cls.query.filter_by(user_id=user_id)
        
        if resource_type:
            query = query.filter_by(resource_type=resource_type.lower())
        
        if start_date:
            query = query.filter(cls.recorded_at >= start_date)
        
        if end_date:
            query = query.filter(cls.recorded_at <= end_date)
        
        return query.order_by(cls.recorded_at.desc()).limit(limit).all()

    def __repr__(self) -> str:
        return f"<UsageHistory {self.resource_type}: {self.consumption} {self.unit}>"


class Preference(db.Model):
    """
    Model for storing user preferences and settings.
    """
    __tablename__ = "preferences"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    
    # Cost settings
    electricity_rate = db.Column(db.Float, default=0.12)  # USD per kWh
    water_rate = db.Column(db.Float, default=0.002)  # USD per liter
    currency = db.Column(db.String(10), default="USD")
    
    # Notification settings
    notifications_enabled = db.Column(db.Boolean, default=True)
    email_reports = db.Column(db.Boolean, default=False)
    alert_threshold_electricity = db.Column(db.Float, default=50.0)  # kWh per day
    alert_threshold_water = db.Column(db.Float, default=500.0)  # liters per day
    
    # Display settings
    prediction_periods = db.Column(db.Integer, default=24)
    theme = db.Column(db.String(20), default="light")
    language = db.Column(db.String(10), default="en")
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, user_id: int, **kwargs):
        """Initialize preferences with defaults."""
        self.user_id = user_id
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "electricity_rate": self.electricity_rate,
            "water_rate": self.water_rate,
            "currency": self.currency,
            "notifications_enabled": self.notifications_enabled,
            "email_reports": self.email_reports,
            "alert_threshold_electricity": self.alert_threshold_electricity,
            "alert_threshold_water": self.alert_threshold_water,
            "prediction_periods": self.prediction_periods,
            "theme": self.theme,
            "language": self.language,
        }

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update preferences from dictionary.
        
        Args:
            data: Dictionary with preference values
        """
        allowed_fields = [
            "electricity_rate", "water_rate", "currency",
            "notifications_enabled", "email_reports",
            "alert_threshold_electricity", "alert_threshold_water",
            "prediction_periods", "theme", "language"
        ]
        
        for field in allowed_fields:
            if field in data:
                setattr(self, field, data[field])

    @classmethod
    def get_or_create(cls, user_id: int) -> "Preference":
        """
        Get existing preferences or create default ones.
        
        Args:
            user_id: User ID
            
        Returns:
            Preference instance
        """
        pref = cls.query.filter_by(user_id=user_id).first()
        if not pref:
            pref = cls(user_id=user_id)
            db.session.add(pref)
            db.session.commit()
        return pref

    def __repr__(self) -> str:
        return f"<Preference user_id={self.user_id}>"


class Prediction(db.Model):
    """
    Model for storing prediction results for analysis and comparison.
    """
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    resource_type = db.Column(db.String(50), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False)
    periods = db.Column(db.Integer, nullable=False)
    frequency = db.Column(db.String(10), default="H")
    
    # Prediction summary
    total_predicted = db.Column(db.Float, nullable=False)
    average_predicted = db.Column(db.Float, nullable=False)
    min_predicted = db.Column(db.Float, nullable=False)
    max_predicted = db.Column(db.Float, nullable=False)
    total_cost = db.Column(db.Float, nullable=False)
    
    # Model info
    model_version = db.Column(db.String(50), default="1.0.0")
    confidence_interval = db.Column(db.Float, default=0.95)
    
    # Full predictions stored as JSON
    predictions_json = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(
        self,
        resource_type: str,
        prediction_date: datetime,
        periods: int,
        total_predicted: float,
        average_predicted: float,
        min_predicted: float,
        max_predicted: float,
        total_cost: float,
        user_id: Optional[int] = None,
        frequency: str = "H",
        predictions_json: Optional[str] = None
    ):
        """Initialize prediction record."""
        self.user_id = user_id
        self.resource_type = resource_type
        self.prediction_date = prediction_date
        self.periods = periods
        self.frequency = frequency
        self.total_predicted = total_predicted
        self.average_predicted = average_predicted
        self.min_predicted = min_predicted
        self.max_predicted = max_predicted
        self.total_cost = total_cost
        self.predictions_json = predictions_json

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "resource_type": self.resource_type,
            "prediction_date": self.prediction_date.isoformat() if self.prediction_date else None,
            "periods": self.periods,
            "frequency": self.frequency,
            "summary": {
                "total_predicted": self.total_predicted,
                "average_predicted": self.average_predicted,
                "min_predicted": self.min_predicted,
                "max_predicted": self.max_predicted,
                "total_cost": self.total_cost,
            },
            "model_version": self.model_version,
            "confidence_interval": self.confidence_interval,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def get_recent(
        cls,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        limit: int = 10
    ):
        """
        Get recent predictions.
        
        Args:
            user_id: Optional user ID filter
            resource_type: Optional resource type filter
            limit: Maximum number of records
            
        Returns:
            List of predictions
        """
        query = cls.query
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        if resource_type:
            query = query.filter_by(resource_type=resource_type.lower())
        
        return query.order_by(cls.created_at.desc()).limit(limit).all()

    def __repr__(self) -> str:
        return f"<Prediction {self.resource_type}: {self.total_predicted}>"


class Alert(db.Model):
    """
    Model for storing user alerts and notifications.
    """
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    alert_type = db.Column(db.String(50), nullable=False)  # high_usage, leak, etc.
    resource_type = db.Column(db.String(50), nullable=True)
    priority = db.Column(db.String(20), default="medium")  # low, medium, high, critical
    title = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    is_dismissed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    read_at = db.Column(db.DateTime, nullable=True)

    def __init__(
        self,
        alert_type: str,
        title: str,
        message: str,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        priority: str = "medium"
    ):
        """Initialize alert."""
        self.user_id = user_id
        self.alert_type = alert_type
        self.resource_type = resource_type
        self.priority = priority
        self.title = title
        self.message = message

    def mark_as_read(self) -> None:
        """Mark alert as read."""
        self.is_read = True
        self.read_at = datetime.utcnow()

    def dismiss(self) -> None:
        """Dismiss alert."""
        self.is_dismissed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "alert_type": self.alert_type,
            "resource_type": self.resource_type,
            "priority": self.priority,
            "title": self.title,
            "message": self.message,
            "is_read": self.is_read,
            "is_dismissed": self.is_dismissed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def get_unread(cls, user_id: int, limit: int = 20):
        """Get unread alerts for user."""
        return cls.query.filter_by(
            user_id=user_id,
            is_read=False,
            is_dismissed=False
        ).order_by(cls.created_at.desc()).limit(limit).all()

    def __repr__(self) -> str:
        return f"<Alert {self.alert_type}: {self.title}>"