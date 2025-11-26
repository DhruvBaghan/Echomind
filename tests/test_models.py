# ============================================
# EchoMind - Database Model Tests
# ============================================

"""
Tests for database models.

Tests cover:
    - User model
    - UsageHistory model
    - Preference model
    - Prediction model
    - Alert model
    - Model relationships
"""

import pytest
from datetime import datetime, timedelta
from werkzeug.security import check_password_hash

# Import test configuration
from tests import get_test_config, TEST_USER_EMAIL, TEST_USER_PASSWORD, TEST_USER_NAME

# Import Flask app and database
from backend.app import create_app
from backend.database import db
from backend.database.models import User, UsageHistory, Preference, Prediction, Alert
from backend.config import TestingConfig


# ===========================================
# Fixtures
# ===========================================

@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app(TestingConfig)
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def db_session(app):
    """Create database session for testing."""
    with app.app_context():
        yield db.session


@pytest.fixture
def sample_user(db_session):
    """Create a sample user for testing."""
    user = User(
        email=TEST_USER_EMAIL,
        password=TEST_USER_PASSWORD,
        name=TEST_USER_NAME,
        household_size=4,
        location='Test City'
    )
    db_session.add(user)
    db_session.commit()
    return user


# ===========================================
# User Model Tests
# ===========================================

class TestUserModel:
    """Tests for User model."""
    
    def test_create_user(self, db_session):
        """Test creating a user."""
        user = User(
            email='test@example.com',
            password='testpassword',
            name='Test User'
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.email == 'test@example.com'
        assert user.name == 'Test User'
        assert user.is_active == True
        assert user.created_at is not None
    
    def test_password_hashing(self, db_session):
        """Test password is hashed."""
        user = User(
            email='hash@example.com',
            password='plainpassword',
            name='Hash Test'
        )
        db_session.add(user)
        db_session.commit()
        
        # Password should be hashed
        assert user.password_hash != 'plainpassword'
        assert user.check_password('plainpassword') == True
        assert user.check_password('wrongpassword') == False
    
    def test_set_password(self, sample_user, db_session):
        """Test setting new password."""
        sample_user.set_password('newpassword')
        db_session.commit()
        
        assert sample_user.check_password('newpassword') == True
        assert sample_user.check_password(TEST_USER_PASSWORD) == False
    
    def test_email_lowercase(self, db_session):
        """Test email is stored lowercase."""
        user = User(
            email='TEST@EXAMPLE.COM',
            password='password',
            name='Test'
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.email == 'test@example.com'
    
    def test_unique_email(self, db_session, sample_user):
        """Test email uniqueness constraint."""
        duplicate = User(
            email=TEST_USER_EMAIL,
            password='password',
            name='Duplicate'
        )
        db_session.add(duplicate)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()
    
    def test_to_dict(self, sample_user):
        """Test user serialization."""
        data = sample_user.to_dict()
        
        assert 'id' in data
        assert 'name' in data
        assert 'email' in data
        assert 'password_hash' not in data
        assert 'household_size' in data
    
    def test_to_dict_without_email(self, sample_user):
        """Test user serialization without email."""
        data = sample_user.to_dict(include_email=False)
        
        assert 'email' not in data
    
    def test_update_last_login(self, sample_user, db_session):
        """Test updating last login."""
        assert sample_user.last_login is None
        
        sample_user.update_last_login()
        db_session.commit()
        
        assert sample_user.last_login is not None
    
    def test_default_values(self, db_session):
        """Test default values are set."""
        user = User(
            email='default@example.com',
            password='password',
            name='Default Test'
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.household_size == 4
        assert user.is_active == True
        assert user.is_verified == False
    
    def test_user_repr(self, sample_user):
        """Test user string representation."""
        repr_str = repr(sample_user)
        
        assert 'User' in repr_str
        assert sample_user.email in repr_str


# ===========================================
# UsageHistory Model Tests
# ===========================================

class TestUsageHistoryModel:
    """Tests for UsageHistory model."""
    
    def test_create_electricity_usage(self, db_session, sample_user):
        """Test creating electricity usage entry."""
        usage = UsageHistory(
            user_id=sample_user.id,
            resource_type='electricity',
            consumption=1.5,
            recorded_at=datetime.now()
        )
        db_session.add(usage)
        db_session.commit()
        
        assert usage.id is not None
        assert usage.resource_type == 'electricity'
        assert usage.unit == 'kWh'
        assert usage.consumption == 1.5
    
    def test_create_water_usage(self, db_session, sample_user):
        """Test creating water usage entry."""
        usage = UsageHistory(
            user_id=sample_user.id,
            resource_type='water',
            consumption=50.0,
            recorded_at=datetime.now()
        )
        db_session.add(usage)
        db_session.commit()
        
        assert usage.unit == 'liters'
    
    def test_usage_to_dict(self, db_session, sample_user):
        """Test usage serialization."""
        usage = UsageHistory(
            user_id=sample_user.id,
            resource_type='electricity',
            consumption=1.5,
            recorded_at=datetime.now(),
            notes='Test note'
        )
        db_session.add(usage)
        db_session.commit()
        
        data = usage.to_dict()
        
        assert 'id' in data
        assert 'resource_type' in data
        assert 'consumption' in data
        assert 'unit' in data
        assert 'notes' in data
    
    def test_get_by_user(self, db_session, sample_user):
        """Test getting usage by user."""
        # Create multiple entries
        for i in range(5):
            usage = UsageHistory(
                user_id=sample_user.id,
                resource_type='electricity',
                consumption=1.0 + i * 0.1,
                recorded_at=datetime.now() - timedelta(hours=i)
            )
            db_session.add(usage)
        db_session.commit()
        
        entries = UsageHistory.get_by_user(sample_user.id)
        
        assert len(entries) == 5
    
    def test_get_by_user_with_filter(self, db_session, sample_user):
        """Test getting usage by user with filters."""
        # Create electricity entries
        for i in range(3):
            db_session.add(UsageHistory(
                user_id=sample_user.id,
                resource_type='electricity',
                consumption=1.0,
                recorded_at=datetime.now()
            ))
        
        # Create water entries
        for i in range(2):
            db_session.add(UsageHistory(
                user_id=sample_user.id,
                resource_type='water',
                consumption=50.0,
                recorded_at=datetime.now()
            ))
        db_session.commit()
        
        electricity_entries = UsageHistory.get_by_user(
            sample_user.id,
            resource_type='electricity'
        )
        
        assert len(electricity_entries) == 3
    
    def test_cascade_delete(self, db_session, sample_user):
        """Test usage entries are deleted with user."""
        usage = UsageHistory(
            user_id=sample_user.id,
            resource_type='electricity',
            consumption=1.5,
            recorded_at=datetime.now()
        )
        db_session.add(usage)
        db_session.commit()
        
        usage_id = usage.id
        
        # Delete user
        db_session.delete(sample_user)
        db_session.commit()
        
        # Usage should be deleted
        deleted_usage = UsageHistory.query.get(usage_id)
        assert deleted_usage is None


# ===========================================
# Preference Model Tests
# ===========================================

class TestPreferenceModel:
    """Tests for Preference model."""
    
    def test_create_preference(self, db_session, sample_user):
        """Test creating preferences."""
        pref = Preference(user_id=sample_user.id)
        db_session.add(pref)
        db_session.commit()
        
        assert pref.id is not None
        assert pref.user_id == sample_user.id
    
    def test_default_values(self, db_session, sample_user):
        """Test default preference values."""
        pref = Preference(user_id=sample_user.id)
        db_session.add(pref)
        db_session.commit()
        
        assert pref.electricity_rate == 0.12
        assert pref.water_rate == 0.002
        assert pref.currency == 'USD'
        assert pref.notifications_enabled == True
        assert pref.theme == 'light'
        assert pref.language == 'en'
    
    def test_to_dict(self, db_session, sample_user):
        """Test preference serialization."""
        pref = Preference(user_id=sample_user.id)
        db_session.add(pref)
        db_session.commit()
        
        data = pref.to_dict()
        
        assert 'electricity_rate' in data
        assert 'water_rate' in data
        assert 'currency' in data
        assert 'user_id' not in data  # Should not expose user_id
    
    def test_update(self, db_session, sample_user):
        """Test updating preferences."""
        pref = Preference(user_id=sample_user.id)
        db_session.add(pref)
        db_session.commit()
        
        pref.update({
            'electricity_rate': 0.15,
            'theme': 'dark',
            'invalid_field': 'ignored'  # Should be ignored
        })
        db_session.commit()
        
        assert pref.electricity_rate == 0.15
        assert pref.theme == 'dark'
    
    def test_get_or_create(self, db_session, sample_user):
        """Test get_or_create method."""
        # First call should create
        pref1 = Preference.get_or_create(sample_user.id)
        assert pref1 is not None
        
        # Second call should return existing
        pref2 = Preference.get_or_create(sample_user.id)
        assert pref1.id == pref2.id
    
    def test_unique_user_constraint(self, db_session, sample_user):
        """Test one preference per user constraint."""
        pref1 = Preference(user_id=sample_user.id)
        db_session.add(pref1)
        db_session.commit()
        
        pref2 = Preference(user_id=sample_user.id)
        db_session.add(pref2)
        
        with pytest.raises(Exception):
            db_session.commit()


# ===========================================
# Prediction Model Tests
# ===========================================

class TestPredictionModel:
    """Tests for Prediction model."""
    
    def test_create_prediction(self, db_session, sample_user):
        """Test creating a prediction record."""
        pred = Prediction(
            user_id=sample_user.id,
            resource_type='electricity',
            prediction_date=datetime.now(),
            periods=24,
            total_predicted=36.5,
            average_predicted=1.52,
            min_predicted=0.5,
            max_predicted=3.2,
            total_cost=4.38
        )
        db_session.add(pred)
        db_session.commit()
        
        assert pred.id is not None
        assert pred.resource_type == 'electricity'
        assert pred.periods == 24
    
    def test_to_dict(self, db_session, sample_user):
        """Test prediction serialization."""
        pred = Prediction(
            user_id=sample_user.id,
            resource_type='electricity',
            prediction_date=datetime.now(),
            periods=24,
            total_predicted=36.5,
            average_predicted=1.52,
            min_predicted=0.5,
            max_predicted=3.2,
            total_cost=4.38
        )
        db_session.add(pred)
        db_session.commit()
        
        data = pred.to_dict()
        
        assert 'id' in data
        assert 'resource_type' in data
        assert 'summary' in data
        assert data['summary']['total_predicted'] == 36.5
    
    def test_get_recent(self, db_session, sample_user):
        """Test getting recent predictions."""
        for i in range(5):
            pred = Prediction(
                user_id=sample_user.id,
                resource_type='electricity',
                prediction_date=datetime.now(),
                periods=24,
                total_predicted=36.5,
                average_predicted=1.52,
                min_predicted=0.5,
                max_predicted=3.2,
                total_cost=4.38
            )
            db_session.add(pred)
        db_session.commit()
        
        recent = Prediction.get_recent(user_id=sample_user.id, limit=3)
        
        assert len(recent) == 3


# ===========================================
# Alert Model Tests
# ===========================================

class TestAlertModel:
    """Tests for Alert model."""
    
    def test_create_alert(self, db_session, sample_user):
        """Test creating an alert."""
        alert = Alert(
            user_id=sample_user.id,
            alert_type='high_usage',
            resource_type='electricity',
            priority='high',
            title='High Usage Alert',
            message='Your electricity usage is above normal.'
        )
        db_session.add(alert)
        db_session.commit()
        
        assert alert.id is not None
        assert alert.is_read == False
        assert alert.is_dismissed == False
    
    def test_mark_as_read(self, db_session, sample_user):
        """Test marking alert as read."""
        alert = Alert(
            user_id=sample_user.id,
            alert_type='info',
            title='Test',
            message='Test message'
        )
        db_session.add(alert)
        db_session.commit()
        
        assert alert.is_read == False
        assert alert.read_at is None
        
        alert.mark_as_read()
        db_session.commit()
        
        assert alert.is_read == True
        assert alert.read_at is not None
    
    def test_dismiss(self, db_session, sample_user):
        """Test dismissing alert."""
        alert = Alert(
            user_id=sample_user.id,
            alert_type='info',
            title='Test',
            message='Test message'
        )
        db_session.add(alert)
        db_session.commit()
        
        alert.dismiss()
        db_session.commit()
        
        assert alert.is_dismissed == True
    
    def test_to_dict(self, db_session, sample_user):
        """Test alert serialization."""
        alert = Alert(
            user_id=sample_user.id,
            alert_type='leak',
            resource_type='water',
            priority='critical',
            title='Leak Detected',
            message='Possible water leak detected.'
        )
        db_session.add(alert)
        db_session.commit()
        
        data = alert.to_dict()
        
        assert 'id' in data
        assert 'alert_type' in data
        assert 'priority' in data
        assert 'title' in data
        assert 'message' in data
    
    def test_get_unread(self, db_session, sample_user):
        """Test getting unread alerts."""
        # Create read alert
        read_alert = Alert(
            user_id=sample_user.id,
            alert_type='info',
            title='Read',
            message='Read alert'
        )
        read_alert.is_read = True
        db_session.add(read_alert)
        
        # Create unread alerts
        for i in range(3):
            unread = Alert(
                user_id=sample_user.id,
                alert_type='info',
                title=f'Unread {i}',
                message=f'Unread alert {i}'
            )
            db_session.add(unread)
        db_session.commit()
        
        unread_alerts = Alert.get_unread(sample_user.id)
        
        assert len(unread_alerts) == 3
    
    def test_priority_values(self, db_session, sample_user):
        """Test valid priority values."""
        valid_priorities = ['low', 'medium', 'high', 'critical']
        
        for priority in valid_priorities:
            alert = Alert(
                user_id=sample_user.id,
                alert_type='test',
                priority=priority,
                title='Test',
                message='Test'
            )
            db_session.add(alert)
        
        db_session.commit()
        
        alerts = Alert.query.filter_by(user_id=sample_user.id).all()
        assert len(alerts) == 4


# ===========================================
# Relationship Tests
# ===========================================

class TestModelRelationships:
    """Tests for model relationships."""
    
    def test_user_usage_relationship(self, db_session, sample_user):
        """Test User to UsageHistory relationship."""
        usage = UsageHistory(
            user_id=sample_user.id,
            resource_type='electricity',
            consumption=1.5,
            recorded_at=datetime.now()
        )
        db_session.add(usage)
        db_session.commit()
        
        # Access through relationship
        assert sample_user.usage_history.count() == 1
    
    def test_user_preferences_relationship(self, db_session, sample_user):
        """Test User to Preference relationship."""
        pref = Preference(user_id=sample_user.id)
        db_session.add(pref)
        db_session.commit()
        
        # Access through relationship
        assert sample_user.preferences is not None
        assert sample_user.preferences.id == pref.id
    
    def test_user_predictions_relationship(self, db_session, sample_user):
        """Test User to Prediction relationship."""
        pred = Prediction(
            user_id=sample_user.id,
            resource_type='electricity',
            prediction_date=datetime.now(),
            periods=24,
            total_predicted=36.5,
            average_predicted=1.52,
            min_predicted=0.5,
            max_predicted=3.2,
            total_cost=4.38
        )
        db_session.add(pred)
        db_session.commit()
        
        # Access through relationship
        assert sample_user.predictions.count() == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])