# ============================================
# EchoMind - API Tests
# ============================================

"""
Tests for API endpoints.

Tests cover:
    - Electricity API routes
    - Water API routes
    - Prediction API routes
    - User API routes
    - Dashboard API routes
    - Error handling
    - Input validation
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import test configuration
from tests import get_test_config, TEST_USER_EMAIL, TEST_USER_PASSWORD, TEST_USER_NAME

# Import Flask app
from backend.app import create_app
from backend.database import db
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
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_client(client):
    """Create authenticated test client."""
    # Register user
    client.post('/api/user/register', json={
        'email': TEST_USER_EMAIL,
        'password': TEST_USER_PASSWORD,
        'name': TEST_USER_NAME
    })
    
    # Login
    client.post('/api/user/login', json={
        'email': TEST_USER_EMAIL,
        'password': TEST_USER_PASSWORD
    })
    
    return client


@pytest.fixture
def sample_electricity_data():
    """Generate sample electricity data for API tests."""
    data = []
    base_time = datetime.now() - timedelta(days=3)
    
    for i in range(72):  # 3 days
        timestamp = base_time + timedelta(hours=i)
        hour = timestamp.hour
        
        if 6 <= hour <= 9:
            value = 2.0
        elif 17 <= hour <= 21:
            value = 2.5
        elif 0 <= hour <= 5:
            value = 0.5
        else:
            value = 1.2
        
        data.append({
            'datetime': timestamp.isoformat(),
            'consumption': round(value + (i % 10) * 0.1, 2)
        })
    
    return data


@pytest.fixture
def sample_water_data():
    """Generate sample water data for API tests."""
    data = []
    base_time = datetime.now() - timedelta(days=3)
    
    for i in range(72):  # 3 days
        timestamp = base_time + timedelta(hours=i)
        hour = timestamp.hour
        
        if 6 <= hour <= 9:
            value = 45
        elif 18 <= hour <= 22:
            value = 40
        elif 0 <= hour <= 5:
            value = 5
        else:
            value = 15
        
        data.append({
            'datetime': timestamp.isoformat(),
            'consumption': round(value + (i % 10), 1)
        })
    
    return data


# ===========================================
# Health Check Tests
# ===========================================

class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns OK."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'service' in data


# ===========================================
# Electricity API Tests
# ===========================================

class TestElectricityAPI:
    """Tests for electricity API endpoints."""
    
    def test_electricity_index(self, client):
        """Test electricity module info endpoint."""
        response = client.get('/api/electricity/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert data['module'] == 'electricity'
        assert 'endpoints' in data
    
    def test_electricity_predict(self, client, sample_electricity_data):
        """Test electricity prediction endpoint."""
        response = client.post(
            '/api/electricity/predict',
            json={
                'data': sample_electricity_data,
                'periods': 24,
                'frequency': 'H'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'predictions' in data
        assert len(data['predictions']) == 24
    
    def test_electricity_predict_no_data(self, client):
        """Test electricity prediction with no data."""
        response = client.post(
            '/api/electricity/predict',
            json={},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
    
    def test_electricity_predict_insufficient_data(self, client):
        """Test electricity prediction with insufficient data."""
        response = client.post(
            '/api/electricity/predict',
            json={
                'data': [{'datetime': '2024-01-01T00:00:00', 'consumption': 1.5}],
                'periods': 24
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
    
    def test_electricity_demo_prediction(self, client):
        """Test electricity demo prediction endpoint."""
        response = client.get('/api/electricity/predict/demo?periods=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data or 'demo' in data
    
    def test_electricity_recommendations(self, client):
        """Test electricity recommendations endpoint."""
        response = client.get('/api/electricity/recommendations')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'recommendations' in data
        assert isinstance(data['recommendations'], list)
    
    def test_electricity_cost_estimate(self, client):
        """Test electricity cost estimate endpoint."""
        response = client.post(
            '/api/electricity/cost-estimate',
            json={'consumption_kwh': 500},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'cost_estimate' in data
        assert 'total_cost' in data['cost_estimate']
    
    def test_electricity_peak_hours(self, client):
        """Test electricity peak hours endpoint."""
        response = client.get('/api/electricity/peak-hours')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'peak_hours' in data
        assert 'off_peak_hours' in data
    
    def test_electricity_analyze(self, client, sample_electricity_data):
        """Test electricity analyze endpoint."""
        response = client.post(
            '/api/electricity/analyze',
            json={'data': sample_electricity_data},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'analysis' in data


# ===========================================
# Water API Tests
# ===========================================

class TestWaterAPI:
    """Tests for water API endpoints."""
    
    def test_water_index(self, client):
        """Test water module info endpoint."""
        response = client.get('/api/water/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert data['module'] == 'water'
    
    def test_water_predict(self, client, sample_water_data):
        """Test water prediction endpoint."""
        response = client.post(
            '/api/water/predict',
            json={
                'data': sample_water_data,
                'periods': 24,
                'frequency': 'H'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'predictions' in data
    
    def test_water_leak_detection(self, client, sample_water_data):
        """Test water leak detection endpoint."""
        response = client.post(
            '/api/water/leak-detection',
            json={
                'data': sample_water_data,
                'sensitivity': 'medium'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'leak_detection' in data
    
    def test_water_leak_detection_invalid_sensitivity(self, client, sample_water_data):
        """Test water leak detection with invalid sensitivity."""
        response = client.post(
            '/api/water/leak-detection',
            json={
                'data': sample_water_data,
                'sensitivity': 'invalid'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_water_recommendations(self, client):
        """Test water recommendations endpoint."""
        response = client.get('/api/water/recommendations')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'recommendations' in data
    
    def test_water_cost_estimate(self, client):
        """Test water cost estimate endpoint."""
        response = client.post(
            '/api/water/cost-estimate',
            json={'consumption_liters': 5000},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'cost_estimate' in data
    
    def test_water_usage_patterns(self, client):
        """Test water usage patterns endpoint."""
        response = client.get('/api/water/usage-patterns')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'usage_patterns' in data
        assert 'benchmarks' in data


# ===========================================
# Prediction API Tests
# ===========================================

class TestPredictionAPI:
    """Tests for unified prediction API endpoints."""
    
    def test_prediction_index(self, client):
        """Test prediction module info endpoint."""
        response = client.get('/api/predict/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert data['module'] == 'prediction'
    
    def test_predict_electricity(self, client, sample_electricity_data):
        """Test unified electricity prediction endpoint."""
        response = client.post(
            '/api/predict/electricity',
            json={
                'data': sample_electricity_data,
                'periods': 24
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_predict_water(self, client, sample_water_data):
        """Test unified water prediction endpoint."""
        response = client.post(
            '/api/predict/water',
            json={
                'data': sample_water_data,
                'periods': 24
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_predict_both(self, client, sample_electricity_data, sample_water_data):
        """Test combined prediction endpoint."""
        response = client.post(
            '/api/predict/both',
            json={
                'electricity': {
                    'data': sample_electricity_data,
                    'periods': 24
                },
                'water': {
                    'data': sample_water_data,
                    'periods': 24
                }
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'resources' in data
    
    def test_predict_demo(self, client):
        """Test demo prediction endpoint."""
        response = client.get('/api/predict/demo?resource=both&periods=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert data['demo'] == True
    
    def test_sustainability_score(self, client):
        """Test sustainability score endpoint."""
        response = client.post(
            '/api/predict/sustainability-score',
            json={
                'electricity': {'daily_consumption_kwh': 25},
                'water': {'daily_consumption_liters': 300},
                'household_size': 4
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'sustainability' in data
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get('/api/predict/model-info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'models' in data
    
    def test_quick_predict(self, client):
        """Test quick predict endpoint."""
        response = client.post(
            '/api/predict/quick-predict',
            json={
                'resource_type': 'electricity',
                'recent_average': 1.5,
                'periods': 24
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert len(data['predictions']) == 24


# ===========================================
# User API Tests
# ===========================================

class TestUserAPI:
    """Tests for user API endpoints."""
    
    def test_user_index(self, client):
        """Test user module info endpoint."""
        response = client.get('/api/user/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert data['module'] == 'user'
    
    def test_user_register(self, client):
        """Test user registration endpoint."""
        response = client.post(
            '/api/user/register',
            json={
                'email': 'newuser@test.com',
                'password': 'password123',
                'name': 'New User',
                'household_size': 4
            },
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'user' in data
    
    def test_user_register_duplicate_email(self, client):
        """Test registration with duplicate email."""
        # First registration
        client.post(
            '/api/user/register',
            json={
                'email': 'duplicate@test.com',
                'password': 'password123',
                'name': 'User 1'
            },
            content_type='application/json'
        )
        
        # Second registration with same email
        response = client.post(
            '/api/user/register',
            json={
                'email': 'duplicate@test.com',
                'password': 'password123',
                'name': 'User 2'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
    
    def test_user_register_invalid_email(self, client):
        """Test registration with invalid email."""
        response = client.post(
            '/api/user/register',
            json={
                'email': 'invalid-email',
                'password': 'password123',
                'name': 'User'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_user_login(self, client):
        """Test user login endpoint."""
        # Register first
        client.post(
            '/api/user/register',
            json={
                'email': 'login@test.com',
                'password': 'password123',
                'name': 'Login User'
            },
            content_type='application/json'
        )
        
        # Login
        response = client.post(
            '/api/user/login',
            json={
                'email': 'login@test.com',
                'password': 'password123'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_user_login_wrong_password(self, client):
        """Test login with wrong password."""
        # Register first
        client.post(
            '/api/user/register',
            json={
                'email': 'wrongpass@test.com',
                'password': 'password123',
                'name': 'User'
            },
            content_type='application/json'
        )
        
        # Login with wrong password
        response = client.post(
            '/api/user/login',
            json={
                'email': 'wrongpass@test.com',
                'password': 'wrongpassword'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 401
    
    def test_user_logout(self, auth_client):
        """Test user logout endpoint."""
        response = auth_client.post('/api/user/logout')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_user_save_usage(self, client):
        """Test save usage endpoint."""
        response = client.post(
            '/api/user/save-usage',
            json={
                'resource_type': 'electricity',
                'consumption': 1.5,
                'datetime': datetime.now().isoformat()
            },
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_user_history(self, client):
        """Test get history endpoint."""
        # Save some data first
        client.post(
            '/api/user/save-usage',
            json={
                'resource_type': 'electricity',
                'consumption': 1.5
            },
            content_type='application/json'
        )
        
        response = client.get('/api/user/history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'entries' in data
    
    def test_user_preferences(self, client):
        """Test get preferences endpoint."""
        response = client.get('/api/user/preferences')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'preferences' in data


# ===========================================
# Dashboard API Tests
# ===========================================

class TestDashboardAPI:
    """Tests for dashboard API endpoints."""
    
    def test_dashboard_index(self, client):
        """Test dashboard module info endpoint."""
        response = client.get('/api/dashboard/')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert data['module'] == 'dashboard'
    
    def test_dashboard_overview(self, client):
        """Test dashboard overview endpoint."""
        response = client.get('/api/dashboard/overview?period=today')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'overview' in data
    
    def test_dashboard_stats(self, client):
        """Test dashboard stats endpoint."""
        response = client.get('/api/dashboard/stats?period=month&resource=both')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'statistics' in data
    
    def test_dashboard_recent(self, client):
        """Test dashboard recent activity endpoint."""
        response = client.get('/api/dashboard/recent?limit=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'recent_activity' in data
    
    def test_dashboard_predictions_summary(self, client):
        """Test dashboard predictions summary endpoint."""
        response = client.get('/api/dashboard/predictions-summary?periods=24')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'predictions_summary' in data
    
    def test_dashboard_cost_summary(self, client):
        """Test dashboard cost summary endpoint."""
        response = client.get('/api/dashboard/cost-summary?period=month')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'cost_summary' in data
    
    def test_dashboard_sustainability(self, client):
        """Test dashboard sustainability endpoint."""
        response = client.get('/api/dashboard/sustainability')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'sustainability' in data
    
    def test_dashboard_alerts(self, client):
        """Test dashboard alerts endpoint."""
        response = client.get('/api/dashboard/alerts')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'alerts' in data
    
    def test_dashboard_tips(self, client):
        """Test dashboard tips endpoint."""
        response = client.get('/api/dashboard/tips?category=all&limit=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'tips' in data


# ===========================================
# Error Handling Tests
# ===========================================

class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get('/api/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] == False
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling."""
        response = client.post(
            '/api/electricity/predict',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code in [400, 500]
    
    def test_missing_required_fields(self, client):
        """Test missing required fields handling."""
        response = client.post(
            '/api/user/register',
            json={'email': 'test@test.com'},  # Missing password and name
            content_type='application/json'
        )
        
        assert response.status_code == 400


# ===========================================
# CORS Tests
# ===========================================

class TestCORS:
    """Tests for CORS headers."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options('/api/electricity/')
        
        # Should allow the request
        assert response.status_code in [200, 204]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])