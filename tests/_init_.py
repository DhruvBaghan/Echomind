# ============================================
# EchoMind - Tests Package Initialization
# ============================================

"""
Test Suite for EchoMind

This package contains all tests for the EchoMind application:

- test_predictions: Tests for prediction models and services
- test_api: Tests for API endpoints
- test_models: Tests for ML models

Running Tests:
    pytest tests/                    # Run all tests
    pytest tests/ -v                 # Verbose output
    pytest tests/ -v --cov=backend   # With coverage
    pytest tests/test_api.py         # Specific test file
    pytest tests/ -k "prediction"    # Tests matching pattern

Test Configuration:
    Tests use a separate test database and mock ML models
    to ensure isolation and fast execution.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set testing environment
os.environ['FLASK_ENV'] = 'testing'
os.environ['TESTING'] = 'True'

# Test configuration constants
TEST_DATABASE_URL = 'sqlite:///:memory:'
TEST_USER_EMAIL = 'test@echomind.io'
TEST_USER_PASSWORD = 'testpassword123'
TEST_USER_NAME = 'Test User'


def get_test_config():
    """Get test configuration dictionary."""
    return {
        'TESTING': True,
        'DEBUG': True,
        'SQLALCHEMY_DATABASE_URI': TEST_DATABASE_URL,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret-key',
        'ELECTRICITY_COST_PER_KWH': 0.12,
        'WATER_COST_PER_LITER': 0.002,
        'DEFAULT_PREDICTION_PERIODS': 24,
        'MAX_PREDICTION_PERIODS': 168,
    }


# Export commonly used test utilities
__all__ = [
    'PROJECT_ROOT',
    'TEST_DATABASE_URL',
    'TEST_USER_EMAIL',
    'TEST_USER_PASSWORD',
    'TEST_USER_NAME',
    'get_test_config',
]