# ============================================
# EchoMind - Backend Package Initialization
# ============================================

"""
EchoMind Backend Package

This package contains the core backend functionality for the EchoMind
resource consumption prediction system. It includes:

- Flask application factory
- API routes for electricity and water predictions
- ML model integration
- Database models and services
- Utility functions

Modules:
    - app: Main Flask application
    - config: Application configuration
    - models: ML prediction models
    - api: REST API routes
    - services: Business logic services
    - database: Database models and connections
    - utils: Helper utilities
"""

__version__ = "1.0.0"
__author__ = "EchoMind Team"
__description__ = "AI-Powered Resource Consumption Predictor"

# Package-level imports for convenience
from backend.config import Config

# Expose key components
__all__ = [
    "Config",
    "__version__",
    "__author__",
    "__description__",
]