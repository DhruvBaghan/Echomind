# ============================================
# EchoMind - Application Configuration
# ============================================

"""
Configuration module for EchoMind application.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration class."""

    # ----- Flask Settings -----
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # ----- Database Settings -----
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'echomind.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # ----- Cost Settings (USD) -----
    ELECTRICITY_COST_PER_KWH = float(os.getenv("ELECTRICITY_COST_PER_KWH", 0.12))
    WATER_COST_PER_LITER = float(os.getenv("WATER_COST_PER_LITER", 0.002))

    # ----- Prediction Settings -----
    DEFAULT_PREDICTION_PERIODS = int(os.getenv("DEFAULT_PREDICTION_PERIODS", 24))
    MAX_PREDICTION_PERIODS = int(os.getenv("MAX_PREDICTION_PERIODS", 168))
    MODEL_CONFIDENCE_INTERVAL = float(os.getenv("MODEL_CONFIDENCE_INTERVAL", 0.95))

    # ----- Model Paths -----
    ML_MODELS_DIR = BASE_DIR / "ml_models"
    ELECTRICITY_MODEL_PATH = os.getenv(
        "ELECTRICITY_MODEL_PATH", str(ML_MODELS_DIR / "electricity_model.pkl")
    )
    WATER_MODEL_PATH = os.getenv(
        "WATER_MODEL_PATH", str(ML_MODELS_DIR / "water_model.pkl")
    )
    MODEL_METADATA_PATH = str(ML_MODELS_DIR / "model_metadata.json")

    # ----- Dataset Paths -----
    DATASETS_DIR = BASE_DIR / "datasets"
    RAW_DATA_DIR = DATASETS_DIR / "raw"
    PROCESSED_DATA_DIR = DATASETS_DIR / "processed"

    # ----- Logging Settings -----
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = BASE_DIR / "logs"
    LOG_FILE = os.getenv("LOG_FILE", str(LOG_DIR / "echomind.log"))

    # ----- Session Settings -----
    SESSION_LIFETIME_HOURS = int(os.getenv("SESSION_LIFETIME_HOURS", 24))
    PERMANENT_SESSION_LIFETIME = SESSION_LIFETIME_HOURS * 3600

    # ----- API Settings -----
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
    API_PREFIX = "/api"

    # ----- Frontend Paths -----
    TEMPLATE_FOLDER = BASE_DIR / "frontend" / "templates"
    STATIC_FOLDER = BASE_DIR / "frontend" / "static"

    @classmethod
    def init_app(cls, app):
        """Initialize application with this configuration."""
        # Ensure required directories exist
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    SQLALCHEMY_ECHO = False

    @classmethod
    def init_app(cls, app):
        """Production-specific initialization."""
        super().init_app(app)

        # Ensure secret key is set in production
        if cls.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be set in production!")


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config():
    """Get configuration based on environment."""
    env = os.getenv("FLASK_ENV", "development")
    return config.get(env, config["default"])