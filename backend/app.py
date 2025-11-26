# ============================================
# EchoMind - Main Flask Application
# ============================================

"""
Main application entry point for EchoMind.
Initializes Flask app, registers blueprints, and configures extensions.
"""

import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS

from backend.config import get_config, Config
from backend.database import db, init_db
from backend.utils.logger import setup_logger, logger

# Import API blueprints
from backend.api.electricity_routes import electricity_bp
from backend.api.water_routes import water_bp
from backend.api.prediction_routes import prediction_bp
from backend.api.user_routes import user_bp
from backend.api.dashboard_routes import dashboard_bp


def create_app(config_class=None):
    """
    Application factory for creating Flask app instance.

    Args:
        config_class: Configuration class to use. Defaults to environment-based config.

    Returns:
        Flask: Configured Flask application instance.
    """
    # Get configuration
    if config_class is None:
        config_class = get_config()

    # Create Flask app with custom template and static folders
    app = Flask(
        __name__,
        template_folder=str(Config.TEMPLATE_FOLDER),
        static_folder=str(Config.STATIC_FOLDER),
    )

    # Load configuration
    app.config.from_object(config_class)

    # Initialize configuration
    config_class.init_app(app)

    # Setup logging
    setup_logger(app)
    logger.info("Starting EchoMind application...")

    # Initialize extensions
    init_extensions(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Register frontend routes
    register_frontend_routes(app)

    logger.info("EchoMind application initialized successfully!")
    return app


def init_extensions(app):
    """Initialize Flask extensions."""
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize database
    init_db(app)

    logger.info("Extensions initialized")


def register_blueprints(app):
    """Register API blueprints."""
    api_prefix = app.config.get("API_PREFIX", "/api")

    # Register API routes
    app.register_blueprint(electricity_bp, url_prefix=f"{api_prefix}/electricity")
    app.register_blueprint(water_bp, url_prefix=f"{api_prefix}/water")
    app.register_blueprint(prediction_bp, url_prefix=f"{api_prefix}/predict")
    app.register_blueprint(user_bp, url_prefix=f"{api_prefix}/user")
    app.register_blueprint(dashboard_bp, url_prefix=f"{api_prefix}/dashboard")

    logger.info("Blueprints registered")


def register_error_handlers(app):
    """Register error handlers."""

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": "Bad Request",
            "message": str(error.description)
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Not Found",
            "message": "The requested resource was not found"
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }), 500

    logger.info("Error handlers registered")


def register_frontend_routes(app):
    """Register frontend template routes."""

    @app.route("/")
    def index():
        """Redirect to dashboard."""
        return render_template("dashboard.html")

    @app.route("/dashboard")
    def dashboard():
        """Dashboard page."""
        return render_template("dashboard.html")

    @app.route("/input")
    def user_input():
        """User input page."""
        return render_template("user_input.html")

    @app.route("/predictions")
    def predictions():
        """Predictions page."""
        return render_template("predictions.html")

    @app.route("/electricity")
    def electricity():
        """Electricity-specific page."""
        return render_template("electricity.html")

    @app.route("/water")
    def water():
        """Water-specific page."""
        return render_template("water.html")

    @app.route("/health")
    def health_check():
        """Health check endpoint for Docker/Kubernetes."""
        return jsonify({
            "status": "healthy",
            "service": "EchoMind",
            "version": "1.0.0"
        }), 200

    logger.info("Frontend routes registered")


# Create app instance only when appropriate.
# Avoid creating the app at import time so utility scripts can import
# backend modules without triggering server initialization. The Flask
# CLI sets FLASK_RUN_FROM_CLI in the environment when running `flask run`.
app = None
if __name__ == "__main__" or os.getenv("FLASK_RUN_FROM_CLI"):
    app = create_app()


if __name__ == "__main__":
    # Run the application when executed directly
    app.run(
        host=app.config.get("HOST", "0.0.0.0"),
        port=app.config.get("PORT", 5000),
        debug=app.config.get("DEBUG", True)
    )