# ============================================
# EchoMind - User API Routes
# ============================================

"""
API routes for user management and data operations.
Handles user profiles, usage history, and preferences.
"""

from flask import Blueprint, request, jsonify, session
from datetime import datetime, timedelta
from typing import Any, Dict, List

from backend.services.user_service import UserService
from backend.database.models import User, UsageHistory
from backend.utils.validators import (
    validate_user_data,
    validate_usage_entry,
    ValidationError,
)
from backend.utils.logger import logger

# Create blueprint
user_bp = Blueprint("user", __name__)

# Initialize service
user_service = UserService()


@user_bp.route("/", methods=["GET"])
def user_index():
    """
    Get user module information.
    
    Returns:
        JSON with module information and available endpoints
    """
    return jsonify({
        "success": True,
        "module": "user",
        "description": "User management and data storage",
        "endpoints": {
            "GET /": "Module information",
            "POST /register": "Register new user",
            "POST /login": "User login (session-based)",
            "GET /profile": "Get user profile",
            "PUT /profile": "Update user profile",
            "POST /save-usage": "Save usage data entry",
            "GET /history": "Get usage history",
            "DELETE /history/<id>": "Delete usage entry",
            "GET /preferences": "Get user preferences",
            "PUT /preferences": "Update user preferences",
            "GET /summary": "Get user consumption summary",
        },
    }), 200


@user_bp.route("/register", methods=["POST"])
def register_user():
    """
    Register a new user.
    
    Request Body:
        {
            "email": "user@example.com",
            "name": "John Doe",
            "password": "securepassword",
            "household_size": 4,  // optional
            "location": "New York"  // optional
        }
    
    Returns:
        JSON with user registration result
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Validate user data
        try:
            validated = validate_user_data(data, registration=True)
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e),
                "details": e.errors if hasattr(e, "errors") else None
            }), 400

        # Create user
        result = user_service.create_user(
            email=validated["email"],
            name=validated["name"],
            password=validated["password"],
            household_size=validated.get("household_size"),
            location=validated.get("location")
        )

        if result.get("success"):
            logger.info(f"New user registered: {validated['email']}")
            return jsonify(result), 201
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error in user registration: {e}")
        return jsonify({
            "success": False,
            "error": "Registration failed",
            "message": str(e)
        }), 500


@user_bp.route("/login", methods=["POST"])
def login_user():
    """
    User login with session creation.
    
    Request Body:
        {
            "email": "user@example.com",
            "password": "securepassword"
        }
    
    Returns:
        JSON with login result
    """
    try:
        data = request.get_json()
        
        if not data or "email" not in data or "password" not in data:
            return jsonify({
                "success": False,
                "error": "Email and password required"
            }), 400

        # Authenticate user
        result = user_service.authenticate_user(
            email=data["email"],
            password=data["password"]
        )

        if result.get("success"):
            # Create session
            session["user_id"] = result["user"]["id"]
            session["email"] = result["user"]["email"]
            session.permanent = True
            
            logger.info(f"User logged in: {data['email']}")
            return jsonify(result), 200
        else:
            return jsonify(result), 401

    except Exception as e:
        logger.error(f"Error in user login: {e}")
        return jsonify({
            "success": False,
            "error": "Login failed",
            "message": str(e)
        }), 500


@user_bp.route("/logout", methods=["POST"])
def logout_user():
    """
    User logout and session destruction.
    
    Returns:
        JSON with logout confirmation
    """
    try:
        email = session.get("email", "Unknown")
        session.clear()
        logger.info(f"User logged out: {email}")
        
        return jsonify({
            "success": True,
            "message": "Logged out successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error in user logout: {e}")
        return jsonify({
            "success": False,
            "error": "Logout failed",
            "message": str(e)
        }), 500


@user_bp.route("/profile", methods=["GET"])
def get_profile():
    """
    Get current user's profile.
    
    Returns:
        JSON with user profile
    """
    try:
        user_id = session.get("user_id")
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "Not authenticated",
                "message": "Please login to access your profile"
            }), 401

        profile = user_service.get_user_profile(user_id)
        
        if profile:
            return jsonify({
                "success": True,
                "profile": profile
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Profile not found"
            }), 404

    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get profile",
            "message": str(e)
        }), 500


@user_bp.route("/profile", methods=["PUT"])
def update_profile():
    """
    Update current user's profile.
    
    Request Body:
        {
            "name": "New Name",  // optional
            "household_size": 5,  // optional
            "location": "Los Angeles"  // optional
        }
    
    Returns:
        JSON with updated profile
    """
    try:
        user_id = session.get("user_id")
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "Not authenticated"
            }), 401

        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No update data provided"
            }), 400

        result = user_service.update_user_profile(user_id, data)
        
        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to update profile",
            "message": str(e)
        }), 500


@user_bp.route("/save-usage", methods=["POST"])
def save_usage():
    """
    Save user consumption data entry.
    
    Request Body:
        {
            "resource_type": "electricity",  // or "water"
            "consumption": 150.5,
            "datetime": "2024-01-15T14:30:00",  // optional, defaults to now
            "notes": "High usage day"  // optional
        }
    
    Returns:
        JSON with saved entry confirmation
    """
    try:
        # Allow both authenticated and anonymous usage
        user_id = session.get("user_id")
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Validate usage entry
        try:
            validated = validate_usage_entry(data)
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e)
            }), 400

        # Save usage entry
        result = user_service.save_usage_entry(
            user_id=user_id,
            resource_type=validated["resource_type"],
            consumption=validated["consumption"],
            datetime_str=validated.get("datetime"),
            notes=validated.get("notes")
        )

        if result.get("success"):
            return jsonify(result), 201
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error saving usage: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to save usage",
            "message": str(e)
        }), 500


@user_bp.route("/save-usage/bulk", methods=["POST"])
def save_usage_bulk():
    """
    Save multiple usage entries at once.
    
    Request Body:
        {
            "entries": [
                {
                    "resource_type": "electricity",
                    "consumption": 150.5,
                    "datetime": "2024-01-15T14:30:00"
                },
                ...
            ]
        }
    
    Returns:
        JSON with bulk save results
    """
    try:
        user_id = session.get("user_id")
        data = request.get_json()
        
        if not data or "entries" not in data:
            return jsonify({
                "success": False,
                "error": "No entries provided"
            }), 400

        entries = data["entries"]
        
        if not isinstance(entries, list) or len(entries) == 0:
            return jsonify({
                "success": False,
                "error": "Entries must be a non-empty array"
            }), 400

        # Limit bulk entries
        if len(entries) > 1000:
            return jsonify({
                "success": False,
                "error": "Too many entries",
                "message": "Maximum 1000 entries per request"
            }), 400

        result = user_service.save_usage_bulk(user_id, entries)
        
        return jsonify(result), 200 if result.get("success") else 400

    except Exception as e:
        logger.error(f"Error in bulk save: {e}")
        return jsonify({
            "success": False,
            "error": "Bulk save failed",
            "message": str(e)
        }), 500


@user_bp.route("/history", methods=["GET"])
def get_history():
    """
    Get user's usage history.
    
    Query Parameters:
        resource_type (str): Filter by resource (electricity/water)
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        limit (int): Maximum entries to return (default: 100)
        offset (int): Offset for pagination (default: 0)
    
    Returns:
        JSON with usage history
    """
    try:
        user_id = session.get("user_id")
        
        # Get query parameters
        resource_type = request.args.get("resource_type")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)

        # Limit maximum results
        limit = min(limit, 1000)

        result = user_service.get_usage_history(
            user_id=user_id,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get history",
            "message": str(e)
        }), 500


@user_bp.route("/history/<int:entry_id>", methods=["DELETE"])
def delete_history_entry(entry_id: int):
    """
    Delete a specific usage history entry.
    
    Args:
        entry_id: ID of the entry to delete
    
    Returns:
        JSON with deletion confirmation
    """
    try:
        user_id = session.get("user_id")
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "Not authenticated"
            }), 401

        result = user_service.delete_usage_entry(user_id, entry_id)
        
        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 404

    except Exception as e:
        logger.error(f"Error deleting entry: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to delete entry",
            "message": str(e)
        }), 500


@user_bp.route("/preferences", methods=["GET"])
def get_preferences():
    """
    Get user preferences.
    
    Returns:
        JSON with user preferences
    """
    try:
        user_id = session.get("user_id")
        
        if not user_id:
            # Return default preferences for anonymous users
            return jsonify({
                "success": True,
                "preferences": user_service.get_default_preferences()
            }), 200

        preferences = user_service.get_user_preferences(user_id)
        
        return jsonify({
            "success": True,
            "preferences": preferences
        }), 200

    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get preferences",
            "message": str(e)
        }), 500


@user_bp.route("/preferences", methods=["PUT"])
def update_preferences():
    """
    Update user preferences.
    
    Request Body:
        {
            "electricity_rate": 0.15,
            "water_rate": 0.003,
            "currency": "USD",
            "notifications_enabled": true,
            "prediction_periods": 48
        }
    
    Returns:
        JSON with updated preferences
    """
    try:
        user_id = session.get("user_id")
        
        if not user_id:
            return jsonify({
                "success": False,
                "error": "Not authenticated"
            }), 401

        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No preferences provided"
            }), 400

        result = user_service.update_user_preferences(user_id, data)
        
        return jsonify(result), 200 if result.get("success") else 400

    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to update preferences",
            "message": str(e)
        }), 500


@user_bp.route("/summary", methods=["GET"])
def get_summary():
    """
    Get user's consumption summary.
    
    Query Parameters:
        period (str): Summary period (week/month/year, default: month)
    
    Returns:
        JSON with consumption summary
    """
    try:
        user_id = session.get("user_id")
        period = request.args.get("period", "month")
        
        if period not in ["week", "month", "year"]:
            return jsonify({
                "success": False,
                "error": "Invalid period",
                "message": "Period must be 'week', 'month', or 'year'"
            }), 400

        summary = user_service.get_consumption_summary(user_id, period)
        
        return jsonify({
            "success": True,
            "summary": summary,
            "period": period
        }), 200

    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get summary",
            "message": str(e)
        }), 500