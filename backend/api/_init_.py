# ============================================
# EchoMind - API Package Initialization
# ============================================

"""
REST API Package

This package contains all API route blueprints for EchoMind:

- electricity_routes: Electricity-specific endpoints
- water_routes: Water-specific endpoints
- prediction_routes: Unified prediction endpoints
- user_routes: User management and data endpoints
- dashboard_routes: Dashboard and overview endpoints

All routes return JSON responses and follow RESTful conventions.
"""

from backend.api.electricity_routes import electricity_bp
from backend.api.water_routes import water_bp
from backend.api.prediction_routes import prediction_bp
from backend.api.user_routes import user_bp
from backend.api.dashboard_routes import dashboard_bp

# Package exports
__all__ = [
    "electricity_bp",
    "water_bp",
    "prediction_bp",
    "user_bp",
    "dashboard_bp",
]

# API version
API_VERSION = "v1"

# Common response helpers
def success_response(data: dict, message: str = "Success", status_code: int = 200):
    """
    Create a standardized success response.

    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code

    Returns:
        Tuple of (response_dict, status_code)
    """
    return {
        "success": True,
        "message": message,
        "data": data,
    }, status_code


def error_response(message: str, status_code: int = 400, errors: list = None):
    """
    Create a standardized error response.

    Args:
        message: Error message
        status_code: HTTP status code
        errors: List of specific errors

    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        "success": False,
        "message": message,
    }
    if errors:
        response["errors"] = errors
    return response, status_code


def paginate_response(
    items: list,
    page: int,
    per_page: int,
    total: int,
    message: str = "Success"
):
    """
    Create a paginated response.

    Args:
        items: List of items for current page
        page: Current page number
        per_page: Items per page
        total: Total number of items
        message: Success message

    Returns:
        Paginated response dictionary
    """
    total_pages = (total + per_page - 1) // per_page
    
    return {
        "success": True,
        "message": message,
        "data": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }