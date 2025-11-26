# ============================================
# EchoMind - Electricity API Routes
# ============================================

"""
API routes for electricity-specific operations.
Handles electricity consumption predictions, analysis, and recommendations.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Any, Dict

from backend.models.electricity_predictor import ElectricityPredictor
from backend.services.electricity_service import ElectricityService
from backend.utils.validators import (
    validate_prediction_request,
    validate_consumption_data,
    ValidationError,
)
from backend.utils.logger import logger

# Create blueprint
electricity_bp = Blueprint("electricity", __name__)

# Initialize service
electricity_service = ElectricityService()


@electricity_bp.route("/", methods=["GET"])
def electricity_index():
    """
    Get electricity module information.
    
    Returns:
        JSON with module information and available endpoints
    """
    return jsonify({
        "success": True,
        "module": "electricity",
        "description": "Electricity consumption prediction and analysis",
        "unit": "kWh",
        "endpoints": {
            "GET /": "Module information",
            "POST /predict": "Generate predictions from user data",
            "GET /predict/demo": "Get demo predictions",
            "POST /analyze": "Analyze consumption patterns",
            "GET /recommendations": "Get general recommendations",
            "POST /cost-estimate": "Calculate cost estimates",
            "GET /peak-hours": "Get peak hour information",
        },
    }), 200


@electricity_bp.route("/predict", methods=["POST"])
def predict_electricity():
    """
    Generate electricity consumption predictions based on user data.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 1.5},
                {"datetime": "2024-01-01T01:00:00", "consumption": 1.2},
                ...
            ],
            "periods": 24,  // optional, default 24
            "frequency": "H"  // optional, H=hourly, D=daily
        }
    
    Returns:
        JSON with predictions and analysis
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided",
                "message": "Please provide consumption data in the request body"
            }), 400

        # Validate request
        try:
            validated = validate_prediction_request(data, resource_type="electricity")
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e),
                "details": e.errors if hasattr(e, "errors") else None
            }), 400

        # Extract parameters
        consumption_data = validated.get("data", [])
        periods = validated.get("periods", 24)
        frequency = validated.get("frequency", "H")

        # Generate predictions
        result = electricity_service.predict_from_user_data(
            user_data=consumption_data,
            periods=periods,
            frequency=frequency
        )

        if result.get("success"):
            logger.info(f"Electricity prediction generated: {periods} periods")
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error in electricity prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@electricity_bp.route("/predict/demo", methods=["GET"])
def predict_electricity_demo():
    """
    Get demo predictions using pre-trained model.
    
    Query Parameters:
        periods (int): Number of periods to predict (default: 24)
        frequency (str): Prediction frequency (default: H)
    
    Returns:
        JSON with demo predictions
    """
    try:
        periods = request.args.get("periods", 24, type=int)
        frequency = request.args.get("frequency", "H", type=str)

        # Limit periods
        periods = min(periods, 168)  # Max 7 days

        result = electricity_service.predict(
            periods=periods,
            frequency=frequency
        )

        if result.get("success"):
            result["demo"] = True
            result["note"] = "This is a demo prediction using pre-trained models"
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error in demo prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Demo prediction failed",
            "message": str(e)
        }), 500


@electricity_bp.route("/analyze", methods=["POST"])
def analyze_electricity():
    """
    Analyze electricity consumption patterns.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 1.5},
                ...
            ]
        }
    
    Returns:
        JSON with consumption analysis
    """
    try:
        data = request.get_json()
        
        if not data or "data" not in data:
            return jsonify({
                "success": False,
                "error": "No data provided",
                "message": "Please provide consumption data to analyze"
            }), 400

        # Validate consumption data
        try:
            validated_data = validate_consumption_data(
                data["data"],
                resource_type="electricity"
            )
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e)
            }), 400

        # Perform analysis
        analysis = electricity_service.analyze_consumption(validated_data)

        return jsonify({
            "success": True,
            "analysis": analysis,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in electricity analysis: {e}")
        return jsonify({
            "success": False,
            "error": "Analysis failed",
            "message": str(e)
        }), 500


@electricity_bp.route("/recommendations", methods=["GET"])
def get_electricity_recommendations():
    """
    Get general electricity saving recommendations.
    
    Returns:
        JSON with recommendations
    """
    recommendations = [
        {
            "id": 1,
            "category": "appliances",
            "title": "Use Energy-Efficient Appliances",
            "description": "Replace old appliances with ENERGY STAR certified models.",
            "potential_savings": "10-50% per appliance",
            "difficulty": "medium",
            "cost": "high",
        },
        {
            "id": 2,
            "category": "lighting",
            "title": "Switch to LED Bulbs",
            "description": "LED bulbs use 75% less energy than incandescent bulbs.",
            "potential_savings": "75% on lighting",
            "difficulty": "easy",
            "cost": "low",
        },
        {
            "id": 3,
            "category": "hvac",
            "title": "Optimize Thermostat Settings",
            "description": "Set thermostat 2-3 degrees lower in winter, higher in summer.",
            "potential_savings": "3% per degree",
            "difficulty": "easy",
            "cost": "free",
        },
        {
            "id": 4,
            "category": "behavior",
            "title": "Unplug Standby Electronics",
            "description": "Unplug devices or use smart power strips to eliminate phantom load.",
            "potential_savings": "5-10% overall",
            "difficulty": "easy",
            "cost": "low",
        },
        {
            "id": 5,
            "category": "timing",
            "title": "Shift Usage to Off-Peak Hours",
            "description": "Run major appliances during off-peak hours (usually nights).",
            "potential_savings": "10-20% on applicable usage",
            "difficulty": "easy",
            "cost": "free",
        },
        {
            "id": 6,
            "category": "maintenance",
            "title": "Regular HVAC Maintenance",
            "description": "Clean/replace filters monthly and schedule annual tune-ups.",
            "potential_savings": "5-15% on heating/cooling",
            "difficulty": "easy",
            "cost": "medium",
        },
    ]

    return jsonify({
        "success": True,
        "recommendations": recommendations,
        "total": len(recommendations),
    }), 200


@electricity_bp.route("/cost-estimate", methods=["POST"])
def estimate_electricity_cost():
    """
    Calculate cost estimates for given consumption.
    
    Request Body:
        {
            "consumption_kwh": 500,  // or
            "data": [{"datetime": "...", "consumption": 1.5}, ...],
            "rate_per_kwh": 0.12  // optional, uses default if not provided
        }
    
    Returns:
        JSON with cost estimates
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Get consumption value
        if "consumption_kwh" in data:
            total_consumption = float(data["consumption_kwh"])
        elif "data" in data:
            total_consumption = sum(
                float(d.get("consumption", 0)) for d in data["data"]
            )
        else:
            return jsonify({
                "success": False,
                "error": "Provide either 'consumption_kwh' or 'data' array"
            }), 400

        # Get rate
        rate = float(data.get("rate_per_kwh", electricity_service.get_cost_per_unit()))

        # Calculate costs
        cost_estimate = electricity_service.calculate_cost(
            consumption=total_consumption,
            rate=rate
        )

        return jsonify({
            "success": True,
            "cost_estimate": cost_estimate,
        }), 200

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": "Invalid value",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in cost estimation: {e}")
        return jsonify({
            "success": False,
            "error": "Cost estimation failed",
            "message": str(e)
        }), 500


@electricity_bp.route("/peak-hours", methods=["GET"])
def get_peak_hours():
    """
    Get peak and off-peak hour information.
    
    Returns:
        JSON with peak hour details
    """
    return jsonify({
        "success": True,
        "peak_hours": {
            "description": "Hours with highest electricity rates",
            "hours": "17:00 - 22:00 (5 PM - 10 PM)",
            "range": list(range(17, 22)),
            "rate_multiplier": 1.5,
            "recommendation": "Avoid running major appliances during these hours"
        },
        "off_peak_hours": {
            "description": "Hours with lowest electricity rates",
            "hours": "00:00 - 07:00 (12 AM - 7 AM)",
            "range": list(range(0, 7)),
            "rate_multiplier": 0.7,
            "recommendation": "Schedule heavy appliances (dishwasher, laundry) during these hours"
        },
        "standard_hours": {
            "description": "Regular rate hours",
            "hours": "07:00 - 17:00 (7 AM - 5 PM)",
            "range": list(range(7, 17)),
            "rate_multiplier": 1.0,
        },
        "timezone": "Local",
        "note": "Peak hours may vary by utility provider and region"
    }), 200


@electricity_bp.route("/appliance-usage", methods=["POST"])
def analyze_appliance_usage():
    """
    Estimate appliance-level electricity usage.
    
    Request Body:
        {
            "total_consumption_kwh": 500,
            "appliances": {  // optional custom profiles
                "refrigerator": 0.15,
                "air_conditioner": 1.5
            }
        }
    
    Returns:
        JSON with estimated appliance breakdown
    """
    try:
        data = request.get_json()
        
        if not data or "total_consumption_kwh" not in data:
            return jsonify({
                "success": False,
                "error": "Please provide 'total_consumption_kwh'"
            }), 400

        total_consumption = float(data["total_consumption_kwh"])
        custom_profiles = data.get("appliances")

        analysis = electricity_service.estimate_appliance_usage(
            total_consumption=total_consumption,
            appliance_profiles=custom_profiles
        )

        return jsonify({
            "success": True,
            "appliance_analysis": analysis,
        }), 200

    except Exception as e:
        logger.error(f"Error in appliance analysis: {e}")
        return jsonify({
            "success": False,
            "error": "Analysis failed",
            "message": str(e)
        }), 500