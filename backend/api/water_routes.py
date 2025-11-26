# ============================================
# EchoMind - Water API Routes
# ============================================

"""
API routes for water-specific operations.
Handles water consumption predictions, leak detection, and recommendations.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Any, Dict

from backend.models.water_predictor import WaterPredictor
from backend.services.water_service import WaterService
from backend.utils.validators import (
    validate_prediction_request,
    validate_consumption_data,
    ValidationError,
)
from backend.utils.logger import logger

# Create blueprint
water_bp = Blueprint("water", __name__)

# Initialize service
water_service = WaterService()


@water_bp.route("/", methods=["GET"])
def water_index():
    """
    Get water module information.
    
    Returns:
        JSON with module information and available endpoints
    """
    return jsonify({
        "success": True,
        "module": "water",
        "description": "Water consumption prediction and analysis",
        "unit": "liters",
        "endpoints": {
            "GET /": "Module information",
            "POST /predict": "Generate predictions from user data",
            "GET /predict/demo": "Get demo predictions",
            "POST /analyze": "Analyze consumption patterns",
            "GET /recommendations": "Get general recommendations",
            "POST /cost-estimate": "Calculate cost estimates",
            "POST /leak-detection": "Analyze for potential leaks",
            "GET /usage-patterns": "Get typical usage patterns",
        },
    }), 200


@water_bp.route("/predict", methods=["POST"])
def predict_water():
    """
    Generate water consumption predictions based on user data.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 50},
                {"datetime": "2024-01-01T01:00:00", "consumption": 20},
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
            validated = validate_prediction_request(data, resource_type="water")
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
        result = water_service.predict_from_user_data(
            user_data=consumption_data,
            periods=periods,
            frequency=frequency
        )

        if result.get("success"):
            logger.info(f"Water prediction generated: {periods} periods")
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error in water prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@water_bp.route("/predict/demo", methods=["GET"])
def predict_water_demo():
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

        result = water_service.predict(
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


@water_bp.route("/analyze", methods=["POST"])
def analyze_water():
    """
    Analyze water consumption patterns.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 50},
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
                resource_type="water"
            )
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e)
            }), 400

        # Perform analysis
        analysis = water_service.analyze_consumption(validated_data)

        return jsonify({
            "success": True,
            "analysis": analysis,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in water analysis: {e}")
        return jsonify({
            "success": False,
            "error": "Analysis failed",
            "message": str(e)
        }), 500


@water_bp.route("/leak-detection", methods=["POST"])
def detect_leaks():
    """
    Analyze consumption data for potential water leaks.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 50},
                ...
            ],
            "sensitivity": "medium"  // low, medium, high
        }
    
    Returns:
        JSON with leak detection results
    """
    try:
        data = request.get_json()
        
        if not data or "data" not in data:
            return jsonify({
                "success": False,
                "error": "No data provided",
                "message": "Please provide consumption data for leak detection"
            }), 400

        sensitivity = data.get("sensitivity", "medium")
        
        # Validate sensitivity
        if sensitivity not in ["low", "medium", "high"]:
            return jsonify({
                "success": False,
                "error": "Invalid sensitivity",
                "message": "Sensitivity must be 'low', 'medium', or 'high'"
            }), 400

        # Perform leak detection
        leak_result = water_service.detect_leaks(
            consumption_data=data["data"],
            sensitivity=sensitivity
        )

        return jsonify({
            "success": True,
            "leak_detection": leak_result,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in leak detection: {e}")
        return jsonify({
            "success": False,
            "error": "Leak detection failed",
            "message": str(e)
        }), 500


@water_bp.route("/recommendations", methods=["GET"])
def get_water_recommendations():
    """
    Get general water saving recommendations.
    
    Returns:
        JSON with recommendations
    """
    recommendations = [
        {
            "id": 1,
            "category": "fixtures",
            "title": "Install Low-Flow Showerheads",
            "description": "Low-flow showerheads can reduce water usage by up to 50%.",
            "potential_savings": "10-15 liters per shower",
            "difficulty": "easy",
            "cost": "low",
        },
        {
            "id": 2,
            "category": "fixtures",
            "title": "Fix Leaky Faucets and Toilets",
            "description": "A dripping faucet can waste over 3,000 liters per year.",
            "potential_savings": "Up to 3,000 liters/year per leak",
            "difficulty": "easy",
            "cost": "low",
        },
        {
            "id": 3,
            "category": "appliances",
            "title": "Use Efficient Washing Machines",
            "description": "Front-loading machines use 40% less water than top-loaders.",
            "potential_savings": "40% per load",
            "difficulty": "medium",
            "cost": "high",
        },
        {
            "id": 4,
            "category": "behavior",
            "title": "Shorter Showers",
            "description": "Reducing shower time by 2 minutes saves up to 20 liters.",
            "potential_savings": "20 liters per shower",
            "difficulty": "easy",
            "cost": "free",
        },
        {
            "id": 5,
            "category": "outdoor",
            "title": "Water Garden in Morning/Evening",
            "description": "Watering during cooler hours reduces evaporation by 25%.",
            "potential_savings": "25% on garden water",
            "difficulty": "easy",
            "cost": "free",
        },
        {
            "id": 6,
            "category": "fixtures",
            "title": "Install Dual-Flush Toilets",
            "description": "Dual-flush toilets use 3-4 liters for liquid waste vs 9 liters standard.",
            "potential_savings": "Up to 60% on toilet water",
            "difficulty": "medium",
            "cost": "medium",
        },
        {
            "id": 7,
            "category": "collection",
            "title": "Collect Rainwater",
            "description": "Use rain barrels for garden watering and outdoor cleaning.",
            "potential_savings": "Up to 5,000 liters/year",
            "difficulty": "medium",
            "cost": "medium",
        },
        {
            "id": 8,
            "category": "behavior",
            "title": "Run Full Loads Only",
            "description": "Wait until dishwasher/washing machine is full before running.",
            "potential_savings": "15-45 liters per avoided load",
            "difficulty": "easy",
            "cost": "free",
        },
    ]

    return jsonify({
        "success": True,
        "recommendations": recommendations,
        "total": len(recommendations),
    }), 200


@water_bp.route("/cost-estimate", methods=["POST"])
def estimate_water_cost():
    """
    Calculate cost estimates for given water consumption.
    
    Request Body:
        {
            "consumption_liters": 5000,  // or
            "data": [{"datetime": "...", "consumption": 50}, ...],
            "rate_per_liter": 0.002  // optional, uses default if not provided
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
        if "consumption_liters" in data:
            total_consumption = float(data["consumption_liters"])
        elif "data" in data:
            total_consumption = sum(
                float(d.get("consumption", 0)) for d in data["data"]
            )
        else:
            return jsonify({
                "success": False,
                "error": "Provide either 'consumption_liters' or 'data' array"
            }), 400

        # Get rate
        rate = float(data.get("rate_per_liter", water_service.get_cost_per_unit()))

        # Calculate costs
        cost_estimate = water_service.calculate_cost(
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


@water_bp.route("/usage-patterns", methods=["GET"])
def get_usage_patterns():
    """
    Get typical water usage patterns and benchmarks.
    
    Returns:
        JSON with usage pattern information
    """
    return jsonify({
        "success": True,
        "usage_patterns": {
            "morning_peak": {
                "hours": "06:00 - 10:00",
                "description": "Showers, toilet use, breakfast preparation",
                "typical_percentage": "30-35%",
            },
            "midday": {
                "hours": "10:00 - 18:00",
                "description": "Cooking, cleaning, laundry",
                "typical_percentage": "20-25%",
            },
            "evening_peak": {
                "hours": "18:00 - 22:00",
                "description": "Showers, dinner preparation, dishwashing",
                "typical_percentage": "35-40%",
            },
            "night": {
                "hours": "22:00 - 06:00",
                "description": "Minimal usage (potential leak indicator)",
                "typical_percentage": "5-10%",
            },
        },
        "benchmarks": {
            "per_person_daily": {
                "excellent": "< 80 liters",
                "good": "80-120 liters",
                "average": "120-150 liters",
                "high": "> 150 liters",
            },
            "household_daily": {
                "description": "Average household of 4 people",
                "excellent": "< 350 liters",
                "average": "400-500 liters",
                "high": "> 600 liters",
            },
        },
        "activity_usage": {
            "shower_5min": "40-50 liters",
            "bath": "150 liters",
            "toilet_flush": "6-9 liters",
            "dishwasher": "12-15 liters",
            "washing_machine": "40-60 liters",
            "hand_washing_dishes": "20-30 liters",
            "brushing_teeth_tap_running": "6 liters",
            "brushing_teeth_tap_off": "0.5 liters",
        },
    }), 200


@water_bp.route("/activity-breakdown", methods=["POST"])
def analyze_activity_breakdown():
    """
    Estimate water usage by household activity.
    
    Request Body:
        {
            "total_consumption_liters": 5000,
            "household_size": 4,  // optional
            "activities": {  // optional custom profiles
                "shower": 65,
                "bath": 150
            }
        }
    
    Returns:
        JSON with estimated activity breakdown
    """
    try:
        data = request.get_json()
        
        if not data or "total_consumption_liters" not in data:
            return jsonify({
                "success": False,
                "error": "Please provide 'total_consumption_liters'"
            }), 400

        total_consumption = float(data["total_consumption_liters"])
        household_size = int(data.get("household_size", 4))
        custom_profiles = data.get("activities")

        analysis = water_service.estimate_activity_usage(
            total_consumption=total_consumption,
            household_size=household_size,
            activity_profiles=custom_profiles
        )

        return jsonify({
            "success": True,
            "activity_analysis": analysis,
        }), 200

    except Exception as e:
        logger.error(f"Error in activity analysis: {e}")
        return jsonify({
            "success": False,
            "error": "Analysis failed",
            "message": str(e)
        }), 500