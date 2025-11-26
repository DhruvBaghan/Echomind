# ============================================
# EchoMind - Prediction API Routes
# ============================================

"""
Unified API routes for predictions.
Handles combined predictions for both electricity and water.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Any, Dict

from backend.models.unified_optimizer import UnifiedOptimizer
from backend.services.prediction_service import PredictionService
from backend.utils.validators import (
    validate_prediction_request,
    validate_combined_request,
    ValidationError,
)
from backend.utils.logger import logger

# Create blueprint
prediction_bp = Blueprint("prediction", __name__)

# Initialize service
prediction_service = PredictionService()


@prediction_bp.route("/", methods=["GET"])
def prediction_index():
    """
    Get prediction module information.
    
    Returns:
        JSON with module information and available endpoints
    """
    return jsonify({
        "success": True,
        "module": "prediction",
        "description": "Unified resource consumption predictions",
        "endpoints": {
            "GET /": "Module information",
            "POST /electricity": "Electricity predictions",
            "POST /water": "Water predictions",
            "POST /both": "Combined predictions for both resources",
            "GET /demo": "Demo predictions",
            "POST /compare": "Compare two time periods",
            "GET /sustainability-score": "Get sustainability score",
            "GET /model-info": "Get model information",
        },
    }), 200


@prediction_bp.route("/electricity", methods=["POST"])
def predict_electricity():
    """
    Generate electricity consumption predictions.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 1.5},
                ...
            ],
            "periods": 24,
            "frequency": "H"
        }
    
    Returns:
        JSON with electricity predictions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Validate request
        try:
            validated = validate_prediction_request(data, resource_type="electricity")
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e)
            }), 400

        result = prediction_service.predict_electricity(
            user_data=validated.get("data"),
            periods=validated.get("periods", 24),
            frequency=validated.get("frequency", "H")
        )

        return jsonify(result), 200 if result.get("success") else 500

    except Exception as e:
        logger.error(f"Error in electricity prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/water", methods=["POST"])
def predict_water():
    """
    Generate water consumption predictions.
    
    Request Body:
        {
            "data": [
                {"datetime": "2024-01-01T00:00:00", "consumption": 50},
                ...
            ],
            "periods": 24,
            "frequency": "H"
        }
    
    Returns:
        JSON with water predictions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Validate request
        try:
            validated = validate_prediction_request(data, resource_type="water")
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e)
            }), 400

        result = prediction_service.predict_water(
            user_data=validated.get("data"),
            periods=validated.get("periods", 24),
            frequency=validated.get("frequency", "H")
        )

        return jsonify(result), 200 if result.get("success") else 500

    except Exception as e:
        logger.error(f"Error in water prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/both", methods=["POST"])
def predict_both():
    """
    Generate predictions for both electricity and water.
    
    Request Body:
        {
            "electricity": {
                "data": [{"datetime": "...", "consumption": 1.5}, ...],
                "periods": 24,
                "frequency": "H"
            },
            "water": {
                "data": [{"datetime": "...", "consumption": 50}, ...],
                "periods": 24,
                "frequency": "H"
            }
        }
    
    Returns:
        JSON with combined predictions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Validate combined request
        try:
            validated = validate_combined_request(data)
        except ValidationError as e:
            return jsonify({
                "success": False,
                "error": "Validation error",
                "message": str(e)
            }), 400

        # Get electricity data
        electricity_data = validated.get("electricity", {})
        water_data = validated.get("water", {})

        result = prediction_service.predict_both(
            electricity_data=electricity_data.get("data"),
            water_data=water_data.get("data"),
            electricity_periods=electricity_data.get("periods", 24),
            water_periods=water_data.get("periods", 24),
            frequency=validated.get("frequency", "H")
        )

        return jsonify(result), 200 if result.get("success") else 500

    except Exception as e:
        logger.error(f"Error in combined prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/demo", methods=["GET"])
def predict_demo():
    """
    Get demo predictions using pre-trained models.
    
    Query Parameters:
        resource (str): Resource type (electricity/water/both, default: both)
        periods (int): Number of periods (default: 24)
        frequency (str): Frequency (default: H)
    
    Returns:
        JSON with demo predictions
    """
    try:
        resource = request.args.get("resource", "both")
        periods = request.args.get("periods", 24, type=int)
        frequency = request.args.get("frequency", "H")

        # Limit periods
        periods = min(periods, 168)

        if resource not in ["electricity", "water", "both"]:
            return jsonify({
                "success": False,
                "error": "Invalid resource",
                "message": "Resource must be 'electricity', 'water', or 'both'"
            }), 400

        result = prediction_service.get_demo_predictions(
            resource=resource,
            periods=periods,
            frequency=frequency
        )

        result["demo"] = True
        result["note"] = "Demo predictions using pre-trained models"
        
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in demo prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Demo prediction failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/compare", methods=["POST"])
def compare_periods():
    """
    Compare consumption between two time periods.
    
    Request Body:
        {
            "period1": {
                "electricity": [{"datetime": "...", "consumption": 1.5}, ...],
                "water": [{"datetime": "...", "consumption": 50}, ...]
            },
            "period2": {
                "electricity": [{"datetime": "...", "consumption": 1.8}, ...],
                "water": [{"datetime": "...", "consumption": 45}, ...]
            }
        }
    
    Returns:
        JSON with comparison analysis
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        if "period1" not in data or "period2" not in data:
            return jsonify({
                "success": False,
                "error": "Both period1 and period2 are required"
            }), 400

        result = prediction_service.compare_periods(
            period1_data=data["period1"],
            period2_data=data["period2"]
        )

        return jsonify({
            "success": True,
            "comparison": result,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in period comparison: {e}")
        return jsonify({
            "success": False,
            "error": "Comparison failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/sustainability-score", methods=["POST"])
def get_sustainability_score():
    """
    Calculate sustainability score based on consumption data.
    
    Request Body:
        {
            "electricity": {
                "daily_consumption_kwh": 25
            },
            "water": {
                "daily_consumption_liters": 300
            },
            "household_size": 4  // optional
        }
    
    Returns:
        JSON with sustainability score
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        electricity_daily = None
        water_daily = None
        household_size = data.get("household_size", 4)

        if "electricity" in data:
            electricity_daily = data["electricity"].get("daily_consumption_kwh")
        
        if "water" in data:
            water_daily = data["water"].get("daily_consumption_liters")

        if electricity_daily is None and water_daily is None:
            return jsonify({
                "success": False,
                "error": "Provide at least electricity or water consumption"
            }), 400

        result = prediction_service.calculate_sustainability_score(
            electricity_daily=electricity_daily,
            water_daily=water_daily,
            household_size=household_size
        )

        return jsonify({
            "success": True,
            "sustainability": result,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error calculating sustainability: {e}")
        return jsonify({
            "success": False,
            "error": "Calculation failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/model-info", methods=["GET"])
def get_model_info():
    """
    Get information about loaded prediction models.
    
    Returns:
        JSON with model information
    """
    try:
        info = prediction_service.get_model_info()
        
        return jsonify({
            "success": True,
            "models": info
        }), 200

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get model info",
            "message": str(e)
        }), 500


@prediction_bp.route("/forecast-accuracy", methods=["POST"])
def check_forecast_accuracy():
    """
    Compare predictions against actual values to assess accuracy.
    
    Request Body:
        {
            "resource_type": "electricity",
            "predictions": [
                {"datetime": "...", "predicted": 1.5},
                ...
            ],
            "actuals": [
                {"datetime": "...", "actual": 1.6},
                ...
            ]
        }
    
    Returns:
        JSON with accuracy metrics
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        required_fields = ["resource_type", "predictions", "actuals"]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        result = prediction_service.calculate_accuracy(
            resource_type=data["resource_type"],
            predictions=data["predictions"],
            actuals=data["actuals"]
        )

        return jsonify({
            "success": True,
            "accuracy": result,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        return jsonify({
            "success": False,
            "error": "Accuracy calculation failed",
            "message": str(e)
        }), 500


@prediction_bp.route("/quick-predict", methods=["POST"])
def quick_predict():
    """
    Quick prediction with minimal data input.
    
    Request Body:
        {
            "resource_type": "electricity",
            "recent_average": 1.5,  // average consumption per hour
            "periods": 24
        }
    
    Returns:
        JSON with quick predictions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        resource_type = data.get("resource_type")
        recent_average = data.get("recent_average")
        periods = data.get("periods", 24)

        if not resource_type or recent_average is None:
            return jsonify({
                "success": False,
                "error": "resource_type and recent_average are required"
            }), 400

        result = prediction_service.quick_predict(
            resource_type=resource_type,
            recent_average=float(recent_average),
            periods=int(periods)
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in quick predict: {e}")
        return jsonify({
            "success": False,
            "error": "Quick prediction failed",
            "message": str(e)
        }), 500