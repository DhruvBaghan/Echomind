# ============================================
# EchoMind - Dashboard API Routes
# ============================================

"""
API routes for dashboard and overview data.
Provides aggregated statistics and summaries for the frontend dashboard.
"""

from flask import Blueprint, request, jsonify, session
from datetime import datetime, timedelta
from typing import Any, Dict

from backend.services.prediction_service import PredictionService
from backend.services.user_service import UserService
from backend.services.electricity_service import ElectricityService
from backend.services.water_service import WaterService
from backend.utils.logger import logger

# Create blueprint
dashboard_bp = Blueprint("dashboard", __name__)

# Initialize services
prediction_service = PredictionService()
user_service = UserService()
electricity_service = ElectricityService()
water_service = WaterService()


@dashboard_bp.route("/", methods=["GET"])
def dashboard_index():
    """
    Get dashboard module information.
    
    Returns:
        JSON with module information and available endpoints
    """
    return jsonify({
        "success": True,
        "module": "dashboard",
        "description": "Dashboard and overview data",
        "endpoints": {
            "GET /": "Module information",
            "GET /overview": "Get dashboard overview",
            "GET /stats": "Get consumption statistics",
            "GET /recent": "Get recent activity",
            "GET /predictions-summary": "Get predictions summary",
            "GET /cost-summary": "Get cost summary",
            "GET /sustainability": "Get sustainability metrics",
            "GET /alerts": "Get active alerts",
            "GET /tips": "Get daily tips",
        },
    }), 200


@dashboard_bp.route("/overview", methods=["GET"])
def get_overview():
    """
    Get complete dashboard overview.
    
    Query Parameters:
        period (str): Time period (today/week/month, default: today)
    
    Returns:
        JSON with complete dashboard data
    """
    try:
        user_id = session.get("user_id")
        period = request.args.get("period", "today")

        # Validate period
        if period not in ["today", "week", "month"]:
            period = "today"

        # Get current stats
        stats = _get_current_stats(user_id, period)
        
        # Get predictions summary with periods based on selected time frame
        periods_map = {"today": 24, "week": 168, "month": 720}
        predictions = _get_predictions_summary(periods=periods_map.get(period, 24))
        
        # Get sustainability score
        sustainability = _get_sustainability_overview(stats)
        
        # Get alerts
        alerts = _get_active_alerts(stats)
        
        # Get quick tips
        tips = _get_dashboard_tips()

        return jsonify({
            "success": True,
            "overview": {
                "period": period,
                "generated_at": datetime.now().isoformat(),
                "statistics": stats,
                "predictions": predictions,
                "sustainability": sustainability,
                "alerts": alerts,
                "tips": tips[:3],  # Top 3 tips
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get overview",
            "message": str(e)
        }), 500


@dashboard_bp.route("/stats", methods=["GET"])
def get_stats():
    """
    Get consumption statistics.
    
    Query Parameters:
        period (str): Time period (today/week/month/year, default: month)
        resource (str): Resource type (electricity/water/both, default: both)
    
    Returns:
        JSON with consumption statistics
    """
    try:
        user_id = session.get("user_id")
        period = request.args.get("period", "month")
        resource = request.args.get("resource", "both")

        stats = {}

        if resource in ["electricity", "both"]:
            stats["electricity"] = electricity_service.get_statistics(
                user_id=user_id,
                period=period
            )

        if resource in ["water", "both"]:
            stats["water"] = water_service.get_statistics(
                user_id=user_id,
                period=period
            )

        # Add comparison with previous period
        if resource == "both":
            stats["comparison"] = _get_period_comparison(user_id, period)

        return jsonify({
            "success": True,
            "period": period,
            "resource": resource,
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500


@dashboard_bp.route("/recent", methods=["GET"])
def get_recent_activity():
    """
    Get recent consumption activity.
    
    Query Parameters:
        limit (int): Number of entries to return (default: 10)
    
    Returns:
        JSON with recent activity
    """
    try:
        user_id = session.get("user_id")
        limit = request.args.get("limit", 10, type=int)
        limit = min(limit, 50)  # Max 50 entries

        activity = user_service.get_recent_activity(user_id, limit)

        return jsonify({
            "success": True,
            "recent_activity": activity,
            "limit": limit
        }), 200

    except Exception as e:
        logger.error(f"Error getting recent activity: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get recent activity",
            "message": str(e)
        }), 500


@dashboard_bp.route("/predictions-summary", methods=["GET"])
def get_predictions_summary():
    """
    Get summary of predictions.
    
    Query Parameters:
        periods (int): Forecast periods (default: 24)
    
    Returns:
        JSON with predictions summary
    """
    try:
        periods = request.args.get("periods", 24, type=int)
        periods = min(periods, 168)

        summary = prediction_service.get_predictions_summary(periods=periods)

        return jsonify({
            "success": True,
            "predictions_summary": summary,
            "forecast_periods": periods
        }), 200

    except Exception as e:
        logger.error(f"Error getting predictions summary: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get predictions summary",
            "message": str(e)
        }), 500


@dashboard_bp.route("/cost-summary", methods=["GET"])
def get_cost_summary():
    """
    Get cost summary and projections.
    
    Query Parameters:
        period (str): Time period (week/month/year, default: month)
    
    Returns:
        JSON with cost summary
    """
    try:
        user_id = session.get("user_id")
        period = request.args.get("period", "month")

        # Get electricity costs
        electricity_cost = electricity_service.get_cost_summary(
            user_id=user_id,
            period=period
        )

        # Get water costs
        water_cost = water_service.get_cost_summary(
            user_id=user_id,
            period=period
        )

        # Calculate totals
        total_current = (
            electricity_cost.get("current", 0) + 
            water_cost.get("current", 0)
        )
        total_projected = (
            electricity_cost.get("projected", 0) + 
            water_cost.get("projected", 0)
        )

        return jsonify({
            "success": True,
            "period": period,
            "cost_summary": {
                "electricity": electricity_cost,
                "water": water_cost,
                "total": {
                    "current": round(total_current, 2),
                    "projected": round(total_projected, 2),
                    "currency": "USD"
                }
            },
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get cost summary",
            "message": str(e)
        }), 500


@dashboard_bp.route("/sustainability", methods=["GET"])
def get_sustainability():
    """
    Get sustainability metrics and score.
    
    Returns:
        JSON with sustainability information
    """
    try:
        user_id = session.get("user_id")

        # Get consumption data
        electricity_daily = electricity_service.get_daily_average(user_id)
        water_daily = water_service.get_daily_average(user_id)

        # Calculate sustainability score
        sustainability = prediction_service.calculate_sustainability_score(
            electricity_daily=electricity_daily,
            water_daily=water_daily
        )

        # Add improvement tips
        sustainability["improvement_tips"] = _get_improvement_tips(sustainability)

        return jsonify({
            "success": True,
            "sustainability": sustainability,
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting sustainability: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get sustainability metrics",
            "message": str(e)
        }), 500


@dashboard_bp.route("/alerts", methods=["GET"])
def get_alerts():
    """
    Get active alerts and notifications.
    
    Returns:
        JSON with active alerts
    """
    try:
        user_id = session.get("user_id")

        alerts = []

        # Check for high usage alerts
        electricity_alert = electricity_service.check_usage_alert(user_id)
        if electricity_alert:
            alerts.append(electricity_alert)

        water_alert = water_service.check_usage_alert(user_id)
        if water_alert:
            alerts.append(water_alert)

        # Check for leak detection
        leak_alert = water_service.check_leak_alert(user_id)
        if leak_alert:
            alerts.append(leak_alert)

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 4))

        return jsonify({
            "success": True,
            "alerts": alerts,
            "total": len(alerts),
            "generated_at": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get alerts",
            "message": str(e)
        }), 500


@dashboard_bp.route("/tips", methods=["GET"])
def get_tips():
    """
    Get daily saving tips.
    
    Query Parameters:
        category (str): Tip category (electricity/water/general, default: all)
        limit (int): Number of tips (default: 5)
    
    Returns:
        JSON with tips
    """
    try:
        category = request.args.get("category", "all")
        limit = request.args.get("limit", 5, type=int)
        limit = min(limit, 20)

        tips = _get_dashboard_tips(category)[:limit]

        return jsonify({
            "success": True,
            "tips": tips,
            "category": category,
            "total": len(tips)
        }), 200

    except Exception as e:
        logger.error(f"Error getting tips: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get tips",
            "message": str(e)
        }), 500


@dashboard_bp.route("/chart-data", methods=["GET"])
def get_chart_data():
    """
    Get data formatted for dashboard charts.
    
    Query Parameters:
        chart_type (str): Type of chart (consumption/cost/comparison)
        period (str): Time period (week/month)
        resource (str): Resource type (electricity/water/both)
    
    Returns:
        JSON with chart-ready data
    """
    try:
        user_id = session.get("user_id")
        chart_type = request.args.get("chart_type", "consumption")
        period = request.args.get("period", "week")
        resource = request.args.get("resource", "both")

        chart_data = _generate_chart_data(user_id, chart_type, period, resource)

        return jsonify({
            "success": True,
            "chart_data": chart_data,
            "chart_type": chart_type,
            "period": period,
            "resource": resource
        }), 200

    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to get chart data",
            "message": str(e)
        }), 500


# ============================================
# Helper Functions
# ============================================

def _get_current_stats(user_id: int, period: str) -> Dict[str, Any]:
    """Get current consumption statistics."""
    return {
        "electricity": {
            "consumption": electricity_service.get_period_consumption(user_id, period),
            "unit": "kWh",
            "trend": electricity_service.get_trend(user_id, period),
        },
        "water": {
            "consumption": water_service.get_period_consumption(user_id, period),
            "unit": "liters",
            "trend": water_service.get_trend(user_id, period),
        }
    }


def _get_predictions_summary(periods: int = 24) -> Dict[str, Any]:
    """Get predictions summary for dashboard."""
    try:
        return prediction_service.get_predictions_summary(periods=periods)
    except Exception:
        return {"available": False}


def _get_sustainability_overview(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Get sustainability overview."""
    return {
        "score": 75,  # Placeholder - calculate based on actual data
        "grade": "B",
        "trend": "improving",
    }


def _get_active_alerts(stats: Dict[str, Any]) -> list:
    """Get active alerts based on current stats."""
    alerts = []
    # Add logic to generate alerts based on stats
    return alerts


def _get_dashboard_tips(category: str = "all") -> list:
    """Get tips for dashboard."""
    all_tips = [
        {
            "id": 1,
            "category": "electricity",
            "tip": "Turn off lights when leaving a room",
            "impact": "Save up to $50/year",
        },
        {
            "id": 2,
            "category": "water",
            "tip": "Fix dripping faucets promptly",
            "impact": "Save 3,000 liters/year",
        },
        {
            "id": 3,
            "category": "electricity",
            "tip": "Use LED bulbs instead of incandescent",
            "impact": "75% less energy for lighting",
        },
        {
            "id": 4,
            "category": "water",
            "tip": "Take shorter showers",
            "impact": "Save 20 liters per shower",
        },
        {
            "id": 5,
            "category": "general",
            "tip": "Monitor your usage daily",
            "impact": "Awareness leads to 10-15% savings",
        },
    ]
    
    if category == "all":
        return all_tips
    return [t for t in all_tips if t["category"] == category]


def _get_improvement_tips(sustainability: Dict[str, Any]) -> list:
    """Get improvement tips based on sustainability score."""
    tips = []
    score = sustainability.get("overall_score", 100)
    
    if score < 60:
        tips.append("Focus on reducing peak-hour electricity usage")
        tips.append("Check for water leaks in your home")
    elif score < 80:
        tips.append("Consider upgrading to energy-efficient appliances")
        tips.append("Install low-flow showerheads")
    else:
        tips.append("Great job! Maintain your current practices")
        tips.append("Share your tips with neighbors")
    
    return tips


def _get_period_comparison(user_id: int, period: str) -> Dict[str, Any]:
    """Compare current period with previous."""
    return {
        "electricity_change": "+5%",  # Placeholder
        "water_change": "-3%",  # Placeholder
        "overall_trend": "stable",
    }


def _generate_chart_data(
    user_id: int,
    chart_type: str,
    period: str,
    resource: str
) -> Dict[str, Any]:
    """Generate chart-ready data."""
    # Placeholder - implement based on actual data
    return {
        "labels": [],
        "datasets": [],
    }