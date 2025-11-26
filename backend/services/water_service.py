# ============================================
# EchoMind - Water Service
# ============================================

"""
Service class for water-related operations.
Handles business logic for water consumption predictions,
leak detection, and recommendations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from backend.models.water_predictor import WaterPredictor
from backend.config import Config
from backend.utils.logger import logger


class WaterService:
    """
    Service for water consumption operations.
    
    Provides methods for:
    - Generating predictions
    - Analyzing consumption patterns
    - Detecting leaks
    - Calculating costs
    - Generating recommendations
    """

    def __init__(self):
        """Initialize water service."""
        self.predictor = WaterPredictor()
        self.cost_per_liter = Config.WATER_COST_PER_LITER
        
        # Usage period settings
        self.morning_peak_hours = list(range(6, 10))
        self.evening_peak_hours = list(range(18, 22))
        
        # Usage thresholds (liters per hour)
        self.high_usage_threshold = 100
        self.normal_usage_threshold = 30
        self.leak_threshold = 10  # Minimum constant flow suggesting leak

    def get_cost_per_unit(self) -> float:
        """Get cost per liter."""
        return self.cost_per_liter

    def predict(
        self,
        periods: int = 24,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate predictions using pre-trained model.

        Args:
            periods: Number of periods to predict
            frequency: Prediction frequency

        Returns:
            Prediction results
        """
        try:
            if not self.predictor.is_loaded:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "message": "Pre-trained water model is not available"
                }

            result = self.predictor.predict(
                periods=periods,
                frequency=frequency
            )
            return result

        except Exception as e:
            logger.error(f"Water prediction error: {e}")
            return {
                "success": False,
                "error": "Prediction failed",
                "message": str(e)
            }

    def predict_from_user_data(
        self,
        user_data: List[Dict[str, Any]],
        periods: int = 24,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate predictions based on user-provided data.

        Args:
            user_data: List of consumption entries
            periods: Number of periods to predict
            frequency: Prediction frequency

        Returns:
            Prediction results
        """
        try:
            if not user_data or len(user_data) < 2:
                return {
                    "success": False,
                    "error": "Insufficient data",
                    "message": "At least 2 data points are required for prediction"
                }

            result = self.predictor.predict_from_user_data(
                user_data=user_data,
                periods=periods,
                frequency=frequency
            )
            return result

        except Exception as e:
            logger.error(f"Water user prediction error: {e}")
            return {
                "success": False,
                "error": "Prediction failed",
                "message": str(e)
            }

    def analyze_consumption(
        self,
        consumption_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze water consumption patterns.

        Args:
            consumption_data: List of consumption entries

        Returns:
            Analysis results
        """
        try:
            if not consumption_data:
                return {"error": "No data provided"}

            # Convert to DataFrame
            df = pd.DataFrame(consumption_data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
            df = df.dropna()

            if len(df) == 0:
                return {"error": "No valid data after processing"}

            # Extract time features
            df["hour"] = df["datetime"].dt.hour
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6])

            # Basic statistics
            stats = {
                "total_consumption": round(float(df["consumption"].sum()), 2),
                "average_consumption": round(float(df["consumption"].mean()), 2),
                "max_consumption": round(float(df["consumption"].max()), 2),
                "min_consumption": round(float(df["consumption"].min()), 2),
                "std_deviation": round(float(df["consumption"].std()), 2),
                "data_points": len(df),
            }

            # Usage period analysis
            morning_df = df[df["hour"].isin(self.morning_peak_hours)]
            evening_df = df[df["hour"].isin(self.evening_peak_hours)]
            night_df = df[df["hour"].isin(range(0, 6))]

            period_analysis = {
                "morning_peak": {
                    "total": round(float(morning_df["consumption"].sum()), 2) if len(morning_df) > 0 else 0,
                    "average": round(float(morning_df["consumption"].mean()), 2) if len(morning_df) > 0 else 0,
                },
                "evening_peak": {
                    "total": round(float(evening_df["consumption"].sum()), 2) if len(evening_df) > 0 else 0,
                    "average": round(float(evening_df["consumption"].mean()), 2) if len(evening_df) > 0 else 0,
                },
                "night": {
                    "total": round(float(night_df["consumption"].sum()), 2) if len(night_df) > 0 else 0,
                    "average": round(float(night_df["consumption"].mean()), 2) if len(night_df) > 0 else 0,
                },
            }

            # Hourly pattern
            hourly_pattern = df.groupby("hour")["consumption"].mean().round(2).to_dict()

            # Day of week pattern
            daily_pattern = df.groupby("day_of_week")["consumption"].mean().round(2).to_dict()
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_pattern = {day_names[k]: v for k, v in daily_pattern.items()}

            # Weekend vs weekday
            weekend_analysis = {
                "weekday_average": round(float(df[~df["is_weekend"]]["consumption"].mean()), 2) if len(df[~df["is_weekend"]]) > 0 else 0,
                "weekend_average": round(float(df[df["is_weekend"]]["consumption"].mean()), 2) if len(df[df["is_weekend"]]) > 0 else 0,
            }

            # Leak detection
            leak_analysis = self._analyze_for_leaks(df)

            # Generate insights
            insights = self._generate_insights(stats, period_analysis, leak_analysis)

            return {
                "statistics": stats,
                "period_analysis": period_analysis,
                "hourly_pattern": hourly_pattern,
                "daily_pattern": daily_pattern,
                "weekend_analysis": weekend_analysis,
                "leak_analysis": leak_analysis,
                "insights": insights,
            }

        except Exception as e:
            logger.error(f"Water analysis error: {e}")
            return {"error": str(e)}

    def _analyze_for_leaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data for potential leaks."""
        # Check night usage (1 AM - 5 AM)
        night_df = df[df["hour"].isin(range(1, 6))]
        
        if len(night_df) == 0:
            return {"analyzed": False, "reason": "No night data available"}

        night_avg = float(night_df["consumption"].mean())
        night_min = float(night_df["consumption"].min())
        
        leak_detected = night_min > self.leak_threshold
        
        result = {
            "analyzed": True,
            "leak_detected": leak_detected,
            "night_average_lph": round(night_avg, 2),
            "night_minimum_lph": round(night_min, 2),
            "threshold_lph": self.leak_threshold,
        }
        
        if leak_detected:
            severity = "low"
            if night_min > 30:
                severity = "high"
            elif night_min > 20:
                severity = "medium"
                
            result["severity"] = severity
            result["estimated_daily_loss"] = round(night_min * 24, 2)
            result["estimated_monthly_cost"] = round(night_min * 24 * 30 * self.cost_per_liter, 2)
            result["recommendation"] = self._get_leak_recommendation(severity)

        return result

    def _get_leak_recommendation(self, severity: str) -> str:
        """Get recommendation based on leak severity."""
        recommendations = {
            "high": "URGENT: Significant water leak detected! Check all faucets, toilets, and pipes immediately. Consider calling a plumber.",
            "medium": "WARNING: Moderate leak detected. Inspect toilets (use dye test), check under sinks, and examine outdoor faucets.",
            "low": "NOTICE: Minor leak possible. Check toilet flappers, faucet drips, and showerheads. Monitor usage closely.",
        }
        return recommendations.get(severity, "")

    def _generate_insights(
        self,
        stats: Dict[str, Any],
        period_analysis: Dict[str, Any],
        leak_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from analysis."""
        insights = []

        # Average consumption insight
        avg = stats.get("average_consumption", 0)
        if avg > self.high_usage_threshold:
            insights.append(f"Your average consumption ({avg} L/h) is above typical household levels.")
        elif avg < self.normal_usage_threshold:
            insights.append(f"Your average consumption ({avg} L/h) is efficient - great job!")

        # Morning vs evening insight
        morning_avg = period_analysis.get("morning_peak", {}).get("average", 0)
        evening_avg = period_analysis.get("evening_peak", {}).get("average", 0)
        if morning_avg > evening_avg * 1.5:
            insights.append("Morning water usage is significantly higher than evening.")
        elif evening_avg > morning_avg * 1.5:
            insights.append("Evening water usage is significantly higher than morning.")

        # Leak insight
        if leak_analysis.get("leak_detected"):
            severity = leak_analysis.get("severity", "unknown")
            insights.append(f"Potential water leak detected (severity: {severity}). Check plumbing.")

        return insights

    def detect_leaks(
        self,
        consumption_data: List[Dict[str, Any]],
        sensitivity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Detect potential water leaks.

        Args:
            consumption_data: List of consumption entries
            sensitivity: Detection sensitivity (low/medium/high)

        Returns:
            Leak detection results
        """
        try:
            # Adjust threshold based on sensitivity
            thresholds = {
                "low": 15,
                "medium": 10,
                "high": 5,
            }
            threshold = thresholds.get(sensitivity, 10)

            # Convert to DataFrame
            df = pd.DataFrame(consumption_data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
            df["hour"] = df["datetime"].dt.hour
            df = df.dropna()

            # Analyze night hours (1 AM - 5 AM)
            night_df = df[df["hour"].isin(range(1, 6))]

            if len(night_df) < 3:
                return {
                    "success": True,
                    "analyzed": False,
                    "reason": "Insufficient night-time data (need at least 3 hours)",
                }

            night_values = night_df["consumption"].values
            night_avg = float(np.mean(night_values))
            night_min = float(np.min(night_values))
            night_max = float(np.max(night_values))
            night_std = float(np.std(night_values))

            # Detect leak
            leak_detected = night_min > threshold
            
            # Determine severity
            severity = "none"
            if leak_detected:
                if night_min > 30:
                    severity = "high"
                elif night_min > 20:
                    severity = "medium"
                else:
                    severity = "low"

            # Calculate potential loss
            daily_loss = night_min * 24 if leak_detected else 0
            monthly_loss = daily_loss * 30
            monthly_cost = monthly_loss * self.cost_per_liter

            return {
                "success": True,
                "analyzed": True,
                "sensitivity": sensitivity,
                "threshold_lph": threshold,
                "results": {
                    "leak_detected": leak_detected,
                    "severity": severity,
                    "night_average_lph": round(night_avg, 2),
                    "night_minimum_lph": round(night_min, 2),
                    "night_maximum_lph": round(night_max, 2),
                    "night_std_deviation": round(night_std, 2),
                    "consistency": "constant" if night_std < 5 else "variable",
                },
                "potential_loss": {
                    "daily_liters": round(daily_loss, 2),
                    "monthly_liters": round(monthly_loss, 2),
                    "yearly_liters": round(monthly_loss * 12, 2),
                    "monthly_cost_usd": round(monthly_cost, 2),
                    "yearly_cost_usd": round(monthly_cost * 12, 2),
                },
                "recommendation": self._get_leak_recommendation(severity) if leak_detected else "No action needed - no leak detected.",
                "next_steps": self._get_leak_next_steps(severity) if leak_detected else [],
            }

        except Exception as e:
            logger.error(f"Leak detection error: {e}")
            return {
                "success": False,
                "error": "Leak detection failed",
                "message": str(e)
            }

    def _get_leak_next_steps(self, severity: str) -> List[str]:
        """Get next steps based on leak severity."""
        steps = {
            "high": [
                "Turn off main water supply if possible",
                "Check for visible leaks under sinks and around toilets",
                "Inspect water heater for leaks",
                "Check outdoor spigots and sprinkler systems",
                "Call a licensed plumber immediately",
            ],
            "medium": [
                "Perform toilet dye test (add food coloring to tank)",
                "Check all faucets for drips",
                "Inspect washing machine and dishwasher connections",
                "Check water meter - if moving with all water off, there's a leak",
                "Schedule plumber visit within a week",
            ],
            "low": [
                "Check toilet flappers for wear",
                "Inspect showerheads and faucet aerators",
                "Look for condensation issues",
                "Monitor water bills for unusual increases",
                "Recheck in one week",
            ],
        }
        return steps.get(severity, [])

    def calculate_cost(
        self,
        consumption: float,
        rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate cost for given water consumption.

        Args:
            consumption: Total consumption in liters
            rate: Optional custom rate per liter

        Returns:
            Cost calculation results
        """
        rate = rate or self.cost_per_liter

        total_cost = consumption * rate

        return {
            "consumption_liters": round(consumption, 2),
            "rate_per_liter": rate,
            "total_cost": round(total_cost, 2),
            "currency": "USD",
            "daily_cost": round(total_cost / 30, 2),
            "weekly_cost": round(total_cost / 4, 2),
            "yearly_projection": round(total_cost * 12, 2),
            "gallons_equivalent": round(consumption * 0.264172, 2),
        }

    def estimate_activity_usage(
        self,
        total_consumption: float,
        household_size: int = 4,
        activity_profiles: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Estimate water usage by household activity.

        Args:
            total_consumption: Total consumption in liters
            household_size: Number of people in household
            activity_profiles: Optional custom activity profiles

        Returns:
            Activity usage breakdown
        """
        # Default activity breakdown percentages
        default_breakdown = {
            "toilet": 0.30,
            "shower_bath": 0.25,
            "faucets": 0.15,
            "clothes_washer": 0.15,
            "dishwasher": 0.05,
            "leaks": 0.05,
            "other": 0.05,
        }

        # Average liters per use
        activity_rates = {
            "toilet": 9,
            "shower_bath": 65,
            "faucets": 8,
            "clothes_washer": 50,
            "dishwasher": 15,
        }

        breakdown = {}
        for activity, percentage in default_breakdown.items():
            usage = total_consumption * percentage
            cost = usage * self.cost_per_liter
            
            activity_data = {
                "consumption_liters": round(usage, 2),
                "percentage": round(percentage * 100, 1),
                "estimated_cost": round(cost, 2),
            }
            
            # Add estimated uses for activities with rates
            if activity in activity_rates:
                estimated_uses = usage / activity_rates[activity]
                activity_data["estimated_uses"] = round(estimated_uses, 1)
                activity_data["liters_per_use"] = activity_rates[activity]
            
            breakdown[activity] = activity_data

        # Per person calculation
        per_person_daily = total_consumption / (household_size * 30)  # Assuming monthly data

        return {
            "total_consumption_liters": round(total_consumption, 2),
            "household_size": household_size,
            "per_person_daily": round(per_person_daily, 2),
            "breakdown": breakdown,
            "efficiency_rating": self._get_efficiency_rating(per_person_daily),
            "note": "Estimates based on typical household usage patterns",
        }

    def _get_efficiency_rating(self, per_person_daily: float) -> str:
        """Get efficiency rating based on per-person daily usage."""
        if per_person_daily < 80:
            return "excellent"
        elif per_person_daily < 120:
            return "good"
        elif per_person_daily < 150:
            return "average"
        else:
            return "needs_improvement"

    def get_statistics(
        self,
        user_id: Optional[int] = None,
        period: str = "month"
    ) -> Dict[str, Any]:
        """Get consumption statistics for dashboard."""
        demo_stats = {
            "week": {
                "total_consumption": 1400,
                "average_daily": 200,
                "peak_day": "Saturday",
                "peak_consumption": 280,
            },
            "month": {
                "total_consumption": 6000,
                "average_daily": 200,
                "peak_day": "2024-01-15",
                "peak_consumption": 310,
            },
            "year": {
                "total_consumption": 72000,
                "average_daily": 197,
                "peak_month": "August",
                "peak_consumption": 7500,
            },
        }
        return demo_stats.get(period, demo_stats["month"])

    def get_period_consumption(
        self,
        user_id: Optional[int] = None,
        period: str = "today"
    ) -> float:
        """Get consumption for a specific period."""
        demo_values = {
            "today": 180,
            "week": 1400,
            "month": 6000,
        }
        return demo_values.get(period, 0.0)

    def get_trend(
        self,
        user_id: Optional[int] = None,
        period: str = "month"
    ) -> str:
        """Get consumption trend."""
        return "decreasing"

    def get_cost_summary(
        self,
        user_id: Optional[int] = None,
        period: str = "month"
    ) -> Dict[str, Any]:
        """Get cost summary for a period."""
        consumption = self.get_period_consumption(user_id, period)
        
        return {
            "current": round(consumption * self.cost_per_liter, 2),
            "projected": round(consumption * self.cost_per_liter * 1.05, 2),
            "previous": round(consumption * self.cost_per_liter * 1.1, 2),
            "unit": "USD",
        }

    def get_daily_average(self, user_id: Optional[int] = None) -> float:
        """Get daily average consumption."""
        return 200.0  # Liters per day

    def check_usage_alert(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Check for high usage alerts."""
        daily_avg = self.get_daily_average(user_id)
        
        if daily_avg > 300:
            return {
                "type": "high_water_usage",
                "priority": "medium",
                "title": "High Water Usage Detected",
                "message": f"Your daily average ({daily_avg}L) is above recommended levels.",
                "timestamp": datetime.now().isoformat(),
            }
        return None

    def check_leak_alert(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Check for leak alerts."""
        # Demo check - would analyze real data in production
        return None