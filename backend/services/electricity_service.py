# ============================================
# EchoMind - Electricity Service
# ============================================

"""
Service class for electricity-related operations.
Handles business logic for electricity consumption predictions,
analysis, and recommendations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from backend.models.electricity_predictor import ElectricityPredictor
from backend.config import Config
from backend.utils.logger import logger


class ElectricityService:
    """
    Service for electricity consumption operations.
    
    Provides methods for:
    - Generating predictions
    - Analyzing consumption patterns
    - Calculating costs
    - Generating recommendations
    """

    def __init__(self):
        """Initialize electricity service."""
        self.predictor = ElectricityPredictor()
        self.cost_per_kwh = Config.ELECTRICITY_COST_PER_KWH
        
        # Peak hour settings
        self.peak_hours = list(range(17, 22))  # 5 PM - 10 PM
        self.off_peak_hours = list(range(0, 7))  # 12 AM - 7 AM
        
        # Usage thresholds (kWh per hour)
        self.high_usage_threshold = 3.0
        self.normal_usage_threshold = 1.5

    def get_cost_per_unit(self) -> float:
        """Get cost per kWh."""
        return self.cost_per_kwh

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
                    "message": "Pre-trained electricity model is not available"
                }

            result = self.predictor.predict(
                periods=periods,
                frequency=frequency
            )
            return result

        except Exception as e:
            logger.error(f"Electricity prediction error: {e}")
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
            logger.error(f"Electricity user prediction error: {e}")
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
        Analyze electricity consumption patterns.

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

            # Peak hour analysis
            peak_df = df[df["hour"].isin(self.peak_hours)]
            off_peak_df = df[df["hour"].isin(self.off_peak_hours)]

            peak_analysis = {
                "peak_consumption": round(float(peak_df["consumption"].sum()), 2) if len(peak_df) > 0 else 0,
                "peak_average": round(float(peak_df["consumption"].mean()), 2) if len(peak_df) > 0 else 0,
                "off_peak_consumption": round(float(off_peak_df["consumption"].sum()), 2) if len(off_peak_df) > 0 else 0,
                "off_peak_average": round(float(off_peak_df["consumption"].mean()), 2) if len(off_peak_df) > 0 else 0,
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

            # Identify high usage periods
            high_usage = df[df["consumption"] > self.high_usage_threshold]
            high_usage_periods = []
            for _, row in high_usage.iterrows():
                high_usage_periods.append({
                    "datetime": row["datetime"].isoformat(),
                    "consumption": round(float(row["consumption"]), 2),
                })

            # Generate insights
            insights = self._generate_insights(stats, peak_analysis, weekend_analysis)

            return {
                "statistics": stats,
                "peak_analysis": peak_analysis,
                "hourly_pattern": hourly_pattern,
                "daily_pattern": daily_pattern,
                "weekend_analysis": weekend_analysis,
                "high_usage_periods": high_usage_periods[:10],  # Limit to 10
                "insights": insights,
            }

        except Exception as e:
            logger.error(f"Electricity analysis error: {e}")
            return {"error": str(e)}

    def _generate_insights(
        self,
        stats: Dict[str, Any],
        peak_analysis: Dict[str, Any],
        weekend_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from analysis."""
        insights = []

        # Average consumption insight
        avg = stats.get("average_consumption", 0)
        if avg > self.high_usage_threshold:
            insights.append(f"Your average consumption ({avg} kWh/h) is above typical household levels.")
        elif avg < self.normal_usage_threshold:
            insights.append(f"Your average consumption ({avg} kWh/h) is below average - great efficiency!")

        # Peak vs off-peak insight
        peak_avg = peak_analysis.get("peak_average", 0)
        off_peak_avg = peak_analysis.get("off_peak_average", 0)
        if peak_avg > 0 and off_peak_avg > 0:
            ratio = peak_avg / off_peak_avg if off_peak_avg > 0 else 0
            if ratio > 1.5:
                insights.append(f"Peak hour usage is {ratio:.1f}x higher than off-peak. Consider shifting activities.")

        # Weekend insight
        weekday_avg = weekend_analysis.get("weekday_average", 0)
        weekend_avg = weekend_analysis.get("weekend_average", 0)
        if weekend_avg > weekday_avg * 1.3:
            insights.append("Weekend consumption is significantly higher than weekdays.")
        elif weekday_avg > weekend_avg * 1.3:
            insights.append("Weekday consumption is significantly higher than weekends.")

        return insights

    def calculate_cost(
        self,
        consumption: float,
        rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate cost for given consumption.

        Args:
            consumption: Total consumption in kWh
            rate: Optional custom rate per kWh

        Returns:
            Cost calculation results
        """
        rate = rate or self.cost_per_kwh

        total_cost = consumption * rate

        return {
            "consumption_kwh": round(consumption, 2),
            "rate_per_kwh": rate,
            "total_cost": round(total_cost, 2),
            "currency": "USD",
            "daily_cost": round(total_cost / 30, 2),  # Assuming monthly
            "weekly_cost": round(total_cost / 4, 2),
            "yearly_projection": round(total_cost * 12, 2),
        }

    def estimate_appliance_usage(
        self,
        total_consumption: float,
        appliance_profiles: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Estimate electricity usage by appliance.

        Args:
            total_consumption: Total consumption in kWh
            appliance_profiles: Optional custom appliance profiles

        Returns:
            Appliance usage breakdown
        """
        # Default appliance consumption percentages
        default_breakdown = {
            "heating_cooling": 0.40,
            "water_heater": 0.15,
            "appliances": 0.13,
            "lighting": 0.10,
            "electronics": 0.08,
            "refrigerator": 0.06,
            "washer_dryer": 0.05,
            "other": 0.03,
        }

        breakdown = {}
        for appliance, percentage in default_breakdown.items():
            usage = total_consumption * percentage
            cost = usage * self.cost_per_kwh
            breakdown[appliance] = {
                "consumption_kwh": round(usage, 2),
                "percentage": round(percentage * 100, 1),
                "estimated_cost": round(cost, 2),
            }

        return {
            "total_consumption_kwh": round(total_consumption, 2),
            "breakdown": breakdown,
            "note": "Estimates based on typical household usage patterns",
        }

    def get_statistics(
        self,
        user_id: Optional[int] = None,
        period: str = "month"
    ) -> Dict[str, Any]:
        """
        Get consumption statistics for dashboard.

        Args:
            user_id: Optional user ID for personalized stats
            period: Time period (week/month/year)

        Returns:
            Statistics dictionary
        """
        # This would typically query the database
        # For now, return demo statistics
        demo_stats = {
            "week": {
                "total_consumption": 45.5,
                "average_daily": 6.5,
                "peak_day": "Saturday",
                "peak_consumption": 8.2,
            },
            "month": {
                "total_consumption": 195.0,
                "average_daily": 6.5,
                "peak_day": "2024-01-15",
                "peak_consumption": 9.1,
            },
            "year": {
                "total_consumption": 2340.0,
                "average_daily": 6.4,
                "peak_month": "July",
                "peak_consumption": 280.0,
            },
        }

        return demo_stats.get(period, demo_stats["month"])

    def get_period_consumption(
        self,
        user_id: Optional[int] = None,
        period: str = "today"
    ) -> float:
        """Get consumption for a specific period."""
        # Demo values - would query database in production
        demo_values = {
            "today": 5.2,
            "week": 45.5,
            "month": 195.0,
        }
        return demo_values.get(period, 0.0)

    def get_trend(
        self,
        user_id: Optional[int] = None,
        period: str = "month"
    ) -> str:
        """Get consumption trend compared to previous period."""
        # Demo trend - would calculate from database in production
        return "stable"  # Options: increasing, decreasing, stable

    def get_cost_summary(
        self,
        user_id: Optional[int] = None,
        period: str = "month"
    ) -> Dict[str, Any]:
        """Get cost summary for a period."""
        consumption = self.get_period_consumption(user_id, period)
        
        return {
            "current": round(consumption * self.cost_per_kwh, 2),
            "projected": round(consumption * self.cost_per_kwh * 1.1, 2),  # Demo projection
            "previous": round(consumption * self.cost_per_kwh * 0.95, 2),
            "unit": "USD",
        }

    def get_daily_average(self, user_id: Optional[int] = None) -> float:
        """Get daily average consumption."""
        # Demo value - would query database in production
        return 6.5  # kWh per day

    def check_usage_alert(self, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Check for high usage alerts."""
        daily_avg = self.get_daily_average(user_id)
        
        if daily_avg > 10.0:  # High usage threshold
            return {
                "type": "high_electricity_usage",
                "priority": "high",
                "title": "High Electricity Usage Detected",
                "message": f"Your daily average ({daily_avg} kWh) is above recommended levels.",
                "timestamp": datetime.now().isoformat(),
            }
        return None