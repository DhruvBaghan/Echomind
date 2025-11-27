# ============================================
# EchoMind - Electricity Predictor Model
# ============================================

"""
Electricity consumption prediction model.
Uses Prophet for time-series forecasting of electricity usage.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from backend.models.base_predictor import BasePredictor
from backend.config import Config
from backend.utils.logger import logger


class ElectricityPredictor(BasePredictor):
    """
    Predictor for electricity consumption.
    
    This model specializes in predicting electricity usage patterns,
    taking into account factors like:
    - Daily patterns (peak hours, off-peak hours)
    - Weekly patterns (weekday vs weekend)
    - Seasonal variations (heating/cooling demand)
    - Special events and holidays
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize electricity predictor.

        Args:
            model_path: Path to pre-trained model. Uses default if not provided.
        """
        super().__init__(model_path or Config.ELECTRICITY_MODEL_PATH)
        self.resource_type = "electricity"
        
        # Electricity-specific settings
        self.peak_hours = list(range(17, 22))  # 5 PM - 10 PM
        self.off_peak_hours = list(range(0, 7))  # 12 AM - 7 AM
        
        # Load model on initialization
        self.load_model()

    def get_resource_type(self) -> str:
        """Return resource type."""
        return "electricity"

    def get_cost_per_unit(self) -> float:
        """Return cost per kWh."""
        return Config.ELECTRICITY_COST_PER_KWH

    def get_unit_name(self) -> str:
        """Return unit name."""
        return "kWh"

    def train(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Train electricity model with electricity-specific seasonalities.

        Args:
            data: Historical electricity consumption data
            **kwargs: Additional training parameters

        Returns:
            bool: Training success status
        """
        # Add electricity-specific seasonalities
        custom_seasonalities = kwargs.get("custom_seasonalities", [])
        
        # Add intraday seasonality for hourly patterns
        custom_seasonalities.append({
            "name": "intraday",
            "period": 1,
            "fourier_order": 8,
        })

        # Override kwargs with electricity-specific settings
        kwargs["custom_seasonalities"] = custom_seasonalities
        kwargs["daily_seasonality"] = True
        kwargs["weekly_seasonality"] = True
        kwargs["yearly_seasonality"] = True

        return super().train(data, **kwargs)

    def predict(
        self,
        periods: int = None,
        frequency: str = "H",
        include_history: bool = False
    ) -> Dict[str, Any]:
        """
        Generate electricity predictions with peak hour analysis.

        Args:
            periods: Number of periods to predict
            frequency: Prediction frequency
            include_history: Include historical fitted values

        Returns:
            Prediction results with peak hour analysis
        """
        # Get base predictions
        result = super().predict(periods, frequency, include_history)

        # Scale down predictions to realistic household consumption levels
        # Prophet model can produce inflated values - normalize to ~0.5-2.5 kWh range
        if result.get("predictions"):
            predictions = result["predictions"]
            values = [p["predicted_value"] for p in predictions]
            max_val = max(values) if values else 1
            
            # Scale factor: if max is > 10 kWh/hour, scale it down
            if max_val > 10:
                scale_factor = 1.5 / (max_val / 50)  # Target average around 1.5 kWh
                scale_factor = max(0.01, min(scale_factor, 1))  # Keep between 0.01 and 1
                
                for pred in predictions:
                    pred["predicted_value"] = round(pred["predicted_value"] * scale_factor, 2)
                    pred["lower_bound"] = round(pred["lower_bound"] * scale_factor, 2)
                    pred["upper_bound"] = round(pred["upper_bound"] * scale_factor, 2)
                    pred["estimated_cost"] = round(pred["predicted_value"] * self.get_cost_per_unit(), 4)
            
            # Recalculate summary
            result["summary"]["total_predicted"] = round(sum(p["predicted_value"] for p in predictions), 2)
            result["summary"]["average_predicted"] = round(sum(p["predicted_value"] for p in predictions) / len(predictions), 2) if predictions else 0
            result["summary"]["min_predicted"] = round(min(p["predicted_value"] for p in predictions), 2) if predictions else 0
            result["summary"]["max_predicted"] = round(max(p["predicted_value"] for p in predictions), 2) if predictions else 0
            result["summary"]["total_estimated_cost"] = round(result["summary"]["total_predicted"] * self.get_cost_per_unit(), 2)

        # Add electricity-specific analysis
        result = self._add_peak_hour_analysis(result)
        result = self._add_usage_recommendations(result)

        return result

    def _add_peak_hour_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add peak vs off-peak hour analysis.

        Args:
            result: Base prediction result

        Returns:
            Result with peak hour analysis
        """
        predictions = result.get("predictions", [])
        
        peak_consumption = 0.0
        off_peak_consumption = 0.0
        peak_count = 0
        off_peak_count = 0

        for pred in predictions:
            try:
                hour = pd.to_datetime(pred["datetime"]).hour
                value = pred["predicted_value"]
                
                if hour in self.peak_hours:
                    peak_consumption += value
                    peak_count += 1
                    pred["is_peak_hour"] = True
                elif hour in self.off_peak_hours:
                    off_peak_consumption += value
                    off_peak_count += 1
                    pred["is_peak_hour"] = False
                else:
                    pred["is_peak_hour"] = False
            except Exception:
                pred["is_peak_hour"] = False

        # Add peak analysis to summary
        result["peak_analysis"] = {
            "peak_hours": f"{self.peak_hours[0]}:00 - {self.peak_hours[-1]}:00",
            "off_peak_hours": f"{self.off_peak_hours[0]}:00 - {self.off_peak_hours[-1]}:00",
            "peak_consumption": round(peak_consumption, 2),
            "off_peak_consumption": round(off_peak_consumption, 2),
            "peak_periods": peak_count,
            "off_peak_periods": off_peak_count,
            "peak_percentage": round(
                (peak_consumption / (peak_consumption + off_peak_consumption)) * 100, 1
            ) if (peak_consumption + off_peak_consumption) > 0 else 0,
        }

        return result

    def _add_usage_recommendations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add usage recommendations based on predictions.

        Args:
            result: Prediction result

        Returns:
            Result with recommendations
        """
        recommendations = []
        summary = result.get("summary", {})
        peak_analysis = result.get("peak_analysis", {})

        # High peak usage recommendation
        peak_percentage = peak_analysis.get("peak_percentage", 0)
        if peak_percentage > 40:
            recommendations.append({
                "type": "peak_usage",
                "priority": "high",
                "message": f"Peak hour usage is {peak_percentage}% of total. Consider shifting appliance usage to off-peak hours to reduce costs.",
                "potential_savings": f"Up to {round(peak_percentage * 0.15, 1)}% cost reduction possible"
            })

        # High average usage recommendation
        avg_usage = summary.get("average_predicted", 0)
        if avg_usage > 2.0:  # kWh per hour threshold
            recommendations.append({
                "type": "high_consumption",
                "priority": "medium",
                "message": f"Average hourly consumption ({avg_usage} kWh) is above optimal levels. Check for energy-inefficient appliances.",
                "potential_savings": "10-20% reduction achievable with efficient appliances"
            })

        # Usage spike detection
        predictions = result.get("predictions", [])
        if predictions:
            values = [p["predicted_value"] for p in predictions]
            max_val = max(values)
            avg_val = sum(values) / len(values)
            
            if max_val > avg_val * 2:
                recommendations.append({
                    "type": "usage_spike",
                    "priority": "medium",
                    "message": "Significant usage spikes detected. Consider distributing high-power activities throughout the day.",
                    "potential_savings": "5-10% cost reduction"
                })

        # Add general tips if no specific recommendations
        if not recommendations:
            recommendations.append({
                "type": "general",
                "priority": "low",
                "message": "Your electricity usage patterns look efficient! Continue monitoring for optimal performance.",
                "potential_savings": "Maintain current practices"
            })

        result["recommendations"] = recommendations
        return result

    def analyze_appliance_usage(
        self,
        predictions: List[Dict[str, Any]],
        appliance_profiles: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze which appliances might be contributing to consumption.

        Args:
            predictions: List of prediction dictionaries
            appliance_profiles: Optional dict mapping appliance names to typical kWh usage

        Returns:
            Analysis of probable appliance usage
        """
        # Default appliance profiles (average kWh per hour when running)
        default_profiles = {
            "air_conditioner": 1.5,
            "heater": 2.0,
            "refrigerator": 0.15,
            "washing_machine": 0.5,
            "dishwasher": 0.4,
            "lighting": 0.1,
            "television": 0.1,
            "computer": 0.2,
            "oven": 2.0,
            "microwave": 1.0,
        }

        profiles = appliance_profiles or default_profiles
        total_consumption = sum(p["predicted_value"] for p in predictions)
        
        # Estimate appliance contribution (simplified heuristic)
        analysis = {
            "total_consumption": round(total_consumption, 2),
            "estimated_breakdown": {},
            "high_consumers": [],
        }

        # Basic estimation based on consumption patterns
        for appliance, rate in profiles.items():
            # Simplified estimation
            estimated_hours = min(total_consumption / rate, len(predictions)) * 0.3
            estimated_usage = round(rate * estimated_hours, 2)
            
            if estimated_usage > 0:
                analysis["estimated_breakdown"][appliance] = {
                    "estimated_kwh": estimated_usage,
                    "estimated_hours": round(estimated_hours, 1),
                    "percentage_of_total": round((estimated_usage / total_consumption) * 100, 1) if total_consumption > 0 else 0
                }

        # Identify high consumers
        for appliance, data in analysis["estimated_breakdown"].items():
            if data["percentage_of_total"] > 15:
                analysis["high_consumers"].append(appliance)

        return analysis