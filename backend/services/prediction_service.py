# ============================================
# EchoMind - Prediction Service
# ============================================

"""
Unified prediction service.
Handles combined predictions for both electricity and water,
sustainability scoring, and comparison analysis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from backend.models.unified_optimizer import UnifiedOptimizer
from backend.models.electricity_predictor import ElectricityPredictor
from backend.models.water_predictor import WaterPredictor
from backend.config import Config
from backend.utils.logger import logger


class PredictionService:
    """
    Unified prediction service for EchoMind.
    
    Provides methods for:
    - Individual resource predictions
    - Combined predictions
    - Sustainability scoring
    - Period comparisons
    - Accuracy calculations
    """

    def __init__(self):
        """Initialize prediction service."""
        self.unified_optimizer = UnifiedOptimizer()
        self.electricity_predictor = ElectricityPredictor()
        self.water_predictor = WaterPredictor()
        
        # Sustainability benchmarks
        self.electricity_benchmark = 30.0  # kWh per day
        self.water_benchmark = 350.0  # Liters per day

    def predict_electricity(
        self,
        user_data: Optional[List[Dict[str, Any]]] = None,
        periods: int = 24,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate electricity predictions.

        Args:
            user_data: Optional user-provided data
            periods: Number of periods to predict
            frequency: Prediction frequency

        Returns:
            Prediction results
        """
        try:
            if user_data and len(user_data) >= 2:
                return self.electricity_predictor.predict_from_user_data(
                    user_data=user_data,
                    periods=periods,
                    frequency=frequency
                )
            elif self.electricity_predictor.is_loaded:
                return self.electricity_predictor.predict(
                    periods=periods,
                    frequency=frequency
                )
            else:
                return {
                    "success": False,
                    "error": "No data provided and model not loaded"
                }

        except Exception as e:
            logger.error(f"Electricity prediction error: {e}")
            return {
                "success": False,
                "error": "Prediction failed",
                "message": str(e)
            }

    def predict_water(
        self,
        user_data: Optional[List[Dict[str, Any]]] = None,
        periods: int = 24,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate water predictions.

        Args:
            user_data: Optional user-provided data
            periods: Number of periods to predict
            frequency: Prediction frequency

        Returns:
            Prediction results
        """
        try:
            if user_data and len(user_data) >= 2:
                return self.water_predictor.predict_from_user_data(
                    user_data=user_data,
                    periods=periods,
                    frequency=frequency
                )
            elif self.water_predictor.is_loaded:
                return self.water_predictor.predict(
                    periods=periods,
                    frequency=frequency
                )
            else:
                return {
                    "success": False,
                    "error": "No data provided and model not loaded"
                }

        except Exception as e:
            logger.error(f"Water prediction error: {e}")
            return {
                "success": False,
                "error": "Prediction failed",
                "message": str(e)
            }

    def predict_both(
        self,
        electricity_data: Optional[List[Dict[str, Any]]] = None,
        water_data: Optional[List[Dict[str, Any]]] = None,
        electricity_periods: int = 24,
        water_periods: int = 24,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate predictions for both electricity and water.

        Args:
            electricity_data: Electricity consumption data
            water_data: Water consumption data
            electricity_periods: Periods for electricity prediction
            water_periods: Periods for water prediction
            frequency: Prediction frequency

        Returns:
            Combined prediction results
        """
        try:
            result = {
                "success": True,
                "generated_at": datetime.now().isoformat(),
                "resources": {},
            }

            # Get electricity predictions
            electricity_result = self.predict_electricity(
                user_data=electricity_data,
                periods=electricity_periods,
                frequency=frequency
            )
            result["resources"]["electricity"] = electricity_result

            # Get water predictions
            water_result = self.predict_water(
                user_data=water_data,
                periods=water_periods,
                frequency=frequency
            )
            result["resources"]["water"] = water_result

            # Generate combined analysis
            result["combined_analysis"] = self._generate_combined_analysis(
                electricity_result,
                water_result
            )

            # Calculate sustainability score
            result["sustainability"] = self._calculate_sustainability(
                electricity_result,
                water_result
            )

            # Generate unified recommendations
            result["recommendations"] = self._generate_recommendations(result)

            return result

        except Exception as e:
            logger.error(f"Combined prediction error: {e}")
            return {
                "success": False,
                "error": "Combined prediction failed",
                "message": str(e)
            }

    def _generate_combined_analysis(
        self,
        electricity_result: Dict[str, Any],
        water_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate combined analysis from both predictions."""
        analysis = {
            "total_cost": 0.0,
            "cost_breakdown": {},
        }

        # Electricity
        if electricity_result.get("success"):
            elec_summary = electricity_result.get("summary", {})
            elec_cost = elec_summary.get("total_estimated_cost", 0)
            analysis["cost_breakdown"]["electricity"] = {
                "consumption": elec_summary.get("total_predicted", 0),
                "unit": "kWh",
                "cost": elec_cost,
            }
            analysis["total_cost"] += elec_cost

        # Water
        if water_result.get("success"):
            water_summary = water_result.get("summary", {})
            water_cost = water_summary.get("total_estimated_cost", 0)
            analysis["cost_breakdown"]["water"] = {
                "consumption": water_summary.get("total_predicted", 0),
                "unit": "liters",
                "cost": water_cost,
            }
            analysis["total_cost"] += water_cost

        analysis["total_cost"] = round(analysis["total_cost"], 2)
        
        return analysis

    def _calculate_sustainability(
        self,
        electricity_result: Dict[str, Any],
        water_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate sustainability score from predictions."""
        scores = {}
        
        # Electricity score
        if electricity_result.get("success"):
            elec_summary = electricity_result.get("summary", {})
            periods = elec_summary.get("periods", 24)
            total = elec_summary.get("total_predicted", 0)
            daily = total * (24 / periods) if periods > 0 else 0
            
            ratio = daily / self.electricity_benchmark
            score = max(0, min(100, 100 - (ratio - 1) * 50))
            
            scores["electricity"] = {
                "score": round(score, 1),
                "daily_consumption": round(daily, 2),
                "benchmark": self.electricity_benchmark,
            }

        # Water score
        if water_result.get("success"):
            water_summary = water_result.get("summary", {})
            periods = water_summary.get("periods", 24)
            total = water_summary.get("total_predicted", 0)
            daily = total * (24 / periods) if periods > 0 else 0
            
            ratio = daily / self.water_benchmark
            score = max(0, min(100, 100 - (ratio - 1) * 50))
            
            scores["water"] = {
                "score": round(score, 1),
                "daily_consumption": round(daily, 2),
                "benchmark": self.water_benchmark,
            }

        # Overall score
        individual_scores = [s["score"] for s in scores.values()]
        overall = sum(individual_scores) / len(individual_scores) if individual_scores else 0

        return {
            "overall_score": round(overall, 1),
            "grade": self._get_grade(overall),
            "individual_scores": scores,
        }

    def _get_grade(self, score: float) -> str:
        """Get letter grade from score."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on predictions."""
        recommendations = []
        
        sustainability = result.get("sustainability", {})
        overall_score = sustainability.get("overall_score", 100)
        
        if overall_score < 60:
            recommendations.append({
                "priority": "high",
                "category": "general",
                "title": "Improve Resource Efficiency",
                "message": f"Your sustainability score ({overall_score}) indicates room for improvement.",
            })

        if overall_score >= 80:
            recommendations.append({
                "priority": "low",
                "category": "general",
                "title": "Great Performance",
                "message": "You're doing well! Keep maintaining these efficient practices.",
            })

        return recommendations

    def get_demo_predictions(
        self,
        resource: str = "both",
        periods: int = 24,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Get demo predictions using pre-trained models.

        Args:
            resource: Resource type (electricity/water/both)
            periods: Number of periods
            frequency: Prediction frequency

        Returns:
            Demo predictions
        """
        try:
            result = {
                "success": True,
                "generated_at": datetime.now().isoformat(),
            }

            if resource in ["electricity", "both"]:
                if self.electricity_predictor.is_loaded:
                    result["electricity"] = self.electricity_predictor.predict(
                        periods=periods,
                        frequency=frequency
                    )
                else:
                    result["electricity"] = self._generate_synthetic_predictions(
                        "electricity", periods
                    )

            if resource in ["water", "both"]:
                if self.water_predictor.is_loaded:
                    result["water"] = self.water_predictor.predict(
                        periods=periods,
                        frequency=frequency
                    )
                else:
                    result["water"] = self._generate_synthetic_predictions(
                        "water", periods
                    )

            return result

        except Exception as e:
            logger.error(f"Demo prediction error: {e}")
            return {
                "success": False,
                "error": "Demo prediction failed",
                "message": str(e)
            }

    def _generate_synthetic_predictions(
        self,
        resource_type: str,
        periods: int
    ) -> Dict[str, Any]:
        """Generate synthetic predictions when model isn't available."""
        base_time = datetime.now()
        
        # Different patterns for different resources
        if resource_type == "electricity":
            base_value = 1.5
            variation = 0.5
            unit = "kWh"
            cost_per_unit = Config.ELECTRICITY_COST_PER_KWH
        else:
            base_value = 25
            variation = 10
            unit = "liters"
            cost_per_unit = Config.WATER_COST_PER_LITER

        predictions = []
        for i in range(periods):
            time = base_time + timedelta(hours=i)
            hour = time.hour
            
            # Add daily pattern
            if 6 <= hour <= 9 or 17 <= hour <= 21:
                multiplier = 1.5  # Peak hours
            elif 0 <= hour <= 5:
                multiplier = 0.5  # Night
            else:
                multiplier = 1.0

            value = base_value * multiplier + np.random.uniform(-variation, variation)
            value = max(0, value)
            
            predictions.append({
                "datetime": time.isoformat(),
                "predicted_value": round(value, 2),
                "lower_bound": round(value * 0.8, 2),
                "upper_bound": round(value * 1.2, 2),
                "estimated_cost": round(value * cost_per_unit, 4),
            })

        total = sum(p["predicted_value"] for p in predictions)
        
        return {
            "success": True,
            "resource_type": resource_type,
            "unit": unit,
            "predictions": predictions,
            "summary": {
                "total_predicted": round(total, 2),
                "average_predicted": round(total / periods, 2),
                "periods": periods,
                "total_estimated_cost": round(total * cost_per_unit, 2),
            },
            "synthetic": True,
            "note": "Synthetic predictions - model not loaded",
        }

    def compare_periods(
        self,
        period1_data: Dict[str, List[Dict[str, Any]]],
        period2_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compare consumption between two periods.

        Args:
            period1_data: First period data
            period2_data: Second period data

        Returns:
            Comparison results
        """
        comparison = {}

        for resource in ["electricity", "water"]:
            p1 = period1_data.get(resource, [])
            p2 = period2_data.get(resource, [])

            if p1 and p2:
                p1_total = sum(d.get("consumption", 0) for d in p1)
                p2_total = sum(d.get("consumption", 0) for d in p2)
                
                change = p2_total - p1_total
                change_pct = ((change / p1_total) * 100) if p1_total > 0 else 0

                comparison[resource] = {
                    "period1_total": round(p1_total, 2),
                    "period2_total": round(p2_total, 2),
                    "absolute_change": round(change, 2),
                    "percentage_change": round(change_pct, 1),
                    "trend": "increased" if change > 0 else "decreased" if change < 0 else "stable",
                }

        return comparison

    def calculate_sustainability_score(
        self,
        electricity_daily: Optional[float] = None,
        water_daily: Optional[float] = None,
        household_size: int = 4
    ) -> Dict[str, Any]:
        """
        Calculate sustainability score.

        Args:
            electricity_daily: Daily electricity consumption (kWh)
            water_daily: Daily water consumption (liters)
            household_size: Number of people in household

        Returns:
            Sustainability score and breakdown
        """
        scores = {}
        
        # Adjust benchmarks for household size
        elec_benchmark = self.electricity_benchmark * (household_size / 4)
        water_benchmark = self.water_benchmark * (household_size / 4)

        if electricity_daily is not None:
            ratio = electricity_daily / elec_benchmark
            score = max(0, min(100, 100 - (ratio - 1) * 50))
            scores["electricity"] = {
                "score": round(score, 1),
                "consumption": electricity_daily,
                "benchmark": round(elec_benchmark, 2),
                "status": self._get_status(score),
            }

        if water_daily is not None:
            ratio = water_daily / water_benchmark
            score = max(0, min(100, 100 - (ratio - 1) * 50))
            scores["water"] = {
                "score": round(score, 1),
                "consumption": water_daily,
                "benchmark": round(water_benchmark, 2),
                "status": self._get_status(score),
            }

        # Calculate overall
        individual = [s["score"] for s in scores.values()]
        overall = sum(individual) / len(individual) if individual else 0

        return {
            "overall_score": round(overall, 1),
            "grade": self._get_grade(overall),
            "status": self._get_status(overall),
            "individual_scores": scores,
            "household_size": household_size,
        }

    def _get_status(self, score: float) -> str:
        """Get status from score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "needs_improvement"

    def calculate_accuracy(
        self,
        resource_type: str,
        predictions: List[Dict[str, Any]],
        actuals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy.

        Args:
            resource_type: Type of resource
            predictions: List of predictions
            actuals: List of actual values

        Returns:
            Accuracy metrics
        """
        try:
            # Match predictions with actuals
            pred_values = []
            actual_values = []

            pred_dict = {p["datetime"]: p["predicted"] for p in predictions}
            
            for actual in actuals:
                dt = actual["datetime"]
                if dt in pred_dict:
                    pred_values.append(pred_dict[dt])
                    actual_values.append(actual["actual"])

            if not pred_values:
                return {"error": "No matching data points found"}

            pred_arr = np.array(pred_values)
            actual_arr = np.array(actual_values)

            # Calculate metrics
            mae = float(np.mean(np.abs(pred_arr - actual_arr)))
            mse = float(np.mean((pred_arr - actual_arr) ** 2))
            rmse = float(np.sqrt(mse))
            mape = float(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100) if np.all(actual_arr != 0) else None

            # R-squared
            ss_res = np.sum((actual_arr - pred_arr) ** 2)
            ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
            r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else None

            return {
                "resource_type": resource_type,
                "data_points": len(pred_values),
                "metrics": {
                    "mae": round(mae, 4),
                    "mse": round(mse, 4),
                    "rmse": round(rmse, 4),
                    "mape": round(mape, 2) if mape is not None else None,
                    "r_squared": round(r_squared, 4) if r_squared is not None else None,
                },
                "accuracy_percentage": round(100 - (mape or 0), 1) if mape else None,
            }

        except Exception as e:
            logger.error(f"Accuracy calculation error: {e}")
            return {"error": str(e)}

    def quick_predict(
        self,
        resource_type: str,
        recent_average: float,
        periods: int = 24
    ) -> Dict[str, Any]:
        """
        Generate quick predictions based on recent average.

        Args:
            resource_type: Type of resource
            recent_average: Recent average consumption
            periods: Number of periods

        Returns:
            Quick predictions
        """
        base_time = datetime.now()
        
        if resource_type == "electricity":
            unit = "kWh"
            cost_per_unit = Config.ELECTRICITY_COST_PER_KWH
        else:
            unit = "liters"
            cost_per_unit = Config.WATER_COST_PER_LITER

        predictions = []
        for i in range(periods):
            time = base_time + timedelta(hours=i)
            
            # Simple variation based on time of day
            hour = time.hour
            if 6 <= hour <= 9 or 17 <= hour <= 21:
                value = recent_average * 1.3
            elif 0 <= hour <= 5:
                value = recent_average * 0.5
            else:
                value = recent_average

            predictions.append({
                "datetime": time.isoformat(),
                "predicted_value": round(value, 2),
                "estimated_cost": round(value * cost_per_unit, 4),
            })

        total = sum(p["predicted_value"] for p in predictions)

        return {
            "success": True,
            "resource_type": resource_type,
            "unit": unit,
            "predictions": predictions,
            "summary": {
                "total_predicted": round(total, 2),
                "average_predicted": round(total / periods, 2),
                "periods": periods,
                "total_estimated_cost": round(total * cost_per_unit, 2),
            },
            "method": "quick_prediction",
            "based_on": f"Recent average: {recent_average} {unit}/hour",
        }

    def get_predictions_summary(self, periods: int = 24) -> Dict[str, Any]:
        """Get predictions summary for dashboard."""
        electricity_summary = self._generate_synthetic_predictions("electricity", periods).get("summary", {})
        water_summary = self._generate_synthetic_predictions("water", periods).get("summary", {})
        
        return {
            "electricity": electricity_summary,
            "water": water_summary,
            "available": True,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "electricity": {
                "loaded": self.electricity_predictor.is_loaded,
                "info": self.electricity_predictor.get_model_info() if self.electricity_predictor.is_loaded else None,
            },
            "water": {
                "loaded": self.water_predictor.is_loaded,
                "info": self.water_predictor.get_model_info() if self.water_predictor.is_loaded else None,
            },
        }