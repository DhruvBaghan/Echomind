# ============================================
# EchoMind - Unified Resource Optimizer
# ============================================

"""
Unified optimizer for combined electricity and water consumption.
Provides holistic resource management and optimization recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from backend.models.electricity_predictor import ElectricityPredictor
from backend.models.water_predictor import WaterPredictor
from backend.config import Config
from backend.utils.logger import logger


class UnifiedOptimizer:
    """
    Unified optimizer for both electricity and water resources.
    
    This class combines predictions from both models and provides:
    - Combined resource predictions
    - Holistic cost analysis
    - Cross-resource optimization recommendations
    - Sustainability scoring
    """

    def __init__(self):
        """Initialize unified optimizer with both predictors."""
        self.electricity_predictor = ElectricityPredictor()
        self.water_predictor = WaterPredictor()
        
        # Sustainability benchmarks
        self.daily_electricity_benchmark = 30.0  # kWh per day (average household)
        self.daily_water_benchmark = 350.0  # liters per day (average household)

    def load_models(self) -> Dict[str, bool]:
        """
        Load both prediction models.

        Returns:
            Dict with load status for each model
        """
        return {
            "electricity": self.electricity_predictor.load_model(),
            "water": self.water_predictor.load_model(),
        }

    def predict_all(
        self,
        periods: int = None,
        frequency: str = "H",
        include_history: bool = False
    ) -> Dict[str, Any]:
        """
        Generate predictions for both electricity and water.

        Args:
            periods: Number of periods to predict
            frequency: Prediction frequency
            include_history: Include historical fitted values

        Returns:
            Combined prediction results for both resources
        """
        results = {
            "success": True,
            "generated_at": datetime.now().isoformat(),
            "resources": {},
            "combined_analysis": {},
        }

        try:
            # Get electricity predictions
            if self.electricity_predictor.is_loaded:
                results["resources"]["electricity"] = self.electricity_predictor.predict(
                    periods=periods,
                    frequency=frequency,
                    include_history=include_history
                )
            else:
                results["resources"]["electricity"] = {
                    "success": False,
                    "error": "Electricity model not loaded"
                }

            # Get water predictions
            if self.water_predictor.is_loaded:
                results["resources"]["water"] = self.water_predictor.predict(
                    periods=periods,
                    frequency=frequency,
                    include_history=include_history
                )
            else:
                results["resources"]["water"] = {
                    "success": False,
                    "error": "Water model not loaded"
                }

            # Generate combined analysis
            results["combined_analysis"] = self._generate_combined_analysis(results["resources"])
            
            # Add sustainability score
            results["sustainability"] = self._calculate_sustainability_score(results["resources"])
            
            # Add unified recommendations
            results["unified_recommendations"] = self._generate_unified_recommendations(results)

        except Exception as e:
            logger.error(f"Error in unified prediction: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    def predict_from_user_data(
        self,
        electricity_data: Optional[List[Dict[str, Any]]] = None,
        water_data: Optional[List[Dict[str, Any]]] = None,
        periods: int = None,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate predictions based on user-provided data for both resources.

        Args:
            electricity_data: User's electricity consumption history
            water_data: User's water consumption history
            periods: Number of periods to predict
            frequency: Prediction frequency

        Returns:
            Combined prediction results
        """
        results = {
            "success": True,
            "generated_at": datetime.now().isoformat(),
            "resources": {},
            "combined_analysis": {},
        }

        try:
            # Process electricity data
            if electricity_data:
                results["resources"]["electricity"] = self.electricity_predictor.predict_from_user_data(
                    user_data=electricity_data,
                    periods=periods,
                    frequency=frequency
                )
            else:
                results["resources"]["electricity"] = {
                    "success": False,
                    "error": "No electricity data provided"
                }

            # Process water data
            if water_data:
                results["resources"]["water"] = self.water_predictor.predict_from_user_data(
                    user_data=water_data,
                    periods=periods,
                    frequency=frequency
                )
            else:
                results["resources"]["water"] = {
                    "success": False,
                    "error": "No water data provided"
                }

            # Generate combined analysis
            results["combined_analysis"] = self._generate_combined_analysis(results["resources"])
            
            # Add sustainability score
            results["sustainability"] = self._calculate_sustainability_score(results["resources"])
            
            # Add unified recommendations
            results["unified_recommendations"] = self._generate_unified_recommendations(results)

        except Exception as e:
            logger.error(f"Error in unified user prediction: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    def _generate_combined_analysis(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined analysis for both resources.

        Args:
            resources: Dictionary containing predictions for each resource

        Returns:
            Combined analysis dictionary
        """
        analysis = {
            "total_cost": 0.0,
            "cost_breakdown": {},
            "consumption_summary": {},
        }

        # Electricity analysis
        elec_result = resources.get("electricity", {})
        if elec_result.get("success"):
            elec_summary = elec_result.get("summary", {})
            elec_cost = elec_summary.get("total_estimated_cost", 0)
            
            analysis["cost_breakdown"]["electricity"] = {
                "total_consumption": elec_summary.get("total_predicted", 0),
                "unit": "kWh",
                "total_cost": elec_cost,
                "average_hourly": elec_summary.get("average_predicted", 0),
            }
            analysis["total_cost"] += elec_cost

        # Water analysis
        water_result = resources.get("water", {})
        if water_result.get("success"):
            water_summary = water_result.get("summary", {})
            water_cost = water_summary.get("total_estimated_cost", 0)
            
            analysis["cost_breakdown"]["water"] = {
                "total_consumption": water_summary.get("total_predicted", 0),
                "unit": "liters",
                "total_cost": water_cost,
                "average_hourly": water_summary.get("average_predicted", 0),
            }
            analysis["total_cost"] += water_cost

        analysis["total_cost"] = round(analysis["total_cost"], 2)
        
        # Calculate cost percentages
        if analysis["total_cost"] > 0:
            for resource, data in analysis["cost_breakdown"].items():
                data["percentage_of_total"] = round(
                    (data["total_cost"] / analysis["total_cost"]) * 100, 1
                )

        return analysis

    def _calculate_sustainability_score(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate sustainability score based on consumption vs benchmarks.

        Args:
            resources: Dictionary containing predictions for each resource

        Returns:
            Sustainability score and breakdown
        """
        scores = {}
        
        # Electricity score
        elec_result = resources.get("electricity", {})
        if elec_result.get("success"):
            elec_summary = elec_result.get("summary", {})
            periods = elec_summary.get("periods", 24)
            daily_consumption = elec_summary.get("total_predicted", 0) * (24 / periods) if periods > 0 else 0
            
            elec_ratio = daily_consumption / self.daily_electricity_benchmark if self.daily_electricity_benchmark > 0 else 1
            elec_score = max(0, min(100, 100 - (elec_ratio - 1) * 50))
            
            scores["electricity"] = {
                "score": round(elec_score, 1),
                "daily_consumption": round(daily_consumption, 2),
                "benchmark": self.daily_electricity_benchmark,
                "status": self._get_status(elec_score),
            }

        # Water score
        water_result = resources.get("water", {})
        if water_result.get("success"):
            water_summary = water_result.get("summary", {})
            periods = water_summary.get("periods", 24)
            daily_consumption = water_summary.get("total_predicted", 0) * (24 / periods) if periods > 0 else 0
            
            water_ratio = daily_consumption / self.daily_water_benchmark if self.daily_water_benchmark > 0 else 1
            water_score = max(0, min(100, 100 - (water_ratio - 1) * 50))
            
            scores["water"] = {
                "score": round(water_score, 1),
                "daily_consumption": round(daily_consumption, 2),
                "benchmark": self.daily_water_benchmark,
                "status": self._get_status(water_score),
            }

        # Overall score
        individual_scores = [s.get("score", 0) for s in scores.values()]
        overall_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0
        
        return {
            "overall_score": round(overall_score, 1),
            "overall_status": self._get_status(overall_score),
            "individual_scores": scores,
            "grade": self._get_grade(overall_score),
        }

    def _get_status(self, score: float) -> str:
        """Get status label based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "needs_improvement"

    def _get_grade(self, score: float) -> str:
        """Get letter grade based on score."""
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

    def _generate_unified_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate unified recommendations across both resources.

        Args:
            results: Complete prediction results

        Returns:
            List of unified recommendations
        """
        recommendations = []
        resources = results.get("resources", {})
        sustainability = results.get("sustainability", {})

        # Check sustainability scores
        overall_score = sustainability.get("overall_score", 100)
        
        if overall_score < 60:
            recommendations.append({
                "type": "sustainability",
                "priority": "high",
                "title": "Improve Resource Efficiency",
                "message": f"Your overall sustainability score is {overall_score}. Focus on reducing consumption in both electricity and water.",
                "impact": "high",
            })

        # Cross-resource optimization
        elec_result = resources.get("electricity", {})
        water_result = resources.get("water", {})
        
        if elec_result.get("success") and water_result.get("success"):
            # Check for water heater optimization
            elec_peak = elec_result.get("peak_analysis", {})
            water_morning = water_result.get("usage_patterns", {}).get("morning_peak", {})
            
            if elec_peak.get("peak_percentage", 0) > 30 and water_morning.get("percentage", 0) > 30:
                recommendations.append({
                    "type": "cross_resource",
                    "priority": "medium",
                    "title": "Water Heater Optimization",
                    "message": "High morning water usage coincides with peak electricity. Consider a timer for your water heater to heat during off-peak hours.",
                    "impact": "medium",
                    "potential_savings": "10-15% on electricity",
                })

        # Smart home recommendation
        combined_cost = results.get("combined_analysis", {}).get("total_cost", 0)
        if combined_cost > 50:  # Monthly cost threshold
            recommendations.append({
                "type": "technology",
                "priority": "low",
                "title": "Consider Smart Monitoring",
                "message": "Based on your usage patterns, smart meters and IoT sensors could help identify inefficiencies and reduce costs.",
                "impact": "medium",
                "potential_savings": "15-25% overall reduction",
            })

        # Behavioral recommendations
        recommendations.append({
            "type": "behavioral",
            "priority": "low",
            "title": "Daily Habits",
            "message": "Small changes like turning off lights, fixing dripping taps, and running full loads can make a significant difference.",
            "impact": "medium",
            "potential_savings": "5-10% on both resources",
        })

        return recommendations

    def get_optimizer_info(self) -> Dict[str, Any]:
        """
        Get information about the optimizer and loaded models.

        Returns:
            Dictionary with optimizer information
        """
        return {
            "name": "EchoMind Unified Optimizer",
            "version": "1.0.0",
            "models": {
                "electricity": self.electricity_predictor.get_model_info(),
                "water": self.water_predictor.get_model_info(),
            },
            "benchmarks": {
                "daily_electricity_kwh": self.daily_electricity_benchmark,
                "daily_water_liters": self.daily_water_benchmark,
            },
        }

    def compare_periods(
        self,
        period1_data: Dict[str, List[Dict[str, Any]]],
        period2_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compare consumption between two time periods.

        Args:
            period1_data: First period data with 'electricity' and 'water' keys
            period2_data: Second period data with 'electricity' and 'water' keys

        Returns:
            Comparison analysis
        """
        comparison = {
            "electricity": {},
            "water": {},
            "overall": {},
        }

        for resource in ["electricity", "water"]:
            p1_data = period1_data.get(resource, [])
            p2_data = period2_data.get(resource, [])

            if p1_data and p2_data:
                p1_total = sum(d.get("consumption", 0) for d in p1_data)
                p2_total = sum(d.get("consumption", 0) for d in p2_data)
                
                change = p2_total - p1_total
                change_pct = ((p2_total - p1_total) / p1_total * 100) if p1_total > 0 else 0

                comparison[resource] = {
                    "period1_total": round(p1_total, 2),
                    "period2_total": round(p2_total, 2),
                    "absolute_change": round(change, 2),
                    "percentage_change": round(change_pct, 1),
                    "trend": "increased" if change > 0 else "decreased" if change < 0 else "stable",
                }

        return comparison