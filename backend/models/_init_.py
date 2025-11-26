# ============================================
# EchoMind - ML Models Package Initialization
# ============================================

"""
Machine Learning Models Package

This package contains the prediction models for EchoMind:

- BasePredictor: Abstract base class for all predictors
- ElectricityPredictor: Model for electricity consumption prediction
- WaterPredictor: Model for water consumption prediction
- UnifiedOptimizer: Combined optimization for both resources

All models use Facebook's Prophet for time-series forecasting.
"""

from backend.models.base_predictor import BasePredictor
from backend.models.electricity_predictor import ElectricityPredictor
from backend.models.water_predictor import WaterPredictor
from backend.models.unified_optimizer import UnifiedOptimizer

# Package exports
__all__ = [
    "BasePredictor",
    "ElectricityPredictor",
    "WaterPredictor",
    "UnifiedOptimizer",
]

# Model registry for easy access
MODEL_REGISTRY = {
    "electricity": ElectricityPredictor,
    "water": WaterPredictor,
    "unified": UnifiedOptimizer,
}


def get_model(model_type: str):
    """
    Factory function to get model instance by type.

    Args:
        model_type: Type of model ("electricity", "water", or "unified")

    Returns:
        Model instance

    Raises:
        ValueError: If model type is not recognized
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]()


def list_available_models():
    """
    List all available model types.

    Returns:
        List of available model type names
    """
    return list(MODEL_REGISTRY.keys())