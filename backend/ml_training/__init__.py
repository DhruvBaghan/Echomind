# ============================================
# EchoMind - ML Training Package Initialization
# ============================================

"""
Machine Learning Training Package

This package contains scripts and utilities for training the prediction models
used in EchoMind.

Modules:
    - train_electricity_model: Training script for electricity model
    - train_water_model: Training script for water model
    - data_preprocessing: Data preprocessing utilities

The trained models are saved to the ml_models/ directory and used by
the prediction services at runtime.

Training Process:
    1. Load raw data from datasets/raw/
    2. Preprocess and clean data
    3. Train Prophet model
    4. Evaluate model performance
    5. Save model to ml_models/
"""

from backend.ml_training.train_electricity_model import (
    train_electricity_model,
    ElectricityModelTrainer,
)
from backend.ml_training.train_water_model import (
    train_water_model,
    WaterModelTrainer,
)
from backend.ml_training.data_preprocessing import (
    preprocess_electricity_data,
    preprocess_water_data,
    DataPreprocessor,
)

# Package exports
__all__ = [
    "train_electricity_model",
    "train_water_model",
    "preprocess_electricity_data",
    "preprocess_water_data",
    "ElectricityModelTrainer",
    "WaterModelTrainer",
    "DataPreprocessor",
]


def train_all_models(
    electricity_data_path: str = None,
    water_data_path: str = None,
    output_dir: str = None
) -> dict:
    """
    Train all prediction models.
    
    Args:
        electricity_data_path: Path to electricity training data
        water_data_path: Path to water training data
        output_dir: Directory to save trained models
        
    Returns:
        Dictionary with training results for each model
    """
    from backend.config import Config
    from backend.utils.logger import logger
    
    results = {}
    
    # Set default paths
    if electricity_data_path is None:
        electricity_data_path = str(Config.RAW_DATA_DIR / "electricity_training_data.csv")
    
    if water_data_path is None:
        water_data_path = str(Config.RAW_DATA_DIR / "water_training_data.csv")
    
    if output_dir is None:
        output_dir = str(Config.ML_MODELS_DIR)
    
    logger.info("Starting training for all models...")
    
    # Train electricity model
    try:
        logger.info("Training electricity model...")
        electricity_result = train_electricity_model(
            data_path=electricity_data_path,
            output_dir=output_dir
        )
        results["electricity"] = electricity_result
        logger.info(f"Electricity model training: {'SUCCESS' if electricity_result.get('success') else 'FAILED'}")
    except Exception as e:
        logger.error(f"Electricity model training failed: {e}")
        results["electricity"] = {"success": False, "error": str(e)}
    
    # Train water model
    try:
        logger.info("Training water model...")
        water_result = train_water_model(
            data_path=water_data_path,
            output_dir=output_dir
        )
        results["water"] = water_result
        logger.info(f"Water model training: {'SUCCESS' if water_result.get('success') else 'FAILED'}")
    except Exception as e:
        logger.error(f"Water model training failed: {e}")
        results["water"] = {"success": False, "error": str(e)}
    
    logger.info("Model training complete")
    return results


def get_training_status() -> dict:
    """
    Get status of trained models.
    
    Returns:
        Dictionary with model status information
    """
    import os
    from pathlib import Path
    from backend.config import Config
    
    status = {
        "electricity": {
            "model_exists": False,
            "model_path": None,
            "last_modified": None,
        },
        "water": {
            "model_exists": False,
            "model_path": None,
            "last_modified": None,
        },
    }
    
    # Check electricity model
    elec_path = Path(Config.ELECTRICITY_MODEL_PATH)
    if elec_path.exists():
        status["electricity"]["model_exists"] = True
        status["electricity"]["model_path"] = str(elec_path)
        status["electricity"]["last_modified"] = os.path.getmtime(elec_path)
    
    # Check water model
    water_path = Path(Config.WATER_MODEL_PATH)
    if water_path.exists():
        status["water"]["model_exists"] = True
        status["water"]["model_path"] = str(water_path)
        status["water"]["last_modified"] = os.path.getmtime(water_path)
    
    return status
