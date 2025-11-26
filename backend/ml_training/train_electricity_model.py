# ============================================
# EchoMind - Electricity Model Training
# ============================================

"""
Training script for the electricity consumption prediction model.

Uses Facebook Prophet for time-series forecasting.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.config import Config
from backend.utils.logger import logger
from backend.ml_training.data_preprocessing import preprocess_electricity_data


class ElectricityModelTrainer:
    """
    Trainer class for electricity consumption prediction model.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        confidence_interval: float = 0.95
    ):
        """
        Initialize the trainer.
        
        Args:
            output_dir: Directory to save trained model
            confidence_interval: Confidence interval for predictions
        """
        self.output_dir = Path(output_dir or Config.ML_MODELS_DIR)
        self.confidence_interval = confidence_interval
        self.model = None
        self.metrics = {}
        self.training_info = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate training data.
        
        Args:
            data_path: Path to training data CSV
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows of data")
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for Prophet.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame with 'ds' and 'y' columns
        """
        logger.info("Preprocessing data...")
        
        # Use the preprocessing module
        processed = preprocess_electricity_data(df)
        
        logger.info(f"Preprocessed data: {len(processed)} rows")
        return processed
    
    def create_model(self) -> Prophet:
        """
        Create and configure Prophet model.
        
        Returns:
            Configured Prophet model
        """
        logger.info("Creating Prophet model...")
        
        model = Prophet(
            interval_width=self.confidence_interval,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )
        
        # Add custom seasonalities for electricity patterns
        model.add_seasonality(
            name="intraday",
            period=1,
            fourier_order=8,
        )
        
        # Add holiday effects (optional)
        # model.add_country_holidays(country_name='US')
        
        return model
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[Prophet, Dict[str, float]]:
        """
        Train the model and evaluate performance.
        
        Args:
            df: Preprocessed DataFrame
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        logger.info("Starting model training...")
        
        # Split data
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Training set: {len(train_df)} rows")
        logger.info(f"Test set: {len(test_df)} rows")
        
        # Create and fit model
        self.model = self.create_model()
        
        start_time = datetime.now()
        self.model.fit(train_df)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
        
        # Evaluate on test set
        metrics = self.evaluate(test_df)
        metrics["training_time_seconds"] = training_time
        metrics["training_samples"] = len(train_df)
        metrics["test_samples"] = len(test_df)
        
        self.metrics = metrics
        self.training_info = {
            "trained_at": datetime.now().isoformat(),
            "training_samples": len(train_df),
            "test_samples": len(test_df),
            "data_date_range": {
                "start": df["ds"].min().isoformat(),
                "end": df["ds"].max().isoformat(),
            },
        }
        
        return self.model, metrics
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Generate predictions for test period
        forecast = self.model.predict(test_df[["ds"]])
        
        # Calculate metrics
        y_true = test_df["y"].values
        y_pred = forecast["yhat"].values
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
            "accuracy_percentage": round(100 - mape, 2),
        }
        
        logger.info(f"Evaluation metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        return metrics
    
    def save_model(self, filename: str = "electricity_model.pkl") -> str:
        """
        Save trained model to disk.
        
        Args:
            filename: Model filename
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = self.output_dir / filename
        joblib.dump(self.model, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        self._save_metadata(filename)
        
        return str(model_path)
    
    def _save_metadata(self, model_filename: str) -> None:
        """Save model metadata to JSON file."""
        metadata_path = self.output_dir / "model_metadata.json"
        
        # Load existing metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        # Update electricity model metadata
        metadata["electricity"] = {
            "model_file": model_filename,
            "metrics": self.metrics,
            "training_info": self.training_info,
            "model_type": "Prophet",
            "confidence_interval": self.confidence_interval,
            "updated_at": datetime.now().isoformat(),
        }
        
        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        initial: str = "365 days",
        period: str = "30 days",
        horizon: str = "7 days"
    ) -> Dict[str, float]:
        """
        Perform cross-validation for more robust evaluation.
        
        Args:
            df: Full dataset
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            Cross-validation metrics
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        logger.info("Performing cross-validation...")
        
        if self.model is None:
            self.model = self.create_model()
            self.model.fit(df)
        
        # Run cross-validation
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        # Calculate performance metrics
        cv_metrics = performance_metrics(cv_results)
        
        return {
            "cv_mae": round(cv_metrics["mae"].mean(), 4),
            "cv_rmse": round(cv_metrics["rmse"].mean(), 4),
            "cv_mape": round(cv_metrics["mape"].mean() * 100, 2),
        }


def train_electricity_model(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to train electricity model.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save model
        
    Returns:
        Training result dictionary
    """
    try:
        # Set default paths
        if data_path is None:
            data_path = str(Config.RAW_DATA_DIR / "electricity_training_data.csv")
        
        if output_dir is None:
            output_dir = str(Config.ML_MODELS_DIR)
        
        # Initialize trainer
        trainer = ElectricityModelTrainer(output_dir=output_dir)
        
        # Load and preprocess data
        df = trainer.load_data(data_path)
        processed_df = trainer.preprocess(df)
        
        # Train model
        model, metrics = trainer.train(processed_df)
        
        # Save model
        model_path = trainer.save_model()
        
        return {
            "success": True,
            "model_path": model_path,
            "metrics": metrics,
            "training_info": trainer.training_info,
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def generate_sample_data(
    output_path: Optional[str] = None,
    days: int = 365,
    frequency: str = "H"
) -> str:
    """
    Generate sample electricity training data.
    
    Args:
        output_path: Path to save generated data
        days: Number of days of data to generate
        frequency: Data frequency ('H' for hourly)
        
    Returns:
        Path to generated file
    """
    import numpy as np
    
    if output_path is None:
        output_path = str(Config.RAW_DATA_DIR / "electricity_training_data.csv")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamps
    periods = days * 24 if frequency == "H" else days
    timestamps = pd.date_range(
        start=datetime.now() - pd.Timedelta(days=days),
        periods=periods,
        freq=frequency
    )
    
    # Generate consumption with realistic patterns
    data = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month
        
        # Base consumption
        base = 1.5
        
        # Daily pattern (higher in morning and evening)
        if 6 <= hour <= 9:
            daily_factor = 1.5
        elif 17 <= hour <= 22:
            daily_factor = 1.8
        elif 0 <= hour <= 5:
            daily_factor = 0.5
        else:
            daily_factor = 1.0
        
        # Weekly pattern (higher on weekends)
        weekly_factor = 1.2 if day_of_week >= 5 else 1.0
        
        # Seasonal pattern (higher in summer/winter for HVAC)
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.4
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.3
        else:
            seasonal_factor = 1.0
        
        # Add noise
        noise = np.random.normal(0, 0.3)
        
        consumption = base * daily_factor * weekly_factor * seasonal_factor + noise
        consumption = max(0.1, consumption)  # Ensure positive
        
        data.append({
            "datetime": ts,
            "consumption": round(consumption, 2),
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Generated {len(df)} rows of sample data to {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate sample data if not exists
    data_path = Config.RAW_DATA_DIR / "electricity_training_data.csv"
    if not data_path.exists():
        logger.info("Training data not found, generating sample data...")
        generate_sample_data()
    
    # Train model
    result = train_electricity_model()
    print(json.dumps(result, indent=2))