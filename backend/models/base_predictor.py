# ============================================
# EchoMind - Base Predictor Model
# ============================================

"""
Abstract base class for all prediction models.
Provides common interface and shared functionality for electricity and water predictors.
"""

import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
try:
    from prophet import Prophet
except Exception:
    Prophet = None

from backend.config import Config
from backend.utils.logger import logger


class BasePredictor(ABC):
    """
    Abstract base class for resource consumption predictors.
    
    This class provides the common interface and shared functionality
    for all prediction models in EchoMind.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the base predictor.

        Args:
            model_path: Path to the pre-trained model file (.pkl)
        """
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        self.model_metadata: Dict[str, Any] = {}
        self.resource_type = "base"  # Override in subclasses
        
        # Default prediction settings
        self.default_periods = Config.DEFAULT_PREDICTION_PERIODS
        self.max_periods = Config.MAX_PREDICTION_PERIODS
        self.confidence_interval = Config.MODEL_CONFIDENCE_INTERVAL

    @abstractmethod
    def get_resource_type(self) -> str:
        """Return the resource type (electricity/water)."""
        pass

    @abstractmethod
    def get_cost_per_unit(self) -> float:
        """Return the cost per unit for this resource."""
        pass

    @abstractmethod
    def get_unit_name(self) -> str:
        """Return the unit name (kWh/liters)."""
        pass

    def load_model(self) -> bool:
        """
        Load pre-trained model from disk.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                self._load_metadata()
                logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def save_model(self, path: Optional[str] = None) -> bool:
        """
        Save trained model to disk.

        Args:
            path: Optional custom path. Uses default if not provided.

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            save_path = path or self.model_path
            if self.model and save_path:
                # Ensure directory exists
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, save_path)
                self._save_metadata()
                logger.info(f"Model saved successfully to {save_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def _load_metadata(self) -> None:
        """Load model metadata from JSON file."""
        try:
            metadata_path = Config.MODEL_METADATA_PATH
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    all_metadata = json.load(f)
                    self.model_metadata = all_metadata.get(self.resource_type, {})
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")

    def _save_metadata(self) -> None:
        """Save model metadata to JSON file."""
        try:
            metadata_path = Config.MODEL_METADATA_PATH
            all_metadata = {}
            
            # Load existing metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    all_metadata = json.load(f)
            
            # Update with current model's metadata
            all_metadata[self.resource_type] = {
                "last_trained": datetime.now().isoformat(),
                "model_type": "Prophet",
                "resource_type": self.resource_type,
                "confidence_interval": self.confidence_interval,
                **self.model_metadata
            }
            
            # Save updated metadata
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(all_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model.
        
        Prophet requires columns named 'ds' (datestamp) and 'y' (value).

        Args:
            data: Input DataFrame with datetime and consumption columns

        Returns:
            pd.DataFrame: Prepared DataFrame with 'ds' and 'y' columns
        """
        df = data.copy()
        
        # Handle different column naming conventions
        if "ds" not in df.columns:
            # Find datetime column
            datetime_cols = ["datetime", "date", "timestamp", "time", "ds"]
            for col in datetime_cols:
                if col in df.columns:
                    df["ds"] = pd.to_datetime(df[col])
                    break
            else:
                # If no datetime column, create one from index
                if isinstance(df.index, pd.DatetimeIndex):
                    df["ds"] = df.index
                else:
                    raise ValueError("No datetime column found in data")

        if "y" not in df.columns:
            # Find value column
            value_cols = ["consumption", "value", "usage", "amount", "y"]
            for col in value_cols:
                if col in df.columns:
                    df["y"] = df[col]
                    break
            else:
                # Use last numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df["y"] = df[numeric_cols[-1]]
                else:
                    raise ValueError("No value column found in data")

        # Ensure proper types
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        
        # Remove any NaN values
        df = df.dropna(subset=["ds", "y"])
        
        # Sort by date
        df = df.sort_values("ds").reset_index(drop=True)

        return df[["ds", "y"]]

    def train(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        Train Prophet model on historical data.

        Args:
            data: DataFrame with historical consumption data
            **kwargs: Additional Prophet parameters

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Prepare data
            df = self.prepare_data(data)
            
            if len(df) < 2:
                logger.error("Insufficient data for training (need at least 2 data points)")
                return False

            # Initialize Prophet model with custom parameters
            self.model = Prophet(
                interval_width=self.confidence_interval,
                daily_seasonality=kwargs.get("daily_seasonality", True),
                weekly_seasonality=kwargs.get("weekly_seasonality", True),
                yearly_seasonality=kwargs.get("yearly_seasonality", True),
                changepoint_prior_scale=kwargs.get("changepoint_prior_scale", 0.05),
            )

            # Add custom seasonalities if provided
            if "custom_seasonalities" in kwargs:
                for seasonality in kwargs["custom_seasonalities"]:
                    self.model.add_seasonality(**seasonality)

            # Fit model
            self.model.fit(df)
            self.is_loaded = True
            
            # Update metadata
            self.model_metadata["training_samples"] = len(df)
            self.model_metadata["training_date_range"] = {
                "start": df["ds"].min().isoformat(),
                "end": df["ds"].max().isoformat()
            }

            logger.info(f"Model trained successfully on {len(df)} samples")
            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict(
        self,
        periods: int = None,
        frequency: str = "H",
        include_history: bool = False
    ) -> Dict[str, Any]:
        """
        Generate predictions for future periods.

        Args:
            periods: Number of periods to predict
            frequency: Frequency of predictions ('H'=hourly, 'D'=daily, etc.)
            include_history: Whether to include historical fitted values

        Returns:
            Dict containing predictions, confidence intervals, and metadata
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        try:
            # Use default periods if not specified
            periods = periods or self.default_periods
            periods = min(periods, self.max_periods)

            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=frequency,
                include_history=include_history
            )

            # Generate predictions
            forecast = self.model.predict(future)

            # Extract relevant columns
            result = self._format_predictions(forecast, include_history)
            
            # Add cost estimates
            result = self._add_cost_estimates(result)

            return result

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def predict_from_user_data(
        self,
        user_data: List[Dict[str, Any]],
        periods: int = None,
        frequency: str = "H"
    ) -> Dict[str, Any]:
        """
        Generate predictions based on user-provided historical data.
        
        This method trains a temporary model on user data and generates predictions.
        Used when users input their own consumption history.

        Args:
            user_data: List of dicts with 'datetime' and 'consumption' keys
            periods: Number of periods to predict
            frequency: Frequency of predictions

        Returns:
            Dict containing predictions and metadata
        """
        try:
            # Convert user data to DataFrame
            df = pd.DataFrame(user_data)
            
            # Train temporary model on user data
            if not self.train(df):
                raise ValueError("Failed to train model on user data")

            # Generate predictions
            return self.predict(periods=periods, frequency=frequency)

        except Exception as e:
            logger.error(f"Error predicting from user data: {e}")
            raise

    def _format_predictions(
        self,
        forecast: pd.DataFrame,
        include_history: bool
    ) -> Dict[str, Any]:
        """
        Format Prophet forecast output into standardized response.

        Args:
            forecast: Prophet forecast DataFrame
            include_history: Whether historical data is included

        Returns:
            Formatted prediction dictionary
        """
        # Filter to future predictions only if history not included
        if not include_history and hasattr(self.model, "history"):
            last_historical = self.model.history["ds"].max()
            forecast = forecast[forecast["ds"] > last_historical]

        predictions = []
        for _, row in forecast.iterrows():
            predictions.append({
                "datetime": row["ds"].isoformat(),
                "predicted_value": round(float(row["yhat"]), 2),
                "lower_bound": round(float(row["yhat_lower"]), 2),
                "upper_bound": round(float(row["yhat_upper"]), 2),
            })

        return {
            "success": True,
            "resource_type": self.get_resource_type(),
            "unit": self.get_unit_name(),
            "predictions": predictions,
            "summary": {
                "total_predicted": round(sum(p["predicted_value"] for p in predictions), 2),
                "average_predicted": round(
                    sum(p["predicted_value"] for p in predictions) / len(predictions), 2
                ) if predictions else 0,
                "min_predicted": round(min(p["predicted_value"] for p in predictions), 2) if predictions else 0,
                "max_predicted": round(max(p["predicted_value"] for p in predictions), 2) if predictions else 0,
                "periods": len(predictions),
            },
            "generated_at": datetime.now().isoformat(),
        }

    def _add_cost_estimates(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add cost estimates to prediction results.

        Args:
            result: Prediction result dictionary

        Returns:
            Updated result with cost estimates
        """
        cost_per_unit = self.get_cost_per_unit()
        
        # Add cost to each prediction
        for prediction in result.get("predictions", []):
            prediction["estimated_cost"] = round(
                prediction["predicted_value"] * cost_per_unit, 4
            )

        # Add total cost to summary
        if "summary" in result:
            result["summary"]["total_estimated_cost"] = round(
                result["summary"]["total_predicted"] * cost_per_unit, 2
            )
            result["summary"]["cost_per_unit"] = cost_per_unit
            result["summary"]["currency"] = "USD"

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "resource_type": self.get_resource_type(),
            "unit": self.get_unit_name(),
            "cost_per_unit": self.get_cost_per_unit(),
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "default_periods": self.default_periods,
            "max_periods": self.max_periods,
            "confidence_interval": self.confidence_interval,
            "metadata": self.model_metadata,
        }