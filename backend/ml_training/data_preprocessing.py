# ============================================
# EchoMind - Data Preprocessing Utilities
# ============================================

"""
Data preprocessing utilities for ML model training.

Provides functions for:
    - Loading and cleaning raw data
    - Feature engineering
    - Data transformation for Prophet
    - Handling missing values and outliers
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

from backend.config import Config
from backend.utils.logger import logger


class DataPreprocessor:
    """
    General data preprocessor for consumption data.
    """
    
    def __init__(self, resource_type: str = "electricity"):
        """
        Initialize preprocessor.
        
        Args:
            resource_type: Type of resource ('electricity' or 'water')
        """
        self.resource_type = resource_type
        self.stats = {}
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        return df
    
    def identify_datetime_column(self, df: pd.DataFrame) -> str:
        """
        Identify the datetime column in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the datetime column
        """
        datetime_candidates = [
            "datetime", "date", "timestamp", "time", "ds",
            "DateTime", "Date", "Timestamp", "Time"
        ]
        
        for col in datetime_candidates:
            if col in df.columns:
                return col
        
        # Try to find a column that can be parsed as datetime
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10))
                return col
            except Exception:
                continue
        
        raise ValueError("Could not identify datetime column")
    
    def identify_value_column(self, df: pd.DataFrame) -> str:
        """
        Identify the consumption value column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the value column
        """
        value_candidates = [
            "consumption", "value", "usage", "amount", "y",
            "Consumption", "Value", "Usage", "Amount",
            "kwh", "kWh", "liters", "Liters", "litres"
        ]
        
        for col in value_candidates:
            if col in df.columns:
                return col
        
        # Find first numeric column that's not the index
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        raise ValueError("Could not identify value column")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing datetime
        datetime_col = self.identify_datetime_column(df)
        df = df.dropna(subset=[datetime_col])
        
        # Handle missing values in consumption
        value_col = self.identify_value_column(df)
        
        # Fill missing values with interpolation
        if df[value_col].isna().sum() > 0:
            df[value_col] = df[value_col].interpolate(method='linear')
            df[value_col] = df[value_col].fillna(df[value_col].mean())
        
        final_len = len(df)
        logger.info(f"Cleaned data: {initial_len} -> {final_len} rows")
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: Method to use ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        initial_len = len(df)
        
        if method == "iqr":
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df = df[(df[column] >= lower) & (df[column] <= upper)]
            
        elif method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            df = df[np.abs((df[column] - mean) / std) <= threshold]
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outliers using {method} method")
        
        return df
    
    def prepare_for_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for Prophet model.
        
        Prophet requires columns named 'ds' (datetime) and 'y' (value).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        result = pd.DataFrame()
        
        # Get datetime column
        datetime_col = self.identify_datetime_column(df)
        result["ds"] = pd.to_datetime(df[datetime_col])
        
        # Get value column
        value_col = self.identify_value_column(df)
        result["y"] = pd.to_numeric(df[value_col], errors="coerce")
        
        # Remove any remaining NaN
        result = result.dropna()
        
        # Sort by datetime
        result = result.sort_values("ds").reset_index(drop=True)
        
        # Store statistics
        self.stats = {
            "rows": len(result),
            "date_range": {
                "start": result["ds"].min().isoformat(),
                "end": result["ds"].max().isoformat(),
            },
            "value_stats": {
                "mean": round(result["y"].mean(), 4),
                "std": round(result["y"].std(), 4),
                "min": round(result["y"].min(), 4),
                "max": round(result["y"].max(), 4),
            }
        }
        
        logger.info(f"Prepared {len(result)} rows for Prophet")
        return result
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the DataFrame.
        
        Args:
            df: DataFrame with 'ds' column
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Extract time features
        df["hour"] = df["ds"].dt.hour
        df["day_of_week"] = df["ds"].dt.dayofweek
        df["day_of_month"] = df["ds"].dt.day
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter
        df["year"] = df["ds"].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        return df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        frequency: str = "H",
        aggregation: str = "mean"
    ) -> pd.DataFrame:
        """
        Resample data to a different frequency.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            frequency: Target frequency ('H', 'D', 'W', 'M')
            aggregation: Aggregation method ('mean', 'sum', 'max', 'min')
            
        Returns:
            Resampled DataFrame
        """
        df = df.set_index("ds")
        
        if aggregation == "mean":
            resampled = df.resample(frequency).mean()
        elif aggregation == "sum":
            resampled = df.resample(frequency).sum()
        elif aggregation == "max":
            resampled = df.resample(frequency).max()
        elif aggregation == "min":
            resampled = df.resample(frequency).min()
        else:
            resampled = df.resample(frequency).mean()
        
        resampled = resampled.reset_index()
        resampled = resampled.dropna()
        
        logger.info(f"Resampled to {frequency}: {len(resampled)} rows")
        return resampled
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data into train, test, and optional validation sets.
        
        Args:
            df: Input DataFrame
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            
        Returns:
            Tuple of (train, test, validation) DataFrames
        """
        n = len(df)
        
        if validation_size > 0:
            val_idx = int(n * (1 - test_size - validation_size))
            test_idx = int(n * (1 - test_size))
            
            train = df.iloc[:val_idx]
            val = df.iloc[val_idx:test_idx]
            test = df.iloc[test_idx:]
            
            logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
            return train, test, val
        else:
            test_idx = int(n * (1 - test_size))
            train = df.iloc[:test_idx]
            test = df.iloc[test_idx:]
            
            logger.info(f"Split: train={len(train)}, test={len(test)}")
            return train, test, None
    
    def save_processed(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        Save processed data to CSV.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self.stats


def preprocess_electricity_data(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Preprocess electricity consumption data.
    
    Args:
        df: Raw electricity data
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outliers
        
    Returns:
        Preprocessed DataFrame ready for Prophet
    """
    preprocessor = DataPreprocessor(resource_type="electricity")
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Prepare for Prophet
    df = preprocessor.prepare_for_prophet(df)
    
    # Remove outliers if requested
    if remove_outliers:
        df = preprocessor.remove_outliers(
            df, "y", method="zscore", threshold=outlier_threshold
        )
    
    # Ensure non-negative values
    df["y"] = df["y"].clip(lower=0)
    
    # Sort by date
    df = df.sort_values("ds").reset_index(drop=True)
    
    return df


def preprocess_water_data(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Preprocess water consumption data.
    
    Args:
        df: Raw water data
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outliers
        
    Returns:
        Preprocessed DataFrame ready for Prophet
    """
    preprocessor = DataPreprocessor(resource_type="water")
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Prepare for Prophet
    df = preprocessor.prepare_for_prophet(df)
    
    # Remove outliers if requested
    if remove_outliers:
        df = preprocessor.remove_outliers(
            df, "y", method="zscore", threshold=outlier_threshold
        )
    
    # Ensure non-negative values
    df["y"] = df["y"].clip(lower=0)
    
    # Sort by date
    df = df.sort_values("ds").reset_index(drop=True)
    
    return df


def load_and_preprocess(
    file_path: str,
    resource_type: str = "electricity",
    save_processed: bool = False
) -> pd.DataFrame:
    """
    Load and preprocess data from file.
    
    Args:
        file_path: Path to raw data file
        resource_type: Type of resource
        save_processed: Whether to save processed data
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor(resource_type=resource_type)
    
    # Load data
    df = preprocessor.load_csv(file_path)
    
    # Preprocess based on resource type
    if resource_type == "electricity":
        processed = preprocess_electricity_data(df)
    else:
        processed = preprocess_water_data(df)
    
    # Save if requested
    if save_processed:
        output_dir = Config.PROCESSED_DATA_DIR
        output_path = output_dir / f"{resource_type}_processed.csv"
        preprocessor.save_processed(processed, str(output_path))
    
    return processed


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return report.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Data quality report
    """
    report = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "issues": [],
        "warnings": [],
    }
    
    # Check for missing values
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            if pct > 10:
                report["issues"].append(f"Column '{col}' has {pct:.1f}% missing values")
            elif pct > 0:
                report["warnings"].append(f"Column '{col}' has {pct:.1f}% missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        pct = (duplicates / len(df)) * 100
        report["warnings"].append(f"Found {duplicates} duplicate rows ({pct:.1f}%)")
    
    # Check datetime column
    try:
        preprocessor = DataPreprocessor()
        dt_col = preprocessor.identify_datetime_column(df)
        report["datetime_column"] = dt_col
        
        # Check for gaps
        df_sorted = df.sort_values(dt_col)
        df_sorted[dt_col] = pd.to_datetime(df_sorted[dt_col])
        diffs = df_sorted[dt_col].diff()
        
        if diffs.max() > pd.Timedelta(days=1):
            report["warnings"].append("Data has gaps larger than 1 day")
            
    except Exception as e:
        report["issues"].append(f"Could not identify datetime column: {e}")
    
    # Check value column
    try:
        preprocessor = DataPreprocessor()
        val_col = preprocessor.identify_value_column(df)
        report["value_column"] = val_col
        
        # Check for negative values
        if (df[val_col] < 0).any():
            neg_count = (df[val_col] < 0).sum()
            report["warnings"].append(f"Found {neg_count} negative values in '{val_col}'")
            
        # Check for zero values
        zero_count = (df[val_col] == 0).sum()
        if zero_count > len(df) * 0.1:
            report["warnings"].append(f"{zero_count} zero values in '{val_col}'")
            
    except Exception as e:
        report["issues"].append(f"Could not identify value column: {e}")
    
    # Overall quality score
    issues_weight = len(report["issues"]) * 10
    warnings_weight = len(report["warnings"]) * 3
    report["quality_score"] = max(0, 100 - issues_weight - warnings_weight)
    
    return report


def merge_datasets(
    *datasets: pd.DataFrame,
    on: str = "ds"
) -> pd.DataFrame:
    """
    Merge multiple datasets on a common column.
    
    Args:
        *datasets: DataFrames to merge
        on: Column to merge on
        
    Returns:
        Merged DataFrame
    """
    if len(datasets) == 0:
        return pd.DataFrame()
    
    if len(datasets) == 1:
        return datasets[0]
    
    result = datasets[0]
    for i, df in enumerate(datasets[1:], start=1):
        result = pd.merge(
            result,
            df,
            on=on,
            how="outer",
            suffixes=("", f"_{i}")
        )
    
    result = result.sort_values(on).reset_index(drop=True)
    logger.info(f"Merged {len(datasets)} datasets: {len(result)} rows")
    
    return result