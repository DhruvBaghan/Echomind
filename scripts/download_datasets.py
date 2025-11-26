#!/usr/bin/env python3
# ============================================
# EchoMind - Dataset Downloader
# ============================================

"""
Script to download or generate datasets for EchoMind.

This script can:
    1. Generate synthetic training data
    2. Download sample datasets from remote sources (if available)
    3. Validate existing datasets

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --generate
    python scripts/download_datasets.py --validate
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from backend.config import Config


def generate_electricity_data(days: int = 365, output_path: str = None) -> str:
    """
    Generate synthetic electricity consumption data.
    
    Args:
        days: Number of days of data to generate
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    print(f"Generating {days} days of electricity data...")
    
    if output_path is None:
        output_path = str(Config.RAW_DATA_DIR / "electricity_training_data.csv")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.now() - timedelta(days=days)
    data = []
    
    for day in range(days):
        for hour in range(24):
            timestamp = start_date + timedelta(days=day, hours=hour)
            h = timestamp.hour
            dow = timestamp.weekday()
            month = timestamp.month
            
            # Base consumption (kWh)
            base = 1.5
            
            # Time of day pattern
            if 6 <= h <= 9:
                time_factor = 1.5 + np.random.uniform(-0.1, 0.1)
            elif 17 <= h <= 22:
                time_factor = 1.8 + np.random.uniform(-0.1, 0.1)
            elif 0 <= h <= 5:
                time_factor = 0.4 + np.random.uniform(-0.05, 0.05)
            else:
                time_factor = 1.0 + np.random.uniform(-0.1, 0.1)
            
            # Weekend factor
            weekend_factor = 1.15 if dow >= 5 else 1.0
            
            # Seasonal factor (HVAC)
            if month in [6, 7, 8]:
                seasonal_factor = 1.35
            elif month in [12, 1, 2]:
                seasonal_factor = 1.25
            else:
                seasonal_factor = 1.0
            
            # Calculate with noise
            consumption = base * time_factor * weekend_factor * seasonal_factor
            consumption += np.random.normal(0, 0.2)
            consumption = max(0.2, consumption)
            
            data.append({
                'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'consumption': round(consumption, 2),
                'notes': ''
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} rows to {output_path}")
    print(f"  - Min: {df['consumption'].min():.2f} kWh")
    print(f"  - Max: {df['consumption'].max():.2f} kWh")
    print(f"  - Mean: {df['consumption'].mean():.2f} kWh")
    
    return output_path


def generate_water_data(days: int = 365, output_path: str = None) -> str:
    """
    Generate synthetic water consumption data.
    
    Args:
        days: Number of days of data to generate
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    print(f"Generating {days} days of water data...")
    
    if output_path is None:
        output_path = str(Config.RAW_DATA_DIR / "water_training_data.csv")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.now() - timedelta(days=days)
    data = []
    
    for day in range(days):
        for hour in range(24):
            timestamp = start_date + timedelta(days=day, hours=hour)
            h = timestamp.hour
            dow = timestamp.weekday()
            month = timestamp.month
            
            # Base consumption (liters per hour)
            base = 15
            
            # Time of day pattern
            if 6 <= h <= 9:
                time_factor = 3.0 + np.random.uniform(-0.3, 0.3)
            elif 18 <= h <= 22:
                time_factor = 2.5 + np.random.uniform(-0.3, 0.3)
            elif 0 <= h <= 5:
                time_factor = 0.2 + np.random.uniform(-0.05, 0.05)
            elif 12 <= h <= 14:
                time_factor = 1.5 + np.random.uniform(-0.2, 0.2)
            else:
                time_factor = 1.0 + np.random.uniform(-0.2, 0.2)
            
            # Weekend factor (laundry on Saturday)
            if dow == 5:
                weekend_factor = 1.4
            elif dow == 6:
                weekend_factor = 1.2
            else:
                weekend_factor = 1.0
            
            # Seasonal factor (summer gardening)
            if month in [6, 7, 8]:
                seasonal_factor = 1.25
            else:
                seasonal_factor = 1.0
            
            # Calculate with noise
            consumption = base * time_factor * weekend_factor * seasonal_factor
            consumption += np.random.normal(0, 3)
            consumption = max(1, consumption)
            
            data.append({
                'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'consumption': round(consumption, 1),
                'notes': ''
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} rows to {output_path}")
    print(f"  - Min: {df['consumption'].min():.1f} L")
    print(f"  - Max: {df['consumption'].max():.1f} L")
    print(f"  - Mean: {df['consumption'].mean():.1f} L")
    
    return output_path


def process_for_prophet(input_path: str, output_path: str) -> str:
    """
    Process raw data into Prophet format.
    
    Args:
        input_path: Input CSV path
        output_path: Output CSV path
        
    Returns:
        Path to processed file
    """
    print(f"Processing {input_path}...")
    
    df = pd.read_csv(input_path)
    
    processed = pd.DataFrame()
    processed['ds'] = pd.to_datetime(df['datetime'])
    processed['y'] = pd.to_numeric(df['consumption'])
    
    processed = processed.sort_values('ds').reset_index(drop=True)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    processed.to_csv(output_path, index=False)
    print(f"✓ Processed {len(processed)} rows to {output_path}")
    
    return output_path


def validate_dataset(file_path: str) -> dict:
    """
    Validate a dataset file.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Validation result dictionary
    """
    print(f"Validating {file_path}...")
    
    result = {
        'file': file_path,
        'valid': True,
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if file exists
    if not os.path.exists(file_path):
        result['valid'] = False
        result['issues'].append(f"File not found: {file_path}")
        return result
    
    # Try to load
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        result['valid'] = False
        result['issues'].append(f"Failed to read CSV: {e}")
        return result
    
    result['stats']['rows'] = len(df)
    result['stats']['columns'] = list(df.columns)
    
    # Check for required columns
    if 'ds' in df.columns and 'y' in df.columns:
        # Prophet format
        result['stats']['format'] = 'prophet'
        datetime_col = 'ds'
        value_col = 'y'
    elif 'datetime' in df.columns and 'consumption' in df.columns:
        # Raw format
        result['stats']['format'] = 'raw'
        datetime_col = 'datetime'
        value_col = 'consumption'
    else:
        result['issues'].append("Missing required columns (datetime/consumption or ds/y)")
        result['valid'] = False
        return result
    
    # Check for nulls
    null_count = df[value_col].isna().sum()
    if null_count > 0:
        pct = (null_count / len(df)) * 100
        if pct > 10:
            result['issues'].append(f"Too many null values: {null_count} ({pct:.1f}%)")
        else:
            result['warnings'].append(f"Some null values: {null_count} ({pct:.1f}%)")
    
    # Check for negative values
    neg_count = (df[value_col] < 0).sum()
    if neg_count > 0:
        result['warnings'].append(f"Negative values found: {neg_count}")
    
    # Stats
    result['stats']['min'] = float(df[value_col].min())
    result['stats']['max'] = float(df[value_col].max())
    result['stats']['mean'] = float(df[value_col].mean())
    result['stats']['std'] = float(df[value_col].std())
    
    # Date range
    try:
        dates = pd.to_datetime(df[datetime_col])
        result['stats']['date_range'] = {
            'start': dates.min().isoformat(),
            'end': dates.max().isoformat()
        }
    except Exception:
        result['warnings'].append("Could not parse date range")
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        result['warnings'].append(f"Duplicate rows: {dup_count}")
    
    if result['issues']:
        result['valid'] = False
    
    status = "✓ Valid" if result['valid'] else "✗ Invalid"
    print(f"  {status}")
    
    return result


def download_sample_datasets():
    """
    Download sample datasets from remote sources.
    
    Note: This is a placeholder. In a real scenario, you might download
    from public datasets like:
    - UCI Machine Learning Repository
    - Kaggle datasets
    - Government open data portals
    """
    print("Downloading sample datasets...")
    print("Note: No remote sources configured. Generating synthetic data instead.")
    
    # Generate synthetic data as fallback
    generate_electricity_data()
    generate_water_data()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Download or generate datasets for EchoMind'
    )
    parser.add_argument(
        '--generate', '-g',
        action='store_true',
        help='Generate synthetic datasets'
    )
    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Download sample datasets'
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate existing datasets'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of data to generate (default: 365)'
    )
    parser.add_argument(
        '--process', '-p',
        action='store_true',
        help='Process raw data for Prophet'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EchoMind - Dataset Manager")
    print("=" * 60)
    
    # Default action: generate if no datasets exist
    if not any([args.generate, args.download, args.validate, args.process]):
        elec_exists = (Config.RAW_DATA_DIR / "electricity_training_data.csv").exists()
        water_exists = (Config.RAW_DATA_DIR / "water_training_data.csv").exists()
        
        if not elec_exists or not water_exists:
            args.generate = True
            args.process = True
        else:
            args.validate = True
    
    results = {}
    
    # Generate datasets
    if args.generate:
        print("\n[1] Generating Datasets")
        print("-" * 40)
        
        elec_path = generate_electricity_data(days=args.days)
        water_path = generate_water_data(days=args.days)
        
        results['generated'] = {
            'electricity': elec_path,
            'water': water_path
        }
    
    # Download datasets
    if args.download:
        print("\n[2] Downloading Datasets")
        print("-" * 40)
        download_sample_datasets()
    
    # Process for Prophet
    if args.process:
        print("\n[3] Processing for Prophet")
        print("-" * 40)
        
        elec_raw = Config.RAW_DATA_DIR / "electricity_training_data.csv"
        elec_proc = Config.PROCESSED_DATA_DIR / "electricity_processed.csv"
        
        water_raw = Config.RAW_DATA_DIR / "water_training_data.csv"
        water_proc = Config.PROCESSED_DATA_DIR / "water_processed.csv"
        
        if elec_raw.exists():
            process_for_prophet(str(elec_raw), str(elec_proc))
        
        if water_raw.exists():
            process_for_prophet(str(water_raw), str(water_proc))
    
    # Validate datasets
    if args.validate:
        print("\n[4] Validating Datasets")
        print("-" * 40)
        
        datasets = [
            Config.RAW_DATA_DIR / "electricity_training_data.csv",
            Config.RAW_DATA_DIR / "water_training_data.csv",
            Config.PROCESSED_DATA_DIR / "electricity_processed.csv",
            Config.PROCESSED_DATA_DIR / "water_processed.csv",
        ]
        
        validation_results = []
        for dataset in datasets:
            if dataset.exists():
                result = validate_dataset(str(dataset))
                validation_results.append(result)
        
        results['validation'] = validation_results
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if 'generated' in results:
        print("\nGenerated files:")
        for resource, path in results['generated'].items():
            print(f"  - {resource}: {path}")
    
    if 'validation' in results:
        print("\nValidation results:")
        for result in results['validation']:
            status = "✓" if result['valid'] else "✗"
            print(f"  {status} {Path(result['file']).name}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"      Issue: {issue}")
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"      Warning: {warning}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()