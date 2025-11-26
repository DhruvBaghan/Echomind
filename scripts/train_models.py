#!/usr/bin/env python3
# ============================================
# EchoMind - Model Training Script
# ============================================

"""
Script to train ML models for EchoMind.

This script:
    1. Loads training data
    2. Preprocesses data for Prophet
    3. Trains electricity and water models
    4. Evaluates model performance
    5. Saves trained models

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --electricity-only
    python scripts/train_models.py --water-only
    python scripts/train_models.py --evaluate
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings

# Suppress Prophet logging
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config
from backend.ml_training import (
    train_electricity_model,
    train_water_model,
    get_training_status
)
from backend.ml_training.data_preprocessing import validate_data_quality


def check_data_availability() -> dict:
    """
    Check if training data is available.
    
    Returns:
        Dictionary with data availability status
    """
    status = {
        'electricity': {
            'raw': (Config.RAW_DATA_DIR / "electricity_training_data.csv").exists(),
            'processed': (Config.PROCESSED_DATA_DIR / "electricity_processed.csv").exists()
        },
        'water': {
            'raw': (Config.RAW_DATA_DIR / "water_training_data.csv").exists(),
            'processed': (Config.PROCESSED_DATA_DIR / "water_processed.csv").exists()
        }
    }
    return status


def generate_missing_data():
    """Generate missing training data."""
    from scripts.download_datasets import generate_electricity_data, generate_water_data, process_for_prophet
    
    data_status = check_data_availability()
    
    if not data_status['electricity']['raw']:
        print("Generating electricity training data...")
        generate_electricity_data()
    
    if not data_status['water']['raw']:
        print("Generating water training data...")
        generate_water_data()
    
    # Process for Prophet
    if not data_status['electricity']['processed']:
        elec_raw = Config.RAW_DATA_DIR / "electricity_training_data.csv"
        elec_proc = Config.PROCESSED_DATA_DIR / "electricity_processed.csv"
        if elec_raw.exists():
            process_for_prophet(str(elec_raw), str(elec_proc))
    
    if not data_status['water']['processed']:
        water_raw = Config.RAW_DATA_DIR / "water_training_data.csv"
        water_proc = Config.PROCESSED_DATA_DIR / "water_processed.csv"
        if water_raw.exists():
            process_for_prophet(str(water_raw), str(water_proc))


def train_all_models(
    train_electricity: bool = True,
    train_water: bool = True,
    validate: bool = True
) -> dict:
    """
    Train all models.
    
    Args:
        train_electricity: Whether to train electricity model
        train_water: Whether to train water model
        validate: Whether to validate data before training
        
    Returns:
        Training results dictionary
    """
    results = {
        'started_at': datetime.now().isoformat(),
        'models': {}
    }
    
    # Validate data if requested
    if validate:
        print("\n" + "=" * 50)
        print("Validating Training Data")
        print("=" * 50)
        
        import pandas as pd
        
        if train_electricity:
            elec_path = Config.PROCESSED_DATA_DIR / "electricity_processed.csv"
            if elec_path.exists():
                df = pd.read_csv(elec_path)
                quality = validate_data_quality(df)
                print(f"\nElectricity data quality score: {quality['quality_score']}/100")
                if quality['issues']:
                    print("Issues:")
                    for issue in quality['issues']:
                        print(f"  - {issue}")
        
        if train_water:
            water_path = Config.PROCESSED_DATA_DIR / "water_processed.csv"
            if water_path.exists():
                df = pd.read_csv(water_path)
                quality = validate_data_quality(df)
                print(f"\nWater data quality score: {quality['quality_score']}/100")
                if quality['issues']:
                    print("Issues:")
                    for issue in quality['issues']:
                        print(f"  - {issue}")
    
    # Train electricity model
    if train_electricity:
        print("\n" + "=" * 50)
        print("Training Electricity Model")
        print("=" * 50)
        
        try:
            elec_result = train_electricity_model()
            results['models']['electricity'] = elec_result
            
            if elec_result.get('success'):
                print(f"\n✓ Electricity model trained successfully!")
                print(f"  Model path: {elec_result.get('model_path')}")
                
                metrics = elec_result.get('metrics', {})
                print(f"\n  Performance Metrics:")
                print(f"    - MAE:  {metrics.get('mae', 'N/A')}")
                print(f"    - RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"    - R²:   {metrics.get('r2', 'N/A')}")
                print(f"    - MAPE: {metrics.get('mape', 'N/A')}%")
                print(f"    - Accuracy: {metrics.get('accuracy_percentage', 'N/A')}%")
            else:
                print(f"\n✗ Electricity model training failed!")
                print(f"  Error: {elec_result.get('error')}")
                
        except Exception as e:
            print(f"\n✗ Error training electricity model: {e}")
            results['models']['electricity'] = {
                'success': False,
                'error': str(e)
            }
    
    # Train water model
    if train_water:
        print("\n" + "=" * 50)
        print("Training Water Model")
        print("=" * 50)
        
        try:
            water_result = train_water_model()
            results['models']['water'] = water_result
            
            if water_result.get('success'):
                print(f"\n✓ Water model trained successfully!")
                print(f"  Model path: {water_result.get('model_path')}")
                
                metrics = water_result.get('metrics', {})
                print(f"\n  Performance Metrics:")
                print(f"    - MAE:  {metrics.get('mae', 'N/A')}")
                print(f"    - RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"    - R²:   {metrics.get('r2', 'N/A')}")
                print(f"    - MAPE: {metrics.get('mape', 'N/A')}%")
                print(f"    - Accuracy: {metrics.get('accuracy_percentage', 'N/A')}%")
                
                # Leak analysis
                leak = water_result.get('leak_analysis', {})
                if leak.get('analyzed'):
                    print(f"\n  Leak Analysis:")
                    print(f"    - Night average: {leak.get('night_average_lph', 'N/A')} L/h")
                    print(f"    - Baseline flow detected: {leak.get('has_baseline_flow', 'N/A')}")
            else:
                print(f"\n✗ Water model training failed!")
                print(f"  Error: {water_result.get('error')}")
                
        except Exception as e:
            print(f"\n✗ Error training water model: {e}")
            results['models']['water'] = {
                'success': False,
                'error': str(e)
            }
    
    results['completed_at'] = datetime.now().isoformat()
    
    return results


def show_model_status():
    """Display current model status."""
    print("\n" + "=" * 50)
    print("Current Model Status")
    print("=" * 50)
    
    status = get_training_status()
    
    for model, info in status.items():
        print(f"\n{model.capitalize()} Model:")
        if info['model_exists']:
            print(f"  ✓ Model exists: {info['model_path']}")
            if info['last_modified']:
                modified = datetime.fromtimestamp(info['last_modified'])
                print(f"  ✓ Last modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  ✗ Model not found")
    
    # Check metadata
    metadata_path = Config.ML_MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\nModel Metadata:")
        for model, info in metadata.items():
            if isinstance(info, dict) and 'metrics' in info:
                metrics = info['metrics']
                print(f"\n  {model.capitalize()}:")
                print(f"    - Accuracy: {metrics.get('accuracy_percentage', 'N/A')}%")
                print(f"    - Training samples: {metrics.get('training_samples', 'N/A')}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train ML models for EchoMind'
    )
    parser.add_argument(
        '--electricity-only', '-e',
        action='store_true',
        help='Train only electricity model'
    )
    parser.add_argument(
        '--water-only', '-w',
        action='store_true',
        help='Train only water model'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation'
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show current model status'
    )
    parser.add_argument(
        '--generate-data', '-g',
        action='store_true',
        help='Generate missing training data'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EchoMind - Model Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show status only
    if args.status:
        show_model_status()
        return
    
    # Check/generate data
    data_status = check_data_availability()
    
    print("\nData Availability:")
    for resource, status in data_status.items():
        raw = "✓" if status['raw'] else "✗"
        proc = "✓" if status['processed'] else "✗"
        print(f"  {resource}: raw {raw}, processed {proc}")
    
    # Generate missing data if requested or needed
    missing_data = not all(
        status['raw'] and status['processed']
        for status in data_status.values()
    )
    
    if missing_data:
        if args.generate_data:
            print("\nGenerating missing data...")
            generate_missing_data()
        else:
            print("\n⚠ Missing training data. Use --generate-data to create it.")
            return
    
    # Determine what to train
    train_electricity = not args.water_only
    train_water = not args.electricity_only
    validate = not args.no_validate
    
    # Train models
    results = train_all_models(
        train_electricity=train_electricity,
        train_water=train_water,
        validate=validate
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    success_count = sum(
        1 for r in results['models'].values()
        if r.get('success')
    )
    total_count = len(results['models'])
    
    print(f"\nModels trained: {success_count}/{total_count}")
    
    for model, result in results['models'].items():
        status = "✓" if result.get('success') else "✗"
        print(f"  {status} {model.capitalize()}")
    
    # Save training log
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTraining log saved to: {log_file}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()