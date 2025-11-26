#!/usr/bin/env python3
"""
Generate both pre-trained models for EchoMind.
Run: python scripts/generate_models.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime

def main():
    print("=" * 60)
    print("EchoMind - Model Generator")
    print("=" * 60)
    
    # Import training functions
    from backend.ml_training.train_electricity_model import train_electricity_model, generate_sample_data as gen_elec
    from backend.ml_training.train_water_model import train_water_model, generate_sample_data as gen_water
    from backend.config import Config
    
    # Ensure directories exist
    Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Generate electricity data and train model
    print("\n[1/2] Electricity Model")
    print("-" * 40)
    try:
        elec_data_path = Config.RAW_DATA_DIR / "electricity_training_data.csv"
        if not elec_data_path.exists():
            print("Generating electricity training data...")
            gen_elec(str(elec_data_path))
        
        result = train_electricity_model()
        results['electricity'] = result
        print(f"✓ Electricity model: {'SUCCESS' if result.get('success') else 'FAILED'}")
    except Exception as e:
        print(f"✗ Electricity model failed: {e}")
        results['electricity'] = {'success': False, 'error': str(e)}
    
    # Generate water data and train model
    print("\n[2/2] Water Model")
    print("-" * 40)
    try:
        water_data_path = Config.RAW_DATA_DIR / "water_training_data.csv"
        if not water_data_path.exists():
            print("Generating water training data...")
            gen_water(str(water_data_path))
        
        result = train_water_model()
        results['water'] = result
        print(f"✓ Water model: {'SUCCESS' if result.get('success') else 'FAILED'}")
    except Exception as e:
        print(f"✗ Water model failed: {e}")
        results['water'] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for model, result in results.items():
        status = "✓" if result.get('success') else "✗"
        print(f"{status} {model.capitalize()}: {result.get('model_path', result.get('error', 'Unknown'))}")
    
    # Save metadata
    metadata_path = Config.ML_MODELS_DIR / "model_metadata.json"
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'models': results
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()