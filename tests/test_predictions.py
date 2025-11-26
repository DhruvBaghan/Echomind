# ============================================
# EchoMind - Prediction Tests
# ============================================

"""
Tests for prediction models and services.

Tests cover:
    - BasePredictor functionality
    - ElectricityPredictor
    - WaterPredictor
    - UnifiedOptimizer
    - PredictionService
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import test configuration
from tests import get_test_config, PROJECT_ROOT

# Import modules to test
from backend.models.base_predictor import BasePredictor
from backend.models.electricity_predictor import ElectricityPredictor
from backend.models.water_predictor import WaterPredictor
from backend.models.unified_optimizer import UnifiedOptimizer
from backend.services.prediction_service import PredictionService


# ===========================================
# Fixtures
# ===========================================

@pytest.fixture
def sample_electricity_data():
    """Generate sample electricity consumption data."""
    data = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(168):  # 7 days * 24 hours
        timestamp = base_time + timedelta(hours=i)
        hour = timestamp.hour
        
        # Create realistic pattern
        if 6 <= hour <= 9:
            value = 2.0 + np.random.uniform(-0.3, 0.3)
        elif 17 <= hour <= 21:
            value = 2.5 + np.random.uniform(-0.3, 0.3)
        elif 0 <= hour <= 5:
            value = 0.5 + np.random.uniform(-0.1, 0.1)
        else:
            value = 1.2 + np.random.uniform(-0.2, 0.2)
        
        data.append({
            'datetime': timestamp.isoformat(),
            'consumption': round(value, 2)
        })
    
    return data


@pytest.fixture
def sample_water_data():
    """Generate sample water consumption data."""
    data = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(168):  # 7 days * 24 hours
        timestamp = base_time + timedelta(hours=i)
        hour = timestamp.hour
        
        # Create realistic pattern
        if 6 <= hour <= 9:
            value = 45 + np.random.uniform(-5, 5)
        elif 18 <= hour <= 22:
            value = 40 + np.random.uniform(-5, 5)
        elif 0 <= hour <= 5:
            value = 5 + np.random.uniform(-2, 2)
        else:
            value = 15 + np.random.uniform(-3, 3)
        
        data.append({
            'datetime': timestamp.isoformat(),
            'consumption': round(max(0, value), 1)
        })
    
    return data


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for training."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        periods=720,
        freq='H'
    )
    
    values = []
    for dt in dates:
        hour = dt.hour
        if 6 <= hour <= 9 or 17 <= hour <= 21:
            base = 2.0
        elif 0 <= hour <= 5:
            base = 0.5
        else:
            base = 1.2
        values.append(base + np.random.uniform(-0.2, 0.2))
    
    return pd.DataFrame({
        'ds': dates,
        'y': values
    })


@pytest.fixture
def mock_prophet_model():
    """Create a mock Prophet model."""
    mock = MagicMock()
    
    # Mock the predict method
    def mock_predict(future_df):
        n = len(future_df)
        return pd.DataFrame({
            'ds': future_df['ds'],
            'yhat': np.random.uniform(1, 3, n),
            'yhat_lower': np.random.uniform(0.5, 1.5, n),
            'yhat_upper': np.random.uniform(2.5, 4, n),
        })
    
    mock.predict = mock_predict
    mock.make_future_dataframe = lambda periods, freq, include_history: pd.DataFrame({
        'ds': pd.date_range(start=datetime.now(), periods=periods, freq=freq)
    })
    mock.history = pd.DataFrame({
        'ds': pd.date_range(start=datetime.now() - timedelta(days=7), periods=168, freq='H')
    })
    
    return mock


@pytest.fixture
def electricity_predictor():
    """Create an ElectricityPredictor instance."""
    with patch.object(ElectricityPredictor, 'load_model', return_value=False):
        predictor = ElectricityPredictor()
    return predictor


@pytest.fixture
def water_predictor():
    """Create a WaterPredictor instance."""
    with patch.object(WaterPredictor, 'load_model', return_value=False):
        predictor = WaterPredictor()
    return predictor


@pytest.fixture
def prediction_service():
    """Create a PredictionService instance."""
    return PredictionService()


# ===========================================
# BasePredictor Tests
# ===========================================

class TestBasePredictor:
    """Tests for BasePredictor abstract class."""
    
    def test_prepare_data_with_datetime_column(self, sample_dataframe):
        """Test data preparation with datetime column."""
        # Create a concrete implementation for testing
        with patch.object(ElectricityPredictor, 'load_model', return_value=False):
            predictor = ElectricityPredictor()
        
        # Rename columns to test detection
        df = sample_dataframe.rename(columns={'ds': 'datetime', 'y': 'consumption'})
        
        result = predictor.prepare_data(df)
        
        assert 'ds' in result.columns
        assert 'y' in result.columns
        assert len(result) > 0
    
    def test_prepare_data_removes_nan(self, electricity_predictor):
        """Test that prepare_data removes NaN values."""
        df = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'y': [1, 2, None, 4, 5, None, 7, 8, 9, 10]
        })
        
        result = electricity_predictor.prepare_data(df)
        
        assert not result['y'].isna().any()
        assert len(result) == 8  # 10 - 2 NaN values
    
    def test_prepare_data_sorts_by_date(self, electricity_predictor):
        """Test that prepare_data sorts by datetime."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='H')
        df = pd.DataFrame({
            'ds': dates[[3, 1, 4, 0, 2]],  # Shuffled
            'y': [4, 2, 5, 1, 3]
        })
        
        result = electricity_predictor.prepare_data(df)
        
        # Check if sorted
        assert result['ds'].is_monotonic_increasing


# ===========================================
# ElectricityPredictor Tests
# ===========================================

class TestElectricityPredictor:
    """Tests for ElectricityPredictor."""
    
    def test_get_resource_type(self, electricity_predictor):
        """Test resource type is correct."""
        assert electricity_predictor.get_resource_type() == 'electricity'
    
    def test_get_unit_name(self, electricity_predictor):
        """Test unit name is correct."""
        assert electricity_predictor.get_unit_name() == 'kWh'
    
    def test_get_cost_per_unit(self, electricity_predictor):
        """Test cost per unit returns a value."""
        cost = electricity_predictor.get_cost_per_unit()
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_train_with_sufficient_data(self, electricity_predictor, sample_dataframe):
        """Test training with sufficient data."""
        result = electricity_predictor.train(sample_dataframe)
        
        assert result == True
        assert electricity_predictor.is_loaded == True
        assert electricity_predictor.model is not None
    
    def test_train_with_insufficient_data(self, electricity_predictor):
        """Test training with insufficient data."""
        df = pd.DataFrame({
            'ds': [datetime.now()],
            'y': [1.5]
        })
        
        result = electricity_predictor.train(df)
        
        assert result == False
    
    def test_predict_with_mock_model(self, electricity_predictor, mock_prophet_model):
        """Test prediction with mocked model."""
        electricity_predictor.model = mock_prophet_model
        electricity_predictor.is_loaded = True
        
        result = electricity_predictor.predict(periods=24)
        
        assert result['success'] == True
        assert 'predictions' in result
        assert 'summary' in result
        assert result['resource_type'] == 'electricity'
        assert result['unit'] == 'kWh'
    
    def test_predict_without_loaded_model(self, electricity_predictor):
        """Test prediction fails without loaded model."""
        electricity_predictor.is_loaded = False
        
        with pytest.raises(ValueError, match="Model not loaded"):
            electricity_predictor.predict(periods=24)
    
    def test_predict_from_user_data(self, electricity_predictor, sample_electricity_data):
        """Test prediction from user-provided data."""
        result = electricity_predictor.predict_from_user_data(
            user_data=sample_electricity_data,
            periods=24
        )
        
        assert result['success'] == True
        assert len(result['predictions']) == 24
    
    def test_peak_hour_analysis(self, electricity_predictor, mock_prophet_model):
        """Test peak hour analysis is included."""
        electricity_predictor.model = mock_prophet_model
        electricity_predictor.is_loaded = True
        
        result = electricity_predictor.predict(periods=24)
        
        assert 'peak_analysis' in result
        assert 'peak_hours' in result['peak_analysis']
        assert 'off_peak_hours' in result['peak_analysis']
    
    def test_recommendations_generated(self, electricity_predictor, mock_prophet_model):
        """Test recommendations are generated."""
        electricity_predictor.model = mock_prophet_model
        electricity_predictor.is_loaded = True
        
        result = electricity_predictor.predict(periods=24)
        
        assert 'recommendations' in result
        assert isinstance(result['recommendations'], list)
    
    def test_cost_estimates_included(self, electricity_predictor, mock_prophet_model):
        """Test cost estimates are included."""
        electricity_predictor.model = mock_prophet_model
        electricity_predictor.is_loaded = True
        
        result = electricity_predictor.predict(periods=24)
        
        assert 'summary' in result
        assert 'total_estimated_cost' in result['summary']
        
        # Each prediction should have cost
        for pred in result['predictions']:
            assert 'estimated_cost' in pred


# ===========================================
# WaterPredictor Tests
# ===========================================

class TestWaterPredictor:
    """Tests for WaterPredictor."""
    
    def test_get_resource_type(self, water_predictor):
        """Test resource type is correct."""
        assert water_predictor.get_resource_type() == 'water'
    
    def test_get_unit_name(self, water_predictor):
        """Test unit name is correct."""
        assert water_predictor.get_unit_name() == 'liters'
    
    def test_get_cost_per_unit(self, water_predictor):
        """Test cost per unit returns a value."""
        cost = water_predictor.get_cost_per_unit()
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_predict_from_user_data(self, water_predictor, sample_water_data):
        """Test prediction from user-provided data."""
        result = water_predictor.predict_from_user_data(
            user_data=sample_water_data,
            periods=24
        )
        
        assert result['success'] == True
        assert len(result['predictions']) == 24
        assert result['unit'] == 'liters'
    
    def test_leak_detection_included(self, water_predictor, mock_prophet_model):
        """Test leak detection is included in results."""
        water_predictor.model = mock_prophet_model
        water_predictor.is_loaded = True
        
        result = water_predictor.predict(periods=24)
        
        assert 'leak_detection' in result
        assert 'leak_detected' in result['leak_detection']
        assert 'severity' in result['leak_detection']
    
    def test_usage_pattern_analysis(self, water_predictor, mock_prophet_model):
        """Test usage pattern analysis is included."""
        water_predictor.model = mock_prophet_model
        water_predictor.is_loaded = True
        
        result = water_predictor.predict(periods=24)
        
        assert 'usage_patterns' in result
        patterns = result['usage_patterns']
        assert 'morning_peak' in patterns
        assert 'evening_peak' in patterns


# ===========================================
# UnifiedOptimizer Tests
# ===========================================

class TestUnifiedOptimizer:
    """Tests for UnifiedOptimizer."""
    
    def test_initialization(self):
        """Test optimizer initializes correctly."""
        optimizer = UnifiedOptimizer()
        
        assert optimizer.electricity_predictor is not None
        assert optimizer.water_predictor is not None
    
    def test_load_models(self):
        """Test loading models."""
        optimizer = UnifiedOptimizer()
        
        result = optimizer.load_models()
        
        assert 'electricity' in result
        assert 'water' in result
        assert isinstance(result['electricity'], bool)
        assert isinstance(result['water'], bool)
    
    def test_predict_from_user_data(
        self,
        sample_electricity_data,
        sample_water_data
    ):
        """Test combined prediction from user data."""
        optimizer = UnifiedOptimizer()
        
        result = optimizer.predict_from_user_data(
            electricity_data=sample_electricity_data,
            water_data=sample_water_data,
            periods=24
        )
        
        assert result['success'] == True
        assert 'resources' in result
        assert 'electricity' in result['resources']
        assert 'water' in result['resources']
    
    def test_combined_analysis(self, sample_electricity_data, sample_water_data):
        """Test combined analysis is generated."""
        optimizer = UnifiedOptimizer()
        
        result = optimizer.predict_from_user_data(
            electricity_data=sample_electricity_data,
            water_data=sample_water_data,
            periods=24
        )
        
        assert 'combined_analysis' in result
        assert 'total_cost' in result['combined_analysis']
        assert 'cost_breakdown' in result['combined_analysis']
    
    def test_sustainability_score(self, sample_electricity_data, sample_water_data):
        """Test sustainability score is calculated."""
        optimizer = UnifiedOptimizer()
        
        result = optimizer.predict_from_user_data(
            electricity_data=sample_electricity_data,
            water_data=sample_water_data,
            periods=24
        )
        
        assert 'sustainability' in result
        assert 'overall_score' in result['sustainability']
        assert 'grade' in result['sustainability']
        assert 0 <= result['sustainability']['overall_score'] <= 100
    
    def test_unified_recommendations(self, sample_electricity_data, sample_water_data):
        """Test unified recommendations are generated."""
        optimizer = UnifiedOptimizer()
        
        result = optimizer.predict_from_user_data(
            electricity_data=sample_electricity_data,
            water_data=sample_water_data,
            periods=24
        )
        
        assert 'unified_recommendations' in result
        assert isinstance(result['unified_recommendations'], list)
    
    def test_get_optimizer_info(self):
        """Test getting optimizer information."""
        optimizer = UnifiedOptimizer()
        
        info = optimizer.get_optimizer_info()
        
        assert 'name' in info
        assert 'version' in info
        assert 'models' in info
        assert 'benchmarks' in info


# ===========================================
# PredictionService Tests
# ===========================================

class TestPredictionService:
    """Tests for PredictionService."""
    
    def test_initialization(self, prediction_service):
        """Test service initializes correctly."""
        assert prediction_service.unified_optimizer is not None
        assert prediction_service.electricity_predictor is not None
        assert prediction_service.water_predictor is not None
    
    def test_predict_electricity(self, prediction_service, sample_electricity_data):
        """Test electricity prediction through service."""
        result = prediction_service.predict_electricity(
            user_data=sample_electricity_data,
            periods=24
        )
        
        assert result['success'] == True
        assert 'predictions' in result
    
    def test_predict_water(self, prediction_service, sample_water_data):
        """Test water prediction through service."""
        result = prediction_service.predict_water(
            user_data=sample_water_data,
            periods=24
        )
        
        assert result['success'] == True
        assert 'predictions' in result
    
    def test_predict_both(
        self,
        prediction_service,
        sample_electricity_data,
        sample_water_data
    ):
        """Test combined prediction through service."""
        result = prediction_service.predict_both(
            electricity_data=sample_electricity_data,
            water_data=sample_water_data,
            electricity_periods=24,
            water_periods=24
        )
        
        assert result['success'] == True
        assert 'resources' in result
        assert 'combined_analysis' in result
    
    def test_get_demo_predictions(self, prediction_service):
        """Test getting demo predictions."""
        result = prediction_service.get_demo_predictions(
            resource='both',
            periods=24
        )
        
        assert result['success'] == True
        # Should have at least one resource
        assert 'electricity' in result or 'water' in result
    
    def test_calculate_sustainability_score(self, prediction_service):
        """Test sustainability score calculation."""
        result = prediction_service.calculate_sustainability_score(
            electricity_daily=25.0,
            water_daily=300.0,
            household_size=4
        )
        
        assert 'overall_score' in result
        assert 'grade' in result
        assert 'individual_scores' in result
    
    def test_quick_predict(self, prediction_service):
        """Test quick prediction."""
        result = prediction_service.quick_predict(
            resource_type='electricity',
            recent_average=1.5,
            periods=24
        )
        
        assert result['success'] == True
        assert len(result['predictions']) == 24
        assert result['method'] == 'quick_prediction'
    
    def test_get_model_info(self, prediction_service):
        """Test getting model information."""
        info = prediction_service.get_model_info()
        
        assert 'electricity' in info
        assert 'water' in info
        assert 'loaded' in info['electricity']


# ===========================================
# Integration Tests
# ===========================================

class TestPredictionIntegration:
    """Integration tests for prediction workflow."""
    
    def test_full_prediction_workflow(self, sample_electricity_data):
        """Test complete prediction workflow."""
        # 1. Create service
        service = PredictionService()
        
        # 2. Generate prediction
        result = service.predict_electricity(
            user_data=sample_electricity_data,
            periods=24
        )
        
        # 3. Verify result structure
        assert result['success'] == True
        assert 'predictions' in result
        assert 'summary' in result
        
        # 4. Verify predictions
        predictions = result['predictions']
        assert len(predictions) == 24
        
        for pred in predictions:
            assert 'datetime' in pred
            assert 'predicted_value' in pred
            assert 'lower_bound' in pred
            assert 'upper_bound' in pred
            assert pred['predicted_value'] >= 0
            assert pred['lower_bound'] <= pred['predicted_value']
            assert pred['upper_bound'] >= pred['predicted_value']
        
        # 5. Verify summary
        summary = result['summary']
        assert summary['periods'] == 24
        assert summary['total_predicted'] > 0
        assert 'total_estimated_cost' in summary
    
    def test_combined_prediction_workflow(
        self,
        sample_electricity_data,
        sample_water_data
    ):
        """Test complete combined prediction workflow."""
        # 1. Create optimizer
        optimizer = UnifiedOptimizer()
        
        # 2. Generate combined prediction
        result = optimizer.predict_from_user_data(
            electricity_data=sample_electricity_data,
            water_data=sample_water_data,
            periods=24
        )
        
        # 3. Verify structure
        assert result['success'] == True
        assert 'resources' in result
        assert 'combined_analysis' in result
        assert 'sustainability' in result
        assert 'unified_recommendations' in result
        
        # 4. Verify both resources
        assert result['resources']['electricity']['success'] == True
        assert result['resources']['water']['success'] == True
        
        # 5. Verify combined analysis
        analysis = result['combined_analysis']
        assert analysis['total_cost'] >= 0
        assert 'electricity' in analysis['cost_breakdown']
        assert 'water' in analysis['cost_breakdown']


# ===========================================
# Edge Case Tests
# ===========================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_data(self, prediction_service):
        """Test handling of empty data."""
        result = prediction_service.predict_electricity(
            user_data=[],
            periods=24
        )
        
        assert result['success'] == False
        assert 'error' in result
    
    def test_single_data_point(self, prediction_service):
        """Test handling of single data point."""
        result = prediction_service.predict_electricity(
            user_data=[{'datetime': '2024-01-01T00:00:00', 'consumption': 1.5}],
            periods=24
        )
        
        assert result['success'] == False
    
    def test_invalid_data_format(self, prediction_service):
        """Test handling of invalid data format."""
        result = prediction_service.predict_electricity(
            user_data=[{'wrong_field': 'value'}],
            periods=24
        )
        
        assert result['success'] == False
    
    def test_negative_periods(self, prediction_service, sample_electricity_data):
        """Test handling of negative periods."""
        # The service should handle this gracefully
        result = prediction_service.predict_electricity(
            user_data=sample_electricity_data,
            periods=-10
        )
        
        # Should either fail or use default periods
        if result['success']:
            assert len(result['predictions']) > 0
    
    def test_very_large_periods(self, prediction_service, sample_electricity_data):
        """Test handling of very large periods request."""
        result = prediction_service.predict_electricity(
            user_data=sample_electricity_data,
            periods=10000  # Very large
        )
        
        # Should cap at max periods
        if result['success']:
            assert len(result['predictions']) <= 168  # Max 7 days


if __name__ == '__main__':
    pytest.main([__file__, '-v'])