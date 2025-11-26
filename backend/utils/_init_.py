# ============================================
# EchoMind - Utilities Package Initialization
# ============================================

"""
Utilities Package

This package contains utility functions and helpers for EchoMind.

Modules:
    - validators: Input validation functions
    - helpers: General helper functions
    - logger: Logging configuration and utilities
"""

from backend.utils.validators import (
    validate_prediction_request,
    validate_consumption_data,
    validate_user_data,
    validate_usage_entry,
    validate_combined_request,
    ValidationError,
)
from backend.utils.helpers import (
    format_datetime,
    parse_datetime,
    calculate_percentage_change,
    round_to_precision,
    generate_date_range,
    convert_units,
)
from backend.utils.logger import (
    logger,
    setup_logger,
    log_request,
    log_error,
)

# Package exports
__all__ = [
    # Validators
    "validate_prediction_request",
    "validate_consumption_data",
    "validate_user_data",
    "validate_usage_entry",
    "validate_combined_request",
    "ValidationError",
    # Helpers
    "format_datetime",
    "parse_datetime",
    "calculate_percentage_change",
    "round_to_precision",
    "generate_date_range",
    "convert_units",
    # Logger
    "logger",
    "setup_logger",
    "log_request",
    "log_error",
]


# Utility constants
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIME_FORMAT = "%H:%M:%S"

# Resource type constants
RESOURCE_TYPES = ["electricity", "water"]
FREQUENCY_OPTIONS = ["H", "D", "W", "M"]  # Hourly, Daily, Weekly, Monthly

# Unit conversions
UNIT_CONVERSIONS = {
    "kwh_to_wh": 1000,
    "wh_to_kwh": 0.001,
    "liters_to_gallons": 0.264172,
    "gallons_to_liters": 3.78541,
    "liters_to_cubic_meters": 0.001,
    "cubic_meters_to_liters": 1000,
}