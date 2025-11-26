# ============================================
# EchoMind - Input Validators
# ============================================

"""
Input validation utilities for API requests.

Provides validation functions for:
    - Prediction requests
    - Consumption data
    - User registration/login
    - Usage entries
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from backend.config import Config


class ValidationError(Exception):
    """
    Custom exception for validation errors.
    
    Attributes:
        message: Error message
        errors: List of specific validation errors
    """
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []
    
    def __str__(self) -> str:
        if self.errors:
            return f"{self.message}: {', '.join(self.errors)}"
        return self.message


def validate_prediction_request(
    data: Dict[str, Any],
    resource_type: str = "electricity"
) -> Dict[str, Any]:
    """
    Validate prediction request data.
    
    Args:
        data: Request data dictionary
        resource_type: Type of resource (electricity/water)
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    validated = {}
    
    # Validate data array
    if "data" in data:
        consumption_data = data["data"]
        
        if not isinstance(consumption_data, list):
            errors.append("'data' must be an array")
        elif len(consumption_data) < 2:
            errors.append("'data' must contain at least 2 entries")
        else:
            try:
                validated["data"] = validate_consumption_data(
                    consumption_data,
                    resource_type
                )
            except ValidationError as e:
                errors.extend(e.errors)
    
    # Validate periods
    periods = data.get("periods", Config.DEFAULT_PREDICTION_PERIODS)
    if not isinstance(periods, int):
        try:
            periods = int(periods)
        except (ValueError, TypeError):
            errors.append("'periods' must be an integer")
            periods = Config.DEFAULT_PREDICTION_PERIODS
    
    if periods < 1:
        errors.append("'periods' must be at least 1")
    elif periods > Config.MAX_PREDICTION_PERIODS:
        errors.append(f"'periods' cannot exceed {Config.MAX_PREDICTION_PERIODS}")
    
    validated["periods"] = min(max(periods, 1), Config.MAX_PREDICTION_PERIODS)
    
    # Validate frequency
    frequency = data.get("frequency", "H")
    valid_frequencies = ["H", "D", "W", "M"]
    if frequency not in valid_frequencies:
        errors.append(f"'frequency' must be one of: {', '.join(valid_frequencies)}")
        frequency = "H"
    
    validated["frequency"] = frequency
    
    # Validate resource type
    if resource_type not in ["electricity", "water"]:
        errors.append("Invalid resource type")
    
    validated["resource_type"] = resource_type
    
    if errors:
        raise ValidationError("Validation failed", errors)
    
    return validated


def validate_consumption_data(
    data: List[Dict[str, Any]],
    resource_type: str = "electricity"
) -> List[Dict[str, Any]]:
    """
    Validate consumption data entries.
    
    Args:
        data: List of consumption entries
        resource_type: Type of resource
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    validated = []
    
    if not isinstance(data, list):
        raise ValidationError("Data must be a list")
    
    if len(data) == 0:
        raise ValidationError("Data cannot be empty")
    
    for i, entry in enumerate(data):
        entry_errors = []
        clean_entry = {}
        
        # Validate datetime
        if "datetime" not in entry and "date" not in entry and "timestamp" not in entry:
            entry_errors.append(f"Entry {i}: missing datetime field")
        else:
            datetime_value = entry.get("datetime") or entry.get("date") or entry.get("timestamp")
            try:
                if isinstance(datetime_value, str):
                    # Try parsing various formats
                    parsed = parse_datetime_string(datetime_value)
                    clean_entry["datetime"] = parsed.isoformat()
                elif isinstance(datetime_value, datetime):
                    clean_entry["datetime"] = datetime_value.isoformat()
                else:
                    entry_errors.append(f"Entry {i}: invalid datetime format")
            except Exception:
                entry_errors.append(f"Entry {i}: could not parse datetime '{datetime_value}'")
        
        # Validate consumption value
        if "consumption" not in entry and "value" not in entry and "usage" not in entry:
            entry_errors.append(f"Entry {i}: missing consumption field")
        else:
            consumption = entry.get("consumption") or entry.get("value") or entry.get("usage")
            try:
                consumption = float(consumption)
                if consumption < 0:
                    entry_errors.append(f"Entry {i}: consumption cannot be negative")
                else:
                    clean_entry["consumption"] = consumption
            except (ValueError, TypeError):
                entry_errors.append(f"Entry {i}: consumption must be a number")
        
        # Validate consumption limits based on resource type
        if "consumption" in clean_entry:
            if resource_type == "electricity":
                # Reasonable electricity limits (kWh per hour)
                if clean_entry["consumption"] > 100:
                    entry_errors.append(
                        f"Entry {i}: electricity consumption {clean_entry['consumption']} kWh/h seems unrealistic"
                    )
            elif resource_type == "water":
                # Reasonable water limits (liters per hour)
                if clean_entry["consumption"] > 10000:
                    entry_errors.append(
                        f"Entry {i}: water consumption {clean_entry['consumption']} L/h seems unrealistic"
                    )
        
        if entry_errors:
            errors.extend(entry_errors)
        else:
            validated.append(clean_entry)
    
    if errors:
        raise ValidationError(f"Found {len(errors)} validation errors", errors)
    
    if len(validated) < 2:
        raise ValidationError("Need at least 2 valid data points")
    
    return validated


def validate_user_data(
    data: Dict[str, Any],
    registration: bool = False
) -> Dict[str, Any]:
    """
    Validate user registration/update data.
    
    Args:
        data: User data dictionary
        registration: Whether this is for registration (requires password)
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    validated = {}
    
    # Validate email
    if "email" in data or registration:
        email = data.get("email", "").strip().lower()
        
        if not email:
            errors.append("Email is required")
        elif not validate_email(email):
            errors.append("Invalid email format")
        else:
            validated["email"] = email
    
    # Validate name
    if "name" in data or registration:
        name = data.get("name", "").strip()
        
        if not name:
            errors.append("Name is required")
        elif len(name) < 2:
            errors.append("Name must be at least 2 characters")
        elif len(name) > 100:
            errors.append("Name cannot exceed 100 characters")
        else:
            validated["name"] = name
    
    # Validate password (required for registration)
    if registration:
        password = data.get("password", "")
        
        if not password:
            errors.append("Password is required")
        elif len(password) < 6:
            errors.append("Password must be at least 6 characters")
        elif len(password) > 128:
            errors.append("Password cannot exceed 128 characters")
        else:
            validated["password"] = password
    
    # Validate household size
    if "household_size" in data:
        household_size = data.get("household_size")
        try:
            household_size = int(household_size)
            if household_size < 1:
                errors.append("Household size must be at least 1")
            elif household_size > 20:
                errors.append("Household size cannot exceed 20")
            else:
                validated["household_size"] = household_size
        except (ValueError, TypeError):
            errors.append("Household size must be a number")
    
    # Validate location
    if "location" in data:
        location = data.get("location", "").strip()
        if location:
            if len(location) > 255:
                errors.append("Location cannot exceed 255 characters")
            else:
                validated["location"] = location
    
    if errors:
        raise ValidationError("Validation failed", errors)
    
    return validated


def validate_usage_entry(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single usage entry.
    
    Args:
        data: Usage entry data
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    validated = {}
    
    # Validate resource type
    resource_type = data.get("resource_type", "").lower()
    if resource_type not in ["electricity", "water"]:
        errors.append("resource_type must be 'electricity' or 'water'")
    else:
        validated["resource_type"] = resource_type
    
    # Validate consumption
    consumption = data.get("consumption")
    if consumption is None:
        errors.append("consumption is required")
    else:
        try:
            consumption = float(consumption)
            if consumption < 0:
                errors.append("consumption cannot be negative")
            else:
                validated["consumption"] = consumption
        except (ValueError, TypeError):
            errors.append("consumption must be a number")
    
    # Validate datetime (optional)
    if "datetime" in data:
        try:
            dt = parse_datetime_string(data["datetime"])
            validated["datetime"] = dt.isoformat()
        except Exception:
            errors.append("Invalid datetime format")
    
    # Validate notes (optional)
    if "notes" in data:
        notes = data.get("notes", "").strip()
        if len(notes) > 500:
            errors.append("Notes cannot exceed 500 characters")
        elif notes:
            validated["notes"] = notes
    
    if errors:
        raise ValidationError("Validation failed", errors)
    
    return validated


def validate_combined_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate combined prediction request for both resources.
    
    Args:
        data: Request data with electricity and water sections
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    validated = {}
    
    # Validate electricity section
    if "electricity" in data:
        try:
            validated["electricity"] = validate_prediction_request(
                data["electricity"],
                resource_type="electricity"
            )
        except ValidationError as e:
            errors.append(f"Electricity: {e.message}")
    
    # Validate water section
    if "water" in data:
        try:
            validated["water"] = validate_prediction_request(
                data["water"],
                resource_type="water"
            )
        except ValidationError as e:
            errors.append(f"Water: {e.message}")
    
    # Need at least one resource
    if "electricity" not in validated and "water" not in validated:
        errors.append("Provide at least electricity or water data")
    
    # Validate common frequency
    frequency = data.get("frequency", "H")
    if frequency not in ["H", "D", "W", "M"]:
        frequency = "H"
    validated["frequency"] = frequency
    
    if errors:
        raise ValidationError("Validation failed", errors)
    
    return validated


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    
    # Basic email regex pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def parse_datetime_string(datetime_str: str) -> datetime:
    """
    Parse datetime string in various formats.
    
    Args:
        datetime_str: Datetime string to parse
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If parsing fails
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]
    
    # Handle timezone suffix
    datetime_str = datetime_str.replace("+00:00", "Z").rstrip("Z")
    
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse datetime: {datetime_str}")


def validate_date_range(
    start_date: Optional[str],
    end_date: Optional[str]
) -> tuple:
    """
    Validate and parse date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        Tuple of (start_datetime, end_datetime)
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    start = None
    end = None
    
    if start_date:
        try:
            start = parse_datetime_string(start_date)
        except ValueError:
            errors.append(f"Invalid start_date format: {start_date}")
    
    if end_date:
        try:
            end = parse_datetime_string(end_date)
        except ValueError:
            errors.append(f"Invalid end_date format: {end_date}")
    
    if start and end and start > end:
        errors.append("start_date cannot be after end_date")
    
    if errors:
        raise ValidationError("Date range validation failed", errors)
    
    return (start, end)


def validate_pagination(
    page: Optional[int] = None,
    per_page: Optional[int] = None,
    max_per_page: int = 100
) -> tuple:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number
        per_page: Items per page
        max_per_page: Maximum items per page
        
    Returns:
        Tuple of (page, per_page)
    """
    # Default values
    page = page or 1
    per_page = per_page or 20
    
    # Ensure valid ranges
    try:
        page = max(1, int(page))
    except (ValueError, TypeError):
        page = 1
    
    try:
        per_page = min(max(1, int(per_page)), max_per_page)
    except (ValueError, TypeError):
        per_page = 20
    
    return (page, per_page)


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    Sanitize string input.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Strip whitespace
    value = value.strip()
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]
    
    # Remove control characters (except newlines and tabs)
    value = "".join(
        char for char in value
        if char == "\n" or char == "\t" or (ord(char) >= 32 and ord(char) != 127)
    )
    
    return value