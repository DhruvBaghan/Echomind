# ============================================
# EchoMind - Helper Utilities
# ============================================

"""
General helper functions for EchoMind.

Provides utilities for:
    - Date/time formatting and parsing
    - Mathematical calculations
    - Data transformations
    - Unit conversions
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import math


def format_datetime(
    dt: datetime,
    format_type: str = "iso"
) -> str:
    """
    Format datetime to string.
    
    Args:
        dt: Datetime object to format
        format_type: Format type ('iso', 'date', 'time', 'display', 'short')
        
    Returns:
        Formatted datetime string
    """
    if not isinstance(dt, datetime):
        return str(dt)
    
    formats = {
        "iso": "%Y-%m-%dT%H:%M:%S",
        "date": "%Y-%m-%d",
        "time": "%H:%M:%S",
        "display": "%B %d, %Y at %I:%M %p",
        "short": "%m/%d/%Y %H:%M",
        "day": "%A, %B %d",
    }
    
    fmt = formats.get(format_type, formats["iso"])
    return dt.strftime(fmt)


def parse_datetime(
    datetime_str: str,
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parse datetime string to datetime object.
    
    Args:
        datetime_str: String to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed datetime or default value
    """
    if not datetime_str:
        return default
    
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
    
    # Clean string
    datetime_str = datetime_str.strip().replace("+00:00", "Z").rstrip("Z")
    
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    
    return default


def calculate_percentage_change(
    old_value: float,
    new_value: float,
    decimal_places: int = 2
) -> Optional[float]:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        decimal_places: Number of decimal places to round to
        
    Returns:
        Percentage change or None if old_value is 0
    """
    if old_value == 0:
        return None
    
    change = ((new_value - old_value) / abs(old_value)) * 100
    return round(change, decimal_places)


def round_to_precision(
    value: float,
    precision: int = 2
) -> float:
    """
    Round value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        
    Returns:
        Rounded value
    """
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return 0.0
    
    return round(value, precision)


def generate_date_range(
    start_date: datetime,
    end_date: datetime,
    frequency: str = "D"
) -> List[datetime]:
    """
    Generate a list of dates between start and end.
    
    Args:
        start_date: Start datetime
        end_date: End datetime
        frequency: Frequency ('H'=hourly, 'D'=daily, 'W'=weekly, 'M'=monthly)
        
    Returns:
        List of datetime objects
    """
    dates = []
    current = start_date
    
    deltas = {
        "H": timedelta(hours=1),
        "D": timedelta(days=1),
        "W": timedelta(weeks=1),
        "M": timedelta(days=30),  # Approximate
    }
    
    delta = deltas.get(frequency, deltas["D"])
    
    while current <= end_date:
        dates.append(current)
        current += delta
    
    return dates


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert between different units.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    # Define conversion factors (to base unit)
    conversions = {
        # Energy (base: kWh)
        "wh": 0.001,
        "kwh": 1.0,
        "mwh": 1000.0,
        "joules": 2.778e-7,
        "btu": 0.000293071,
        
        # Volume (base: liters)
        "ml": 0.001,
        "liters": 1.0,
        "l": 1.0,
        "gallons": 3.78541,
        "gal": 3.78541,
        "cubic_meters": 1000.0,
        "m3": 1000.0,
        "cubic_feet": 28.3168,
        "ft3": 28.3168,
    }
    
    from_unit = from_unit.lower().replace(" ", "_")
    to_unit = to_unit.lower().replace(" ", "_")
    
    if from_unit not in conversions or to_unit not in conversions:
        raise ValueError(f"Unknown unit: {from_unit} or {to_unit}")
    
    # Convert to base unit, then to target unit
    base_value = value * conversions[from_unit]
    result = base_value / conversions[to_unit]
    
    return round(result, 6)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {
            "count": 0,
            "sum": 0,
            "mean": 0,
            "min": 0,
            "max": 0,
            "range": 0,
            "std": 0,
            "variance": 0,
        }
    
    n = len(values)
    total = sum(values)
    mean = total / n
    min_val = min(values)
    max_val = max(values)
    
    # Calculate variance and standard deviation
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    
    return {
        "count": n,
        "sum": round(total, 4),
        "mean": round(mean, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "range": round(max_val - min_val, 4),
        "std": round(std, 4),
        "variance": round(variance, 4),
    }


def calculate_moving_average(
    values: List[float],
    window: int = 3
) -> List[float]:
    """
    Calculate moving average of a list of values.
    
    Args:
        values: List of numeric values
        window: Window size for moving average
        
    Returns:
        List of moving averages
    """
    if not values or window <= 0:
        return []
    
    if window > len(values):
        window = len(values)
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        avg = sum(window_values) / len(window_values)
        result.append(round(avg, 4))
    
    return result


def detect_anomalies(
    values: List[float],
    threshold: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Detect anomalies in a list of values using z-score method.
    
    Args:
        values: List of numeric values
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        List of anomaly dictionaries with index and value
    """
    if len(values) < 3:
        return []
    
    stats = calculate_statistics(values)
    mean = stats["mean"]
    std = stats["std"]
    
    if std == 0:
        return []
    
    anomalies = []
    for i, value in enumerate(values):
        z_score = abs((value - mean) / std)
        if z_score > threshold:
            anomalies.append({
                "index": i,
                "value": value,
                "z_score": round(z_score, 2),
                "type": "high" if value > mean else "low",
            })
    
    return anomalies


def format_currency(
    amount: float,
    currency: str = "USD",
    decimal_places: int = 2
) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency code
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "INR": "₹",
        "CAD": "C$",
        "AUD": "A$",
    }
    
    symbol = symbols.get(currency, currency + " ")
    formatted = f"{symbol}{amount:,.{decimal_places}f}"
    
    return formatted


def format_consumption(
    value: float,
    resource_type: str,
    short: bool = False
) -> str:
    """
    Format consumption value with appropriate unit.
    
    Args:
        value: Consumption value
        resource_type: Type of resource (electricity/water)
        short: Use short unit name
        
    Returns:
        Formatted consumption string
    """
    if resource_type == "electricity":
        unit = "kWh" if short else "kilowatt-hours"
    elif resource_type == "water":
        unit = "L" if short else "liters"
    else:
        unit = "units"
    
    return f"{value:,.2f} {unit}"


def get_time_period_name(period: str) -> str:
    """
    Get human-readable name for time period code.
    
    Args:
        period: Period code (H, D, W, M, Y)
        
    Returns:
        Human-readable period name
    """
    periods = {
        "H": "Hourly",
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly",
        "Y": "Yearly",
        "today": "Today",
        "week": "This Week",
        "month": "This Month",
        "year": "This Year",
    }
    
    return periods.get(period, period)


def chunk_list(
    items: List[Any],
    chunk_size: int
) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        return [items]
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(
    *dicts: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        deep: Perform deep merge for nested dicts
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
        
        for key, value in d.items():
            if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
    
    return result


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(
    value: float,
    min_value: float,
    max_value: float
) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def interpolate_missing(
    data: List[Optional[float]],
    method: str = "linear"
) -> List[float]:
    """
    Interpolate missing values in a list.
    
    Args:
        data: List with potential None values
        method: Interpolation method ('linear', 'forward', 'backward', 'mean')
        
    Returns:
        List with interpolated values
    """
    result = list(data)
    n = len(result)
    
    if method == "mean":
        # Fill with mean of non-None values
        valid_values = [v for v in result if v is not None]
        mean_val = sum(valid_values) / len(valid_values) if valid_values else 0
        return [v if v is not None else mean_val for v in result]
    
    elif method == "forward":
        # Forward fill
        last_valid = 0
        for i in range(n):
            if result[i] is not None:
                last_valid = result[i]
            else:
                result[i] = last_valid
        return result
    
    elif method == "backward":
        # Backward fill
        last_valid = 0
        for i in range(n - 1, -1, -1):
            if result[i] is not None:
                last_valid = result[i]
            else:
                result[i] = last_valid
        return result
    
    else:  # linear
        # Linear interpolation
        for i in range(n):
            if result[i] is None:
                # Find previous and next valid values
                prev_idx = None
                next_idx = None
                
                for j in range(i - 1, -1, -1):
                    if result[j] is not None:
                        prev_idx = j
                        break
                
                for j in range(i + 1, n):
                    if result[j] is not None:
                        next_idx = j
                        break
                
                if prev_idx is not None and next_idx is not None:
                    # Linear interpolation
                    ratio = (i - prev_idx) / (next_idx - prev_idx)
                    result[i] = result[prev_idx] + ratio * (result[next_idx] - result[prev_idx])
                elif prev_idx is not None:
                    result[i] = result[prev_idx]
                elif next_idx is not None:
                    result[i] = result[next_idx]
                else:
                    result[i] = 0
        
        return result


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID string.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random portion
        
    Returns:
        Generated ID string
    """
    import random
    import string
    
    chars = string.ascii_lowercase + string.digits
    random_part = "".join(random.choices(chars, k=length))
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part