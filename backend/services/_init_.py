# ============================================
# EchoMind - Services Package Initialization
# ============================================

"""
Business Logic Services Package

This package contains service classes that handle the core business logic
for EchoMind. Services act as an intermediary between API routes and
data models/predictors.

Services:
    - ElectricityService: Electricity consumption operations
    - WaterService: Water consumption operations
    - PredictionService: Unified prediction operations
    - UserService: User management operations

Each service encapsulates:
    - Data validation and transformation
    - Business rule enforcement
    - Interaction with ML models
    - Database operations
"""

from backend.services.electricity_service import ElectricityService
from backend.services.water_service import WaterService
from backend.services.prediction_service import PredictionService
from backend.services.user_service import UserService

# Package exports
__all__ = [
    "ElectricityService",
    "WaterService",
    "PredictionService",
    "UserService",
]

# Service registry for dependency injection
SERVICE_REGISTRY = {
    "electricity": ElectricityService,
    "water": WaterService,
    "prediction": PredictionService,
    "user": UserService,
}


def get_service(service_name: str):
    """
    Factory function to get service instance by name.

    Args:
        service_name: Name of the service

    Returns:
        Service instance

    Raises:
        ValueError: If service name is not recognized
    """
    if service_name not in SERVICE_REGISTRY:
        raise ValueError(
            f"Unknown service: {service_name}. "
            f"Available services: {list(SERVICE_REGISTRY.keys())}"
        )
    return SERVICE_REGISTRY[service_name]()


def list_available_services():
    """
    List all available services.

    Returns:
        List of available service names
    """
    return list(SERVICE_REGISTRY.keys())