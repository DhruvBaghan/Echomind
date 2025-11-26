# ============================================
# EchoMind - User Service
# ============================================

"""
Service class for user-related operations.
Handles user management, authentication, usage history, and preferences.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import hashlib
import secrets

from backend.config import Config
from backend.utils.logger import logger


class UserService:
    """
    Service for user management operations.
    
    Provides methods for:
    - User registration and authentication
    - Profile management
    - Usage history tracking
    - User preferences
    - Consumption summaries
    """

    def __init__(self):
        """Initialize user service."""
        # In-memory storage for demo (replace with database in production)
        self._users: Dict[int, Dict[str, Any]] = {}
        self._usage_history: Dict[int, List[Dict[str, Any]]] = {}
        self._preferences: Dict[int, Dict[str, Any]] = {}
        self._next_user_id = 1
        self._next_entry_id = 1

    def create_user(
        self,
        email: str,
        name: str,
        password: str,
        household_size: Optional[int] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            email: User email
            name: User name
            password: User password
            household_size: Optional household size
            location: Optional location

        Returns:
            Result with user data or error
        """
        try:
            # Check if email exists
            for user in self._users.values():
                if user["email"].lower() == email.lower():
                    return {
                        "success": False,
                        "error": "Email already registered"
                    }

            # Hash password
            password_hash = self._hash_password(password)

            # Create user
            user_id = self._next_user_id
            self._next_user_id += 1

            user = {
                "id": user_id,
                "email": email.lower(),
                "name": name,
                "password_hash": password_hash,
                "household_size": household_size or 4,
                "location": location,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            self._users[user_id] = user
            self._usage_history[user_id] = []
            self._preferences[user_id] = self.get_default_preferences()

            logger.info(f"User created: {email}")

            return {
                "success": True,
                "message": "User created successfully",
                "user": {
                    "id": user_id,
                    "email": email,
                    "name": name,
                }
            }

        except Exception as e:
            logger.error(f"User creation error: {e}")
            return {
                "success": False,
                "error": "Failed to create user",
                "message": str(e)
            }

    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user.

        Args:
            email: User email
            password: User password

        Returns:
            Result with user data or error
        """
        try:
            # Find user by email
            user = None
            for u in self._users.values():
                if u["email"].lower() == email.lower():
                    user = u
                    break

            if not user:
                return {
                    "success": False,
                    "error": "Invalid email or password"
                }

            # Verify password
            password_hash = self._hash_password(password)
            if password_hash != user["password_hash"]:
                return {
                    "success": False,
                    "error": "Invalid email or password"
                }

            return {
                "success": True,
                "message": "Login successful",
                "user": {
                    "id": user["id"],
                    "email": user["email"],
                    "name": user["name"],
                }
            }

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                "success": False,
                "error": "Authentication failed",
                "message": str(e)
            }

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user profile.

        Args:
            user_id: User ID

        Returns:
            User profile or None
        """
        user = self._users.get(user_id)
        if not user:
            return None

        return {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "household_size": user["household_size"],
            "location": user["location"],
            "created_at": user["created_at"],
        }

    def update_user_profile(
        self,
        user_id: int,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user profile.

        Args:
            user_id: User ID
            data: Update data

        Returns:
            Update result
        """
        try:
            user = self._users.get(user_id)
            if not user:
                return {
                    "success": False,
                    "error": "User not found"
                }

            # Update allowed fields
            allowed_fields = ["name", "household_size", "location"]
            for field in allowed_fields:
                if field in data:
                    user[field] = data[field]

            user["updated_at"] = datetime.now().isoformat()

            return {
                "success": True,
                "message": "Profile updated",
                "profile": self.get_user_profile(user_id)
            }

        except Exception as e:
            logger.error(f"Profile update error: {e}")
            return {
                "success": False,
                "error": "Failed to update profile",
                "message": str(e)
            }

    def save_usage_entry(
        self,
        user_id: Optional[int],
        resource_type: str,
        consumption: float,
        datetime_str: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save a usage entry.

        Args:
            user_id: Optional user ID (None for anonymous)
            resource_type: Type of resource (electricity/water)
            consumption: Consumption value
            datetime_str: Optional datetime string
            notes: Optional notes

        Returns:
            Save result
        """
        try:
            entry_id = self._next_entry_id
            self._next_entry_id += 1

            entry = {
                "id": entry_id,
                "resource_type": resource_type,
                "consumption": consumption,
                "datetime": datetime_str or datetime.now().isoformat(),
                "notes": notes,
                "created_at": datetime.now().isoformat(),
            }

            # Store entry
            if user_id:
                if user_id not in self._usage_history:
                    self._usage_history[user_id] = []
                self._usage_history[user_id].append(entry)
            else:
                # Store in temporary anonymous storage
                if 0 not in self._usage_history:
                    self._usage_history[0] = []
                self._usage_history[0].append(entry)

            return {
                "success": True,
                "message": "Usage entry saved",
                "entry": entry
            }

        except Exception as e:
            logger.error(f"Save usage error: {e}")
            return {
                "success": False,
                "error": "Failed to save entry",
                "message": str(e)
            }

    def save_usage_bulk(
        self,
        user_id: Optional[int],
        entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Save multiple usage entries.

        Args:
            user_id: Optional user ID
            entries: List of entries

        Returns:
            Bulk save result
        """
        try:
            saved = 0
            errors = []

            for entry in entries:
                result = self.save_usage_entry(
                    user_id=user_id,
                    resource_type=entry.get("resource_type", "electricity"),
                    consumption=entry.get("consumption", 0),
                    datetime_str=entry.get("datetime"),
                    notes=entry.get("notes")
                )
                
                if result.get("success"):
                    saved += 1
                else:
                    errors.append(result.get("error"))

            return {
                "success": True,
                "saved": saved,
                "total": len(entries),
                "errors": errors if errors else None,
            }

        except Exception as e:
            logger.error(f"Bulk save error: {e}")
            return {
                "success": False,
                "error": "Bulk save failed",
                "message": str(e)
            }

    def get_usage_history(
        self,
        user_id: Optional[int],
        resource_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get usage history.

        Args:
            user_id: Optional user ID
            resource_type: Optional resource type filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            Usage history
        """
        try:
            # Get entries
            entries = self._usage_history.get(user_id or 0, [])
            
            # Filter by resource type
            if resource_type:
                entries = [e for e in entries if e["resource_type"] == resource_type]

            # Filter by date range
            if start_date:
                start = datetime.fromisoformat(start_date)
                entries = [e for e in entries if datetime.fromisoformat(e["datetime"]) >= start]
            
            if end_date:
                end = datetime.fromisoformat(end_date)
                entries = [e for e in entries if datetime.fromisoformat(e["datetime"]) <= end]

            # Sort by datetime (newest first)
            entries = sorted(entries, key=lambda x: x["datetime"], reverse=True)

            # Paginate
            total = len(entries)
            entries = entries[offset:offset + limit]

            return {
                "success": True,
                "entries": entries,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            }

        except Exception as e:
            logger.error(f"Get history error: {e}")
            return {
                "success": False,
                "error": "Failed to get history",
                "message": str(e)
            }

    def delete_usage_entry(self, user_id: int, entry_id: int) -> Dict[str, Any]:
        """
        Delete a usage entry.

        Args:
            user_id: User ID
            entry_id: Entry ID to delete

        Returns:
            Delete result
        """
        try:
            entries = self._usage_history.get(user_id, [])
            
            for i, entry in enumerate(entries):
                if entry["id"] == entry_id:
                    entries.pop(i)
                    return {
                        "success": True,
                        "message": "Entry deleted"
                    }

            return {
                "success": False,
                "error": "Entry not found"
            }

        except Exception as e:
            logger.error(f"Delete entry error: {e}")
            return {
                "success": False,
                "error": "Failed to delete entry",
                "message": str(e)
            }

    def get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences."""
        return {
            "electricity_rate": Config.ELECTRICITY_COST_PER_KWH,
            "water_rate": Config.WATER_COST_PER_LITER,
            "currency": "USD",
            "notifications_enabled": True,
            "email_reports": False,
            "prediction_periods": Config.DEFAULT_PREDICTION_PERIODS,
            "theme": "light",
            "language": "en",
        }

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Get user preferences.

        Args:
            user_id: User ID

        Returns:
            User preferences
        """
        return self._preferences.get(user_id, self.get_default_preferences())

    def update_user_preferences(
        self,
        user_id: int,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update user preferences.

        Args:
            user_id: User ID
            preferences: New preferences

        Returns:
            Update result
        """
        try:
            current = self._preferences.get(user_id, self.get_default_preferences())
            
            # Update allowed preferences
            allowed = [
                "electricity_rate", "water_rate", "currency",
                "notifications_enabled", "email_reports",
                "prediction_periods", "theme", "language"
            ]
            
            for key in allowed:
                if key in preferences:
                    current[key] = preferences[key]

            self._preferences[user_id] = current

            return {
                "success": True,
                "message": "Preferences updated",
                "preferences": current
            }

        except Exception as e:
            logger.error(f"Update preferences error: {e}")
            return {
                "success": False,
                "error": "Failed to update preferences",
                "message": str(e)
            }

    def get_consumption_summary(
        self,
        user_id: Optional[int],
        period: str = "month"
    ) -> Dict[str, Any]:
        """
        Get consumption summary.

        Args:
            user_id: Optional user ID
            period: Time period (week/month/year)

        Returns:
            Consumption summary
        """
        try:
            entries = self._usage_history.get(user_id or 0, [])
            
            # Calculate date range
            now = datetime.now()
            if period == "week":
                start = now - timedelta(days=7)
            elif period == "month":
                start = now - timedelta(days=30)
            else:  # year
                start = now - timedelta(days=365)

            # Filter entries
            filtered = [
                e for e in entries
                if datetime.fromisoformat(e["datetime"]) >= start
            ]

            # Calculate by resource type
            summary = {
                "electricity": {
                    "total": 0,
                    "count": 0,
                    "average": 0,
                },
                "water": {
                    "total": 0,
                    "count": 0,
                    "average": 0,
                }
            }

            for entry in filtered:
                rt = entry["resource_type"]
                if rt in summary:
                    summary[rt]["total"] += entry["consumption"]
                    summary[rt]["count"] += 1

            # Calculate averages
            for rt in summary:
                if summary[rt]["count"] > 0:
                    summary[rt]["average"] = round(
                        summary[rt]["total"] / summary[rt]["count"], 2
                    )
                summary[rt]["total"] = round(summary[rt]["total"], 2)

            # Calculate costs
            elec_cost = summary["electricity"]["total"] * Config.ELECTRICITY_COST_PER_KWH
            water_cost = summary["water"]["total"] * Config.WATER_COST_PER_LITER

            return {
                "period": period,
                "start_date": start.isoformat(),
                "end_date": now.isoformat(),
                "electricity": summary["electricity"],
                "water": summary["water"],
                "costs": {
                    "electricity": round(elec_cost, 2),
                    "water": round(water_cost, 2),
                    "total": round(elec_cost + water_cost, 2),
                    "currency": "USD",
                },
                "entries_analyzed": len(filtered),
            }

        except Exception as e:
            logger.error(f"Summary error: {e}")
            return {
                "error": str(e)
            }

    def get_recent_activity(
        self,
        user_id: Optional[int],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent user activity.

        Args:
            user_id: Optional user ID
            limit: Maximum entries to return

        Returns:
            List of recent activities
        """
        entries = self._usage_history.get(user_id or 0, [])
        
        # Sort by datetime (newest first)
        sorted_entries = sorted(
            entries,
            key=lambda x: x.get("created_at", x.get("datetime", "")),
            reverse=True
        )
        
        return sorted_entries[:limit]