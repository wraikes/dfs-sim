"""Configuration constants for projection adjustments."""

from dataclasses import dataclass


@dataclass
class NewsletterConfig:
    """Newsletter signal adjustment parameters."""
    # Target multipliers
    TARGET_BASE_MULTIPLIER: float = 1.02  # Base boost for targets
    TARGET_CONFIDENCE_MULTIPLIER: float = 0.08  # Additional boost per confidence point

    # Avoid multipliers
    AVOID_BASE_MULTIPLIER: float = 0.96  # Base reduction for avoids
    AVOID_CONFIDENCE_MULTIPLIER: float = 0.06  # Additional reduction per confidence point

    # Ownership bounds
    MIN_OWNERSHIP: float = 0.1  # Minimum ownership percentage
    MAX_OWNERSHIP: float = 100.0  # Maximum ownership percentage


@dataclass
class OwnershipConfig:
    """Salary-ownership correlation parameters."""
    # Salary-based ownership range
    MIN_SALARY_OWNERSHIP: float = 2.0   # Min ownership for lowest salary
    MAX_SALARY_OWNERSHIP: float = 20.0  # Max ownership for highest salary
    DEFAULT_OWNERSHIP: float = 15.0     # Default when all salaries equal

    # Blending weights
    ORIGINAL_WEIGHT: float = 0.5  # Weight for original ownership
    SALARY_WEIGHT: float = 0.5    # Weight for salary-based ownership


@dataclass
class ValidationConfig:
    """Validation thresholds for projection adjustments."""
    # Reasonable ranges
    MIN_PROJECTION: float = 0.1
    MAX_PROJECTION: float = 200.0

    MIN_FLOOR: float = 0.0
    MAX_FLOOR: float = 150.0

    MIN_CEILING: float = 0.1
    MAX_CEILING: float = 300.0

    # Relationships
    MIN_CEILING_BUFFER: float = 1.05  # Ceiling should be >= projection * 1.05
    MAX_FLOOR_RATIO: float = 0.95     # Floor should be <= projection * 0.95


# Default instances
NEWSLETTER_CONFIG = NewsletterConfig()
OWNERSHIP_CONFIG = OwnershipConfig()
VALIDATION_CONFIG = ValidationConfig()