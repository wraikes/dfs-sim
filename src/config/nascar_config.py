"""NASCAR-specific configuration constants."""

from dataclasses import dataclass


@dataclass
class NASCARConfig:
    """NASCAR projection adjustment parameters."""
    # Position differential multipliers
    PD_CEILING_MULTIPLIER: float = 0.02  # Boost ceiling for PD upside

    # Dominator scoring
    DOMINATOR_THRESHOLD: float = 5.0     # Minimum dominator score for boost
    DOMINATOR_MULTIPLIER: float = 0.01   # Projection boost per dominator point

    # Track expertise
    TRACK_EXPERT_THRESHOLD: float = 10.0  # Avg finish to be considered expert
    TRACK_EXPERT_FLOOR_BOOST: float = 1.1 # Floor multiplier for track experts

    # DNF risk
    DNF_FLOOR_PENALTY: float = 0.95      # Floor reduction for high DNF risk
    HIGH_DNF_THRESHOLD: float = 15.0     # DNF % threshold


# Default instance
NASCAR_CONFIG = NASCARConfig()