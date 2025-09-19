"""NASCAR-specific configuration constants."""

from dataclasses import dataclass


@dataclass
class NASCARConfig:
    """NASCAR projection adjustment parameters."""
    # Position differential multipliers
    PD_CEILING_MULTIPLIER: float = 0.02  # Boost ceiling for PD upside
    PD_TARGET_POSITION: int = 15         # Target finish position for PD calculation

    # Dominator scoring
    DOMINATOR_THRESHOLD: float = 5.0     # Minimum dominator score for boost
    DOMINATOR_MULTIPLIER: float = 0.01   # Projection boost per dominator point

    # Track expertise
    TRACK_EXPERT_THRESHOLD: float = 10.0  # Avg finish to be considered expert
    TRACK_EXPERT_FLOOR_BOOST: float = 1.1 # Floor multiplier for track experts
    TRACK_FPPG_THRESHOLD: float = 35.0   # FPPG threshold for track specialists

    # DNF risk thresholds by starting position
    DNF_FRONT_RUNNERS: float = 0.08      # DNF risk for positions 1-5
    DNF_MID_PACK: float = 0.12           # DNF risk for positions 6-15
    DNF_BACK_MID: float = 0.18           # DNF risk for positions 16-25
    DNF_BACK_PACK: float = 0.25          # DNF risk for positions 26+
    DNF_NO_POSITION: float = 0.15        # DNF risk when no starting position data
    DNF_FLOOR_PENALTY: float = 0.95      # Floor reduction for high DNF risk
    HIGH_DNF_THRESHOLD: float = 15.0     # DNF % threshold

    # Starting position thresholds
    FRONT_RUNNER_THRESHOLD: int = 5      # Top positions (1-5)
    MID_PACK_THRESHOLD: int = 15         # Mid positions (6-15)
    BACK_MID_THRESHOLD: int = 25         # Back-mid positions (16-25)

    # Variance multipliers by position
    FRONT_RUNNER_VARIANCE: float = 0.9   # More consistent
    BACK_PACK_VARIANCE: float = 1.2      # More volatile

    # Ceiling adjustments by position
    BACK_CEILING_BOOST: float = 1.15     # Positions 16+ ceiling boost
    MID_CEILING_BOOST: float = 1.05      # Positions 11-15 ceiling boost

    # Floor calculation parameters
    FRONT_RUNNER_FLOOR: int = 25         # Base floor points for positions 1-5
    MID_PACK_FLOOR: int = 15             # Base floor points for positions 6-15
    BACK_MID_FLOOR: int = 8              # Base floor points for positions 16-25
    BACK_PACK_FLOOR: int = 3             # Base floor points for positions 26+
    MAX_FLOOR_RATIO: float = 0.30        # Floor can't exceed 30% of projection
    MAX_UPDATED_FLOOR_RATIO: float = 0.8 # Updated floor max ratio to projection

    # DNF estimation parameters
    MAX_DNF_ESTIMATE: float = 30.0       # Cap estimated DNF rate at 30%
    DNF_CALCULATION_MULTIPLIER: float = 25.0  # Multiplier for FPPG deficit to DNF rate


# Default instance
NASCAR_CONFIG = NASCARConfig()