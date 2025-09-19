"""NASCAR-specific configuration constants."""

from dataclasses import dataclass
from typing import Dict, Any


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



@dataclass
class NASCAROptimizationConfig:
    """NASCAR optimization-specific parameters."""
    # Basic constraints
    SALARY_CAP: int = 50000
    ROSTER_SIZE: int = 6
    MIN_SALARY_REMAINING: int = 300
    MIN_LEVERAGE_PLAYS: int = 2
    MAX_MANUFACTURER_COUNT: int = 2
    MAX_TEAM_COUNT: int = 2

    # Superspeedway constraints
    SUPERSPEEDWAY_MAX_SALARY_REMAINING: int = 2000
    SUPERSPEEDWAY_MAX_OWNERSHIP: float = 120.0
    SUPERSPEEDWAY_MAX_OVERLAP: float = 0.4

    # Non-superspeedway constraints
    INTERMEDIATE_MAX_SALARY_REMAINING: int = 1500
    INTERMEDIATE_MAX_OWNERSHIP: float = 135.0
    INTERMEDIATE_MAX_OVERLAP: float = 0.5


@dataclass
class NASCARPositionConfig:
    """Position-based configuration for NASCAR."""
    # Position thresholds
    DOMINATOR_THRESHOLD: int = 12  # P1-P12 are dominators
    FRONT_RUNNER_THRESHOLD: int = 5  # P1-P5 are front runners
    MID_TIER_THRESHOLD: int = 17  # P13-P17 are mid-tier
    PD_THRESHOLD: int = 18  # P18+ are PD plays
    BACK_PACK_THRESHOLD: int = 23  # P23+ are back pack

    # Superspeedway rules
    SUPERSPEEDWAY_MAX_FROM_TOP12: int = 1
    SUPERSPEEDWAY_MIN_FROM_BACK: int = 4

    # Non-superspeedway rules
    INTERMEDIATE_MAX_DOMINATORS: int = 2
    INTERMEDIATE_MIN_PD_PLAYS: int = 3
    INTERMEDIATE_MAX_FROM_TOP12: int = 2


@dataclass
class NASCARScoringConfig:
    """NASCAR-specific scoring parameters."""
    # Bonus values
    PD_PLAY_BONUS: float = 5.0  # Bonus per P18+ driver
    DEEP_PD_BONUS: float = 3.0  # Extra bonus for P25+ drivers

    # Dominator bonuses
    SUPERSPEEDWAY_DOMINATOR_BONUS: float = 10.0
    INTERMEDIATE_DOMINATOR_BONUS_SINGLE: float = 8.0
    INTERMEDIATE_DOMINATOR_BONUS_DOUBLE: float = 5.0

    # Track bonuses
    SUPERSPEEDWAY_CHAOS_BONUS: float = 15.0
    INTERMEDIATE_BALANCE_BONUS: float = 10.0

    # Penalties
    DOMINATOR_VIOLATION_PENALTY: float = -15.0
    INTERMEDIATE_DOMINATOR_PENALTY: float = -10.0
    MANUFACTURER_VIOLATION_PENALTY: float = -10.0
    TEAM_VIOLATION_PENALTY: float = -8.0

    # Value metrics normalization
    MAX_DOMINATOR_SCORE: float = 15.0
    MAX_PD_UPSIDE: float = 10.0


@dataclass
class NASCARValueConfig:
    """NASCAR value calculation parameters."""
    # Position-based value scores
    FRONT_RUNNER_VALUE: float = 0.8  # P1-P12
    BACK_MARKER_VALUE: float = 0.7   # P25+
    MID_PACK_VALUE: float = 0.5      # P13-P24

    # Component weights for PD plays
    EFFICIENCY_WEIGHT: float = 0.4
    UPSIDE_WEIGHT: float = 0.3
    DOMINATOR_WEIGHT: float = 0.3

    # Selection probabilities
    DOMINATOR_SELECTION_PROBABILITY: float = 0.7
    BEST_PD_SELECTION_PROBABILITY: float = 0.8

    # Candidate limits
    MAX_DOMINATOR_CANDIDATES: int = 6
    MAX_PD_CANDIDATES: int = 12
    MAX_PD_EXPANDED: int = 10


@dataclass
class NASCAROwnershipConfig:
    """NASCAR ownership-based configuration."""
    LEVERAGE_THRESHOLD: float = 12.0  # ≤12% owned players are leverage
    HIGH_OWNERSHIP_THRESHOLD: float = 25.0  # ≥25% owned players are chalk
    MAX_HIGH_OWNED: int = 3  # Max 3 drivers ≥25% owned


@dataclass
class NASCARTrackRules:
    """Track-specific rule configurations."""

    def get_superspeedway_rules(self) -> Dict[str, Any]:
        """Get superspeedway-specific rules."""
        return {
            'track_type': 'superspeedway',
            'max_from_top12': NASCARPositionConfig.SUPERSPEEDWAY_MAX_FROM_TOP12,
            'min_from_back': NASCARPositionConfig.SUPERSPEEDWAY_MIN_FROM_BACK,
            'max_manufacturer_count': NASCAROptimizationConfig.MAX_MANUFACTURER_COUNT,
            'max_team_count': NASCAROptimizationConfig.MAX_TEAM_COUNT,
        }

    def get_intermediate_rules(self) -> Dict[str, Any]:
        """Get intermediate/road course rules."""
        return {
            'track_type': 'intermediate',
            'max_dominators': NASCARPositionConfig.INTERMEDIATE_MAX_DOMINATORS,
            'min_pd_plays': NASCARPositionConfig.INTERMEDIATE_MIN_PD_PLAYS,
            'max_from_top12': NASCARPositionConfig.INTERMEDIATE_MAX_FROM_TOP12,
            'max_manufacturer_count': NASCAROptimizationConfig.MAX_MANUFACTURER_COUNT,
            'max_team_count': NASCAROptimizationConfig.MAX_TEAM_COUNT,
        }


# Default instances
NASCAR_CONFIG = NASCARConfig()
NASCAR_OPTIMIZATION_CONFIG = NASCAROptimizationConfig()
NASCAR_POSITION_CONFIG = NASCARPositionConfig()
NASCAR_SCORING_CONFIG = NASCARScoringConfig()
NASCAR_VALUE_CONFIG = NASCARValueConfig()
NASCAR_OWNERSHIP_CONFIG = NASCAROwnershipConfig()
NASCAR_TRACK_RULES = NASCARTrackRules()