"""Configuration for correlation matrices."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Set


class TrackType(Enum):
    """NASCAR track types."""
    SUPERSPEEDWAY = "superspeedway"
    INTERMEDIATE = "intermediate"
    ROAD = "road"
    SHORT_TRACK = "short_track"


class CorrelationType(Enum):
    """Types of correlations."""
    MANUFACTURER = auto()
    TEAM = auto()
    STARTING_POSITION = auto()
    PRACTICE_PERFORMANCE = auto()
    POSITION_ADVANCEMENT = auto()
    TRACK_SPECIALIST = auto()
    CONSISTENCY = auto()
    OPPONENT = auto()
    FIGHTING_STYLE = auto()
    QB_STACK = auto()
    GAME_STACK = auto()
    DEFENSE_STACK = auto()
    SAME_TEAM = auto()
    NEGATIVE_CORRELATION = auto()


class Manufacturer(Enum):
    """NASCAR manufacturers."""
    CHEVROLET = "Chevrolet"
    FORD = "Ford"
    TOYOTA = "Toyota"


class FightingStyle(Enum):
    """MMA fighting styles."""
    FINISHER = "finisher"
    DECISION = "decision"
    BALANCED = "balanced"


@dataclass
class NASCARCorrelationConfig:
    """NASCAR correlation configuration."""
    # Track multipliers
    SUPERSPEEDWAY_MULTIPLIER: float = 1.2
    DEFAULT_MULTIPLIER: float = 1.0

    # Team/manufacturer correlations
    SAME_TEAM_CORRELATION: float = 0.30
    SAME_MANUFACTURER_CORRELATION: float = 0.15

    # Position correlations
    STARTING_POSITION_CORRELATION: float = 0.10
    STARTING_POSITION_THRESHOLD: int = 3

    # Dominator vs field
    FRONT_RUNNER_THRESHOLD: int = 5
    BACK_MARKER_THRESHOLD: int = 25
    FRONT_VS_BACK_CORRELATION: float = -0.15

    # Pack racing (superspeedway)
    PACK_RACING_CORRELATION: float = 0.05

    # Practice performance
    PRACTICE_VERY_SIMILAR_TIME: float = 0.05  # seconds
    PRACTICE_SIMILAR_TIME: float = 0.10  # seconds
    PRACTICE_VERY_SIMILAR_CORRELATION: float = 0.18
    PRACTICE_SIMILAR_CORRELATION: float = 0.10

    # Position advancement
    PASS_DIFF_THRESHOLD: float = 3.0
    AGGRESSIVE_DRIVER_THRESHOLD: float = 2.0
    AGGRESSIVE_CORRELATION: float = 0.12
    NORMAL_CORRELATION: float = 0.08

    # Track specialists
    TRACK_ADVANTAGE_THRESHOLD: float = 2.0
    SPECIALIST_ADVANTAGE_THRESHOLD: float = 3.0
    SPECIALIST_CORRELATION: float = 0.15
    NORMAL_TRACK_CORRELATION: float = 0.10

    # Consistency vs volatility
    HIGH_QUALITY_PASSES_THRESHOLD: int = 65
    LOW_QUALITY_PASSES_THRESHOLD: int = 35
    CONSISTENCY_ANTI_CORRELATION: float = -0.06

    # Road course specialists
    ROAD_SPECIALIST_THRESHOLD: float = 10.0
    ROAD_SPECIALIST_CORRELATION: float = 0.25

    # Valid manufacturers
    VALID_MANUFACTURERS: Set[str] = None

    def __post_init__(self):
        if self.VALID_MANUFACTURERS is None:
            self.VALID_MANUFACTURERS = {m.value for m in Manufacturer}


@dataclass
class MMACorrelationConfig:
    """MMA correlation configuration."""
    # Opponent correlations
    OPPONENT_CORRELATION: float = -0.95

    # Favorites correlations
    STRONG_FAVORITE_THRESHOLD: int = -150
    FAVORITE_ODDS_DIFF_THRESHOLD: int = 100
    FAVORITE_CORRELATION: float = 0.15

    # Ownership anti-correlation
    HIGH_OWNERSHIP_THRESHOLD: float = 25.0
    LOW_OWNERSHIP_THRESHOLD: float = 10.0
    OWNERSHIP_ANTI_CORRELATION: float = -0.10

    # ITD correlations
    HIGH_ITD_THRESHOLD: float = 0.4
    VERY_HIGH_ITD_THRESHOLD: float = 0.6
    ITD_CORRELATION: float = 0.20

    # Fighting style correlations
    DECISION_FIGHTER_THRESHOLD: float = 0.25
    FINISHER_CORRELATION: float = 0.30
    DECISION_CORRELATION: float = 0.25
    STYLE_ANTI_CORRELATION: float = -0.05

    # Default ITD probability
    DEFAULT_ITD_PROBABILITY: float = 0.35


@dataclass
class NFLCorrelationConfig:
    """NFL correlation configuration."""
    # Stacking correlations
    QB_PASS_CATCHER_CORRELATION: float = 0.75
    QB_RB_CORRELATION: float = 0.25
    RB_DST_CORRELATION: float = 0.45
    WR_WR_CORRELATION: float = 0.20
    TE_WR_CORRELATION: float = 0.15

    # Game stack correlation
    GAME_STACK_CORRELATION: float = 0.20

    # Negative correlations
    DST_OPPOSING_OFFENSE_CORRELATION: float = -0.30

    # Position sets
    PASS_CATCHERS: Set[str] = None
    OFFENSIVE_POSITIONS: Set[str] = None

    def __post_init__(self):
        if self.PASS_CATCHERS is None:
            self.PASS_CATCHERS = {'WR', 'TE'}
        if self.OFFENSIVE_POSITIONS is None:
            self.OFFENSIVE_POSITIONS = {'QB', 'WR', 'TE', 'RB'}


# Default instances
NASCAR_CORRELATION_CONFIG = NASCARCorrelationConfig()
MMA_CORRELATION_CONFIG = MMACorrelationConfig()
NFL_CORRELATION_CONFIG = NFLCorrelationConfig()