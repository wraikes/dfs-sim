"""Configuration constants for optimization engine."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SimulationConfig:
    """Monte Carlo simulation parameters."""
    DEFAULT_N_SIMULATIONS: int = 10000
    CONSENSUS_N_SIMULATIONS: int = 25000
    MAX_ATTEMPTS: int = 100
    CANDIDATE_MULTIPLIER: int = 200  # Generate 200 candidates per lineup


@dataclass
class ConstraintsConfig:
    """Universal constraint parameters."""
    DEFAULT_MAX_SALARY_REMAINING: int = 1500
    DEFAULT_MIN_SALARY_REMAINING: int = 300
    DEFAULT_MAX_LINEUP_OWNERSHIP: float = 140.0
    DEFAULT_MIN_LEVERAGE_PLAYS: int = 2
    DEFAULT_MAX_LINEUP_OVERLAP: float = 0.33


@dataclass
class ScoringConfig:
    """Lineup scoring parameters."""
    # Value-based leverage thresholds
    VALUE_THRESHOLD: float = 0.4  # Minimum value score for leverage
    BASE_VALUE_WEIGHT: float = 0.6  # Weight for base metrics
    SPORT_VALUE_WEIGHT: float = 0.4  # Weight for sport metrics

    # Value component weights
    EFFICIENCY_WEIGHT: float = 0.5
    UPSIDE_WEIGHT: float = 0.5

    # Leverage scoring
    VALUE_BONUS_MULTIPLIER: float = 12.0  # 0.4-1.0 → 4.8-12.0 bonus
    OWNERSHIP_TIEBREAKER_MAX: float = 3.0  # Max ownership points
    OWNERSHIP_REFERENCE: float = 50.0  # Reference ownership percentage


@dataclass
class EfficiencyConfig:
    """Sport-specific efficiency thresholds for value scoring."""
    THRESHOLDS: Dict[str, float] = None

    def __post_init__(self):
        if self.THRESHOLDS is None:
            self.THRESHOLDS = {
                'nfl': 3.0,    # NFL has lower pts/$
                'nba': 5.0,    # NBA is middle
                'mlb': 3.5,    # MLB similar to NFL
                'nascar': 5.0, # NASCAR similar to NBA
                'mma': 12.0,   # MMA has highest pts/$
                'pga': 7.0     # PGA in between
            }

    def get_threshold(self, sport: str) -> float:
        """Get efficiency threshold for sport."""
        return self.THRESHOLDS.get(sport.lower(), 5.0)


@dataclass
class ConsensusConfig:
    """Consensus optimization parameters."""
    MAX_SEEDS: int = 100
    NFL_CONSENSUS_SEEDS: int = 25

    # Selection methods
    NFL_SELECTION_METHOD: str = 'highest_salary'  # NFL uses highest salary
    DEFAULT_SELECTION_METHOD: str = 'duplicate_detection'  # Others use duplicates


@dataclass
class FieldConfig:
    """Field generation parameters."""
    DEFAULT_FIELD_SIZE: int = 10000
    MIN_SALARY_RANGE: int = 48000
    MAX_SALARY_RANGE: int = 50000


# Default instances
SIMULATION_CONFIG = SimulationConfig()
CONSTRAINTS_CONFIG = ConstraintsConfig()
SCORING_CONFIG = ScoringConfig()
EFFICIENCY_CONFIG = EfficiencyConfig()
CONSENSUS_CONFIG = ConsensusConfig()
FIELD_CONFIG = FieldConfig()