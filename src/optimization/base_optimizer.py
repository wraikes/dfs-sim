"""Base optimizer class for sport-agnostic DFS optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config.optimization_config import (
    CONSENSUS_CONFIG,
    CONSTRAINTS_CONFIG,
    EFFICIENCY_CONFIG,
    FIELD_CONFIG,
    SCORING_CONFIG,
    SIMULATION_CONFIG,
)
from ..models.player import Player
from ..simulation.correlations import build_correlation_matrix
from ..simulation.simulator import Simulator
from .field_generator import BaseFieldGenerator


@dataclass
class SportConstraints:
    """Sport-specific constraints and configuration."""
    salary_cap: int
    roster_size: int
    max_salary_remaining: int = None
    min_salary_remaining: int = None
    max_lineup_ownership: float = None
    min_leverage_plays: int = None
    max_lineup_overlap: float = None

    # Sport-specific rules
    sport_rules: dict[str, Any] = None

    def __post_init__(self):
        if self.sport_rules is None:
            self.sport_rules = {}

        # Apply defaults from config if not provided
        if self.max_salary_remaining is None:
            self.max_salary_remaining = CONSTRAINTS_CONFIG.DEFAULT_MAX_SALARY_REMAINING
        if self.min_salary_remaining is None:
            self.min_salary_remaining = CONSTRAINTS_CONFIG.DEFAULT_MIN_SALARY_REMAINING
        if self.max_lineup_ownership is None:
            self.max_lineup_ownership = CONSTRAINTS_CONFIG.DEFAULT_MAX_LINEUP_OWNERSHIP
        if self.min_leverage_plays is None:
            self.min_leverage_plays = CONSTRAINTS_CONFIG.DEFAULT_MIN_LEVERAGE_PLAYS
        if self.max_lineup_overlap is None:
            self.max_lineup_overlap = CONSTRAINTS_CONFIG.DEFAULT_MAX_LINEUP_OVERLAP


@dataclass
class BaseLineup:
    """Base lineup class that all sports inherit from."""
    players: list[Player]
    total_salary: int = 0
    total_projection: float = 0
    total_ownership: float = 0
    total_ceiling: float = 0

    # GPP metrics
    leverage_score: float = 0
    uniqueness_score: float = 0
    gpp_score: float = 0

    # Simulation results
    simulated_scores: np.ndarray | None = None
    percentile_25: float = 0
    percentile_50: float = 0
    percentile_75: float = 0
    percentile_95: float = 0
    percentile_99: float = 0

    def __post_init__(self):
        self.total_salary = sum(p.salary for p in self.players)
        self.total_projection = sum(p.projection for p in self.players)
        self.total_ownership = sum(p.ownership for p in self.players)
        self.total_ceiling = sum(p.ceiling for p in self.players)

    def calculate_overlap_with_lineup(self, other: 'BaseLineup') -> float:
        """Calculate overlap percentage with another lineup."""
        my_ids = {p.player_id for p in self.players}
        other_ids = {p.player_id for p in other.players}
        overlap = len(my_ids & other_ids) / len(self.players)
        return overlap


class BaseOptimizer(ABC):
    """Abstract base class for sport-specific DFS optimizers."""

    def __init__(self, players: list[Player], sport: str, field_size: int = None):
        self.players = players
        self.sport = sport.lower()
        self.field_size = field_size or FIELD_CONFIG.DEFAULT_FIELD_SIZE
        self.cash_game_mode = False  # Default to GPP mode

        # Get sport-specific constraints
        self.constraints = self._get_sport_constraints()

        # Build correlation matrix
        self.correlation_matrix, self.correlation_rules = build_correlation_matrix(
            self.sport, players
        )

        # Initialize simulator
        self.simulator = Simulator(
            n_simulations=SIMULATION_CONFIG.DEFAULT_N_SIMULATIONS,
            correlation_matrix=self.correlation_matrix
        )

        # Initialize field generator
        self.field_generator = self._create_field_generator()
        self.field = []

        # Track generated lineups
        self.generated_lineups: list[BaseLineup] = []

    @abstractmethod
    def _get_sport_constraints(self) -> SportConstraints:
        """Get sport-specific constraints."""
        pass

    @abstractmethod
    def _create_field_generator(self) -> BaseFieldGenerator:
        """Create sport-specific field generator."""
        pass

    @abstractmethod
    def _validate_lineup(self, players: list[Player]) -> bool:
        """Validate lineup meets sport-specific rules."""
        pass

    @abstractmethod
    def _create_lineup(self, players: list[Player]) -> BaseLineup:
        """Create sport-specific lineup object."""
        pass

    def _calculate_value_adjusted_leverage(self, player: Player,
                                           sport_value_metrics: dict[str, float] = None) -> float:
        """
        Universal value-first leverage calculation.

        Args:
            player: Player to evaluate
            sport_value_metrics: Optional sport-specific value components (0-1 scale)
                                e.g., {'itd_prob': 0.5, 'dominator': 0.8}

        Returns:
            Value-adjusted leverage score
        """
        # Base value metrics (universal)
        pts_per_dollar = (player.adjusted_projection / player.salary) * 1000
        ceiling_upside = player.ceiling / player.adjusted_projection if player.adjusted_projection > 0 else 1.0

        # Sport-specific efficiency thresholds
        efficiency_cap = EFFICIENCY_CONFIG.get_threshold(self.sport)

        # Base value score components
        efficiency_score = min(pts_per_dollar / efficiency_cap, 1.0)
        upside_score = min(ceiling_upside / 2.0, 1.0)

        # Combine base metrics with sport-specific metrics
        if sport_value_metrics:
            # Weight base metrics vs sport-specific
            base_weight = SCORING_CONFIG.BASE_VALUE_WEIGHT
            sport_weight = SCORING_CONFIG.SPORT_VALUE_WEIGHT

            base_value = (efficiency_score * SCORING_CONFIG.EFFICIENCY_WEIGHT +
                         upside_score * SCORING_CONFIG.UPSIDE_WEIGHT) * base_weight
            sport_value = sum(sport_value_metrics.values()) / len(sport_value_metrics) * sport_weight
            value_score = base_value + sport_value
        else:
            # No sport-specific metrics, use base only
            value_score = (efficiency_score * SCORING_CONFIG.EFFICIENCY_WEIGHT +
                          upside_score * SCORING_CONFIG.UPSIDE_WEIGHT)

        # Only give leverage if player has good value
        if value_score >= SCORING_CONFIG.VALUE_THRESHOLD:
            # Value-first approach: value is primary, ownership is tiebreaker
            base_value_bonus = value_score * SCORING_CONFIG.VALUE_BONUS_MULTIPLIER

            # Small ownership tiebreaker (doesn't overwhelm value)
            ownership_tiebreaker = (max(0, SCORING_CONFIG.OWNERSHIP_REFERENCE - player.ownership) /
                                  SCORING_CONFIG.OWNERSHIP_REFERENCE * SCORING_CONFIG.OWNERSHIP_TIEBREAKER_MAX)

            return base_value_bonus + ownership_tiebreaker

        return 0

    def _calculate_lineup_value_leverage(self, lineup: BaseLineup,
                                        get_sport_metrics_fn: callable = None) -> float:
        """
        Calculate total value-adjusted leverage for a lineup.

        Args:
            lineup: Lineup to evaluate
            get_sport_metrics_fn: Optional function that returns sport-specific
                                 value metrics for a player

        Returns:
            Total value-adjusted leverage score
        """
        total_leverage = 0

        for player in lineup.players:
            sport_metrics = None
            if get_sport_metrics_fn:
                sport_metrics = get_sport_metrics_fn(player)

            player_leverage = self._calculate_value_adjusted_leverage(player, sport_metrics)
            total_leverage += player_leverage

        return total_leverage

    @abstractmethod
    def _score_lineup_gpp(self, lineup: BaseLineup) -> float:
        """Score lineup for GPP success."""
        pass

    def load_players_from_csv(self, csv_path: str) -> list[Player]:
        """Load players from processed CSV data."""
        if not pd.io.common.file_exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Filter out inactive players (zero projections)
        original_count = len(df)
        df = df[df['projection'] > 0]
        active_count = len(df)

        if original_count > active_count:
            print(f"   📋 Filtered {original_count - active_count} inactive players, using {active_count} active players")

        players = []
        for _, row in df.iterrows():
            player = self._create_player_from_row(row)
            players.append(player)

        return players

    def load_players_from_dataframe(self, df: pd.DataFrame) -> list[Player]:
        """Load players from DataFrame."""
        # Filter out inactive players (zero projections)
        original_count = len(df)
        df = df[df['projection'] > 0]
        active_count = len(df)

        if original_count > active_count:
            print(f"   📋 Filtered {original_count - active_count} inactive players, using {active_count} active players")

        players = []
        for _, row in df.iterrows():
            player = self._create_player_from_row(row)
            players.append(player)

        self.players = players

        # Recreate field generator with loaded players
        self.field_generator = self._create_field_generator()

        return players

    @abstractmethod
    def _create_player_from_row(self, row: pd.Series) -> Player:
        """Create Player object from CSV row."""
        pass

    def generate_field(self):
        """Generate opponent field for uniqueness scoring."""
        print(f"\n🎯 Generating opponent field for {self.sport.upper()} optimization...")
        self.field = self.field_generator.generate_field(self.field_size)

    def _calculate_lineup_uniqueness(self, lineup: BaseLineup) -> float:
        """Calculate how unique a lineup is vs the field."""
        if not self.field:
            return 1.0

        lineup_ids = {p.player_id for p in lineup.players}

        overlap_scores = []
        sample_size = min(1000, len(self.field))  # Sample for speed

        for field_lineup in self.field[:sample_size]:
            field_ids = field_lineup.get_player_ids()
            overlap = len(lineup_ids & field_ids) / len(lineup.players)
            overlap_scores.append(overlap)

        avg_overlap = np.mean(overlap_scores)
        uniqueness = 1.0 - avg_overlap

        return uniqueness

    def _calculate_lineup_diversity(self, lineup: BaseLineup) -> float:
        """Calculate diversity vs other generated lineups."""
        if not self.generated_lineups:
            return 1.0

        min_overlap = 1.0
        for other in self.generated_lineups:
            overlap = lineup.calculate_overlap_with_lineup(other)
            min_overlap = min(min_overlap, overlap)

        diversity = 1.0 - min_overlap
        return diversity

    def generate_lineup_candidates(self, num_candidates: int) -> list[BaseLineup]:
        """Generate candidate lineups for optimization."""
        candidates = []
        max_attempts = num_candidates * 100
        attempts = 0

        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1

            lineup = self._generate_single_lineup()
            if lineup and self._validate_lineup(lineup.players):
                # Check for duplicates
                is_duplicate = any(
                    lineup.calculate_overlap_with_lineup(existing) > 0.8
                    for existing in candidates
                )

                if not is_duplicate:
                    candidates.append(lineup)

        return candidates

    @abstractmethod
    def _generate_single_lineup(self) -> BaseLineup | None:
        """Generate a single candidate lineup."""
        pass

    def optimize_lineups(self, num_lineups: int = 20, use_consensus: bool = False) -> list[BaseLineup]:
        """Main optimization pipeline with optional multi-seed consensus."""
        if use_consensus and num_lineups == 1:
            return self._optimize_with_consensus()
        else:
            return self._optimize_standard(num_lineups)

    def _optimize_standard(self, num_lineups: int) -> list[BaseLineup]:
        """Standard optimization pipeline with deterministic robust selection."""
        print(f"\n🚀 Generating {num_lineups} {self.sport.upper()} GPP lineups...")
        print("=" * 60)

        # Generate field if not done
        if not self.field:
            self.generate_field()

        # Generate large pool of candidates for robust selection
        target_candidates = 200  # Manageable pool size
        candidates = self.generate_lineup_candidates(target_candidates)
        print(f"📊 Generated {len(candidates)} candidates for scoring...")

        # Score all candidates with GPP scoring
        for lineup in candidates:
            lineup.gpp_score = self._score_lineup_gpp(lineup)

        # Apply simple deterministic selection
        candidates = self._apply_deterministic_selection(candidates, num_lineups)

        # Take top N lineups (already sorted deterministically)
        final_lineups = candidates[:num_lineups]

        for lineup in final_lineups:
            self.generated_lineups.append(lineup)

        print(f"✅ Selected {len(final_lineups)} optimized lineups")
        return final_lineups

    def _optimize_with_consensus(self) -> list[BaseLineup]:
        """Multi-seed consensus optimization for perfect determinism."""
        # NFL-specific: Use highest salary selection
        if self.sport.lower() == 'nfl':
            return self._optimize_nfl_consensus()
        else:
            return self._optimize_duplicate_consensus()

    def _optimize_nfl_consensus(self) -> list[BaseLineup]:
        """NFL-specific consensus: Generate 25 lineups and select highest salary."""
        import random
        print("\n🎯 Generating consensus NFL lineup using highest-salary selection...")
        print("=" * 80)
        print("Generating 25 lineups and selecting the one with highest salary usage")
        print("=" * 80)

        # Store original simulation count and field
        original_n_sims = self.simulator.n_simulations
        original_field = self.field

        # Set parameters for consensus runs
        self.simulator.n_simulations = CONSENSUS_CONFIG.NFL_CONSENSUS_SEEDS * 1000
        all_lineups = []

        for seed in range(1, CONSENSUS_CONFIG.NFL_CONSENSUS_SEEDS + 1):
            print(f"🎲 Seed {seed:2d}/{CONSENSUS_CONFIG.NFL_CONSENSUS_SEEDS} ({seed*100//CONSENSUS_CONFIG.NFL_CONSENSUS_SEEDS:3d}% complete)...")

            # Set seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)

            # Reset field for this seed
            self.field = []

            # Run single optimization
            candidates = self.generate_lineup_candidates(SIMULATION_CONFIG.CANDIDATE_MULTIPLIER)

            # Score candidates
            for lineup in candidates:
                lineup.gpp_score = self._score_lineup_gpp(lineup)

            # Get best lineup using deterministic selection
            best_candidates = self._apply_deterministic_selection(candidates, 1)
            if best_candidates:
                best_lineup = best_candidates[0]
                all_lineups.append(best_lineup)

        # Restore original settings
        self.simulator.n_simulations = original_n_sims
        self.field = original_field

        # Select lineup with highest salary usage
        if all_lineups:
            highest_salary_lineup = max(all_lineups, key=lambda x: x.total_salary)

            print("\n🏆 NFL CONSENSUS RESULTS")
            print("=" * 50)
            print(f"Selected lineup with highest salary: ${highest_salary_lineup.total_salary:,}")
            print(f"Salary usage: {highest_salary_lineup.total_salary/50000*100:.1f}%")
            print(f"GPP score: {highest_salary_lineup.gpp_score:.1f}")

            self.generated_lineups.append(highest_salary_lineup)
            print("✅ Selected highest-salary consensus lineup")
            return [highest_salary_lineup]
        else:
            print("❌ Error: No lineups generated")
            return []

    def _optimize_duplicate_consensus(self) -> list[BaseLineup]:
        """General consensus: Run until duplicate found (for non-NFL sports)."""
        import random
        print(f"\n🎯 Generating consensus {self.sport.upper()} lineup using duplicate-detection approach...")
        print("=" * 80)
        print("Running seeds until duplicate lineup found (max 100 seeds)")
        print("=" * 80)

        # Store original simulation count and field
        original_n_sims = self.simulator.n_simulations
        original_field = self.field

        # Set parameters for consensus runs
        self.simulator.n_simulations = CONSENSUS_CONFIG.NFL_CONSENSUS_SEEDS * 1000
        seen_lineups = set()
        all_lineups = []
        max_seeds = CONSENSUS_CONFIG.MAX_SEEDS

        for seed in range(1, max_seeds + 1):
            print(f"🎲 Seed {seed:2d}/{max_seeds} ({seed*100//max_seeds:3d}% complete)...")

            # Set seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)

            # Reset field for this seed
            self.field = []

            # Run single optimization
            candidates = self.generate_lineup_candidates(SIMULATION_CONFIG.CANDIDATE_MULTIPLIER)

            # Score candidates
            for lineup in candidates:
                lineup.gpp_score = self._score_lineup_gpp(lineup)

            # Get best lineup using deterministic selection
            best_candidates = self._apply_deterministic_selection(candidates, 1)
            if best_candidates:
                best_lineup = best_candidates[0]
                lineup_signature = tuple(sorted(p.player_id for p in best_lineup.players))

                # Check for duplicate
                if lineup_signature in seen_lineups:
                    # Found duplicate! This is true consensus
                    print("\n🏆 DUPLICATE FOUND!")
                    print("=" * 50)
                    print(f"Lineup appeared twice after {seed} seeds - TRUE CONSENSUS!")

                    # Restore original settings
                    self.simulator.n_simulations = original_n_sims
                    self.field = original_field

                    self.generated_lineups.append(best_lineup)
                    print(f"✅ Selected consensus lineup (duplicate at seed {seed})")
                    return [best_lineup]

                # Add to seen lineups
                seen_lineups.add(lineup_signature)
                all_lineups.append((lineup_signature, best_lineup))

        # Restore original settings
        self.simulator.n_simulations = original_n_sims
        self.field = original_field

        # No duplicates found - fall back to highest GPP score
        print("\n🏆 NO DUPLICATES FOUND")
        print("=" * 50)
        print(f"All {max_seeds} lineups were unique - selecting highest GPP score")

        if all_lineups:
            best_lineup = max(all_lineups, key=lambda x: x[1].gpp_score)[1]
            self.generated_lineups.append(best_lineup)
            print(f"✅ Selected best lineup (GPP score: {best_lineup.gpp_score:.1f})")
            return [best_lineup]
        else:
            print("❌ Error: No lineups generated")
            return []

    def _apply_deterministic_selection(self, candidates: list[BaseLineup],
                                       target_lineups: int) -> list[BaseLineup]:
        """Apply simple deterministic selection from equivalent lineups."""
        if not candidates:
            return candidates

        # Find uncertainty threshold (5% of best score)
        max_score = max(lineup.gpp_score for lineup in candidates)
        threshold = max_score * 0.95

        # Get lineups within uncertainty threshold (statistically equivalent)
        equivalent_lineups = [lineup for lineup in candidates if lineup.gpp_score >= threshold]

        print(f"🎯 Deterministic selection: {len(equivalent_lineups)} candidates within 5% of best score ({max_score:.1f})")

        # Multi-level deterministic ranking using raw player data
        equivalent_lineups.sort(key=lambda x: (
            x.total_ownership,                              # 1st: Lowest total ownership (most contrarian)
            -(x.total_projection / x.total_salary * 1000),  # 2nd: Highest salary efficiency
            x.total_salary,                                 # 3rd: Lowest total salary (more savings)
            -x.total_projection,                            # 4th: Highest total projection (negative for desc)
            tuple(sorted(p.ownership for p in x.players)), # 6th: Player ownership pattern
            tuple(sorted(p.salary for p in x.players)),    # 7th: Player salary pattern
            tuple(sorted(p.adjusted_projection for p in x.players)), # 8th: Player projection pattern
            tuple(sorted(p.player_id for p in x.players))  # 9th: Final deterministic ID-based tiebreaker
        ))

        # If we need more lineups than equivalent ones, add the rest sorted by GPP score
        remaining_candidates = [lineup for lineup in candidates if lineup.gpp_score < threshold]
        remaining_candidates.sort(key=lambda x: x.gpp_score, reverse=True)

        # Combine: equivalent lineups first (sorted deterministically), then remaining by score
        all_candidates = equivalent_lineups + remaining_candidates

        return all_candidates

    def display_lineup_stats(self, lineups: list[BaseLineup]):
        """Display lineup statistics - can be overridden by sport."""
        print(f"\n📈 {self.sport.upper()} LINEUP STATISTICS")
        print("=" * 60)

        for i, lineup in enumerate(lineups[:5], 1):
            print(f"\n🏆 LINEUP #{i}")
            print("-" * 50)
            print(f"Score: {lineup.gpp_score:.1f}")
            print(f"25th %ile: {lineup.percentile_25:.1f}")
            print(f"50th %ile: {lineup.percentile_50:.1f}")
            print(f"75th %ile: {lineup.percentile_75:.1f}")
            print(f"95th %ile: {lineup.percentile_95:.1f}")
            print(f"99th %ile: {lineup.percentile_99:.1f}")
            print(f"Leverage: {lineup.leverage_score:.1f}")
            if not (hasattr(self, 'cash_game_mode') and self.cash_game_mode):
                print(f"Uniqueness: {lineup.uniqueness_score:.2%}")

            salary_remaining = self.constraints.salary_cap - lineup.total_salary
            print(f"\n💰 Salary: ${lineup.total_salary:,} (${salary_remaining:,} left)")
            print(f"👥 Ownership: {lineup.total_ownership:.1f}%")

            self._display_lineup_players(lineup)

    @abstractmethod
    def _display_lineup_players(self, lineup: BaseLineup):
        """Display lineup players - sport-specific formatting."""
        pass
