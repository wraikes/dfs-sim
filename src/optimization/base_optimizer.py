"""Base optimizer class for sport-agnostic DFS optimization."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..models.player import Player, Position
from ..simulation.correlations import build_correlation_matrix
from ..simulation.simulator import Simulator
from .field_generator import BaseFieldGenerator


@dataclass
class SportConstraints:
    """Sport-specific constraints and configuration."""
    salary_cap: int
    roster_size: int
    max_salary_remaining: int = 1200
    min_salary_remaining: int = 300
    max_lineup_ownership: float = 140.0
    min_leverage_plays: int = 2
    max_lineup_overlap: float = 0.33
    
    # Sport-specific rules
    sport_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sport_rules is None:
            self.sport_rules = {}


@dataclass
class BaseLineup:
    """Base lineup class that all sports inherit from."""
    players: List[Player]
    total_salary: int = 0
    total_projection: float = 0
    total_ownership: float = 0
    total_ceiling: float = 0
    
    # GPP metrics
    leverage_score: float = 0
    uniqueness_score: float = 0
    gpp_score: float = 0
    
    # Simulation results
    simulated_scores: Optional[np.ndarray] = None
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
    
    def __init__(self, players: List[Player], sport: str, field_size: int = 10000):
        self.players = players
        self.sport = sport.lower()
        self.field_size = field_size
        self.cash_game_mode = False  # Default to GPP mode
        
        # Get sport-specific constraints
        self.constraints = self._get_sport_constraints()
        
        # Build correlation matrix
        self.correlation_matrix, self.correlation_rules = build_correlation_matrix(
            self.sport, players
        )
        
        # Initialize simulator
        self.simulator = Simulator(
            n_simulations=10000,
            correlation_matrix=self.correlation_matrix
        )
        
        # Initialize field generator
        self.field_generator = self._create_field_generator()
        self.field = []
        
        # Track generated lineups
        self.generated_lineups: List[BaseLineup] = []
    
    @abstractmethod
    def _get_sport_constraints(self) -> SportConstraints:
        """Get sport-specific constraints."""
        pass
    
    @abstractmethod
    def _create_field_generator(self) -> BaseFieldGenerator:
        """Create sport-specific field generator."""
        pass
    
    @abstractmethod
    def _validate_lineup(self, players: List[Player]) -> bool:
        """Validate lineup meets sport-specific rules."""
        pass
    
    @abstractmethod
    def _create_lineup(self, players: List[Player]) -> BaseLineup:
        """Create sport-specific lineup object."""
        pass
    
    def _calculate_value_adjusted_leverage(self, player: Player,
                                           sport_value_metrics: Dict[str, float] = None) -> float:
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
        efficiency_thresholds = {
            'nfl': 3.0,    # NFL has lower pts/$
            'nba': 5.0,    # NBA is middle
            'mlb': 3.5,    # MLB similar to NFL
            'nascar': 5.0, # NASCAR similar to NBA
            'mma': 12.0,   # MMA has highest pts/$
            'pga': 7.0     # PGA in between
        }
        efficiency_cap = efficiency_thresholds.get(self.sport, 5.0)

        # Base value score components
        efficiency_score = min(pts_per_dollar / efficiency_cap, 1.0)
        upside_score = min(ceiling_upside / 2.0, 1.0)

        # Combine base metrics with sport-specific metrics
        if sport_value_metrics:
            # Weight base metrics at 60%, sport-specific at 40%
            base_weight = 0.6
            sport_weight = 0.4

            base_value = (efficiency_score * 0.5 + upside_score * 0.5) * base_weight
            sport_value = sum(sport_value_metrics.values()) / len(sport_value_metrics) * sport_weight
            value_score = base_value + sport_value
        else:
            # No sport-specific metrics, use base only
            value_score = efficiency_score * 0.5 + upside_score * 0.5

        # Only give leverage if player has good value (0.4+ score)
        if value_score >= 0.4:
            # Value-first approach: value is primary, ownership is tiebreaker
            base_value_bonus = value_score * 12  # 0.4-1.0 → 4.8-12.0 bonus

            # Small ownership tiebreaker (doesn't overwhelm value)
            ownership_tiebreaker = max(0, 50 - player.ownership) / 50 * 3  # Max 3 points

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
    
    def load_players_from_csv(self, csv_path: str) -> List[Player]:
        """Load players from processed CSV data."""
        if not pd.io.common.file_exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        players = []
        for _, row in df.iterrows():
            player = self._create_player_from_row(row)
            players.append(player)
        
        return players
    
    def load_players_from_dataframe(self, df: pd.DataFrame) -> List[Player]:
        """Load players from DataFrame."""
        players = []
        for _, row in df.iterrows():
            player = self._create_player_from_row(row)
            players.append(player)
        
        self.players = players
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
    
    def generate_lineup_candidates(self, num_candidates: int) -> List[BaseLineup]:
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
    def _generate_single_lineup(self) -> Optional[BaseLineup]:
        """Generate a single candidate lineup."""
        pass
    
    def optimize_lineups(self, num_lineups: int = 20, use_consensus: bool = False) -> List[BaseLineup]:
        """Main optimization pipeline with optional multi-seed consensus."""
        if use_consensus and num_lineups == 1:
            return self._optimize_with_consensus()
        else:
            return self._optimize_standard(num_lineups)

    def _optimize_standard(self, num_lineups: int) -> List[BaseLineup]:
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

    def _optimize_with_consensus(self) -> List[BaseLineup]:
        """Multi-seed consensus optimization for perfect determinism."""
        import random
        from collections import Counter

        print(f"\n🎯 Generating consensus {self.sport.upper()} lineup using multi-seed approach...")
        print("=" * 80)
        print("Running 25 seeds × 25,000 simulations = 625,000 total simulations")
        print("=" * 80)

        # Store original simulation count and field
        original_n_sims = self.simulator.n_simulations
        original_field = self.field

        # Set parameters for consensus runs
        self.simulator.n_simulations = 25000
        consensus_seeds = list(range(1, 26))  # Seeds 1-25
        all_lineups = []

        for i, seed in enumerate(consensus_seeds, 1):
            print(f"🎲 Seed {seed:2d}/25 ({i*4:3d}% complete)...")

            # Set seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)

            # Reset field for this seed
            self.field = []

            # Run single optimization
            candidates = self.generate_lineup_candidates(200)

            # Score candidates
            for lineup in candidates:
                lineup.gpp_score = self._score_lineup_gpp(lineup)

            # Get best lineup using deterministic selection
            best_candidates = self._apply_deterministic_selection(candidates, 1)
            if best_candidates:
                best_lineup = best_candidates[0]
                lineup_signature = tuple(sorted(p.player_id for p in best_lineup.players))
                all_lineups.append((lineup_signature, best_lineup))

        # Restore original settings
        self.simulator.n_simulations = original_n_sims
        self.field = original_field

        # Find consensus lineup
        lineup_counts = Counter(sig for sig, _ in all_lineups)
        most_common_sig, frequency = lineup_counts.most_common(1)[0]

        print(f"\n🏆 CONSENSUS RESULTS")
        print("=" * 50)
        print(f"Most common lineup appeared: {frequency}/25 times ({frequency/25*100:.1f}%)")

        # Get the actual lineup object for the consensus signature
        consensus_lineup = None
        for sig, lineup in all_lineups:
            if sig == most_common_sig:
                consensus_lineup = lineup
                break

        if consensus_lineup:
            self.generated_lineups.append(consensus_lineup)
            print(f"✅ Selected consensus lineup with {frequency}/25 seed agreement")
            return [consensus_lineup]
        else:
            print("❌ Error: Could not find consensus lineup")
            return []

    def _apply_deterministic_selection(self, candidates: List[BaseLineup],
                                       target_lineups: int) -> List[BaseLineup]:
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
    
    def display_lineup_stats(self, lineups: List[BaseLineup]):
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