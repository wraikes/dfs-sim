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
    
    def optimize_lineups(self, num_lineups: int = 20) -> List[BaseLineup]:
        """Main optimization pipeline."""
        print(f"\n🚀 Generating {num_lineups} {self.sport.upper()} GPP lineups...")
        print("=" * 60)
        
        # Generate field if not done
        if not self.field:
            self.generate_field()
        
        # Generate candidates
        candidates = self.generate_lineup_candidates(num_lineups * 5)
        print(f"📊 Generated {len(candidates)} candidates for scoring...")
        
        # Score all candidates
        for lineup in candidates:
            lineup.gpp_score = self._score_lineup_gpp(lineup)
        
        # Sort by GPP score
        candidates.sort(key=lambda x: x.gpp_score, reverse=True)
        
        # Select diverse lineups
        final_lineups = []
        for lineup in candidates:
            if len(final_lineups) >= num_lineups:
                break
            
            # Check diversity constraint
            if all(lineup.calculate_overlap_with_lineup(existing) <= self.constraints.max_lineup_overlap 
                   for existing in final_lineups):
                final_lineups.append(lineup)
                self.generated_lineups.append(lineup)
        
        print(f"✅ Selected {len(final_lineups)} optimized lineups")
        return final_lineups
    
    def display_lineup_stats(self, lineups: List[BaseLineup]):
        """Display lineup statistics - can be overridden by sport."""
        print(f"\n📈 {self.sport.upper()} LINEUP STATISTICS")
        print("=" * 60)
        
        for i, lineup in enumerate(lineups[:5], 1):
            print(f"\n🏆 LINEUP #{i}")
            print("-" * 50)
            print(f"GPP Score: {lineup.gpp_score:.1f}")
            print(f"95th %ile: {lineup.percentile_95:.1f}")
            print(f"99th %ile: {lineup.percentile_99:.1f}")
            print(f"Leverage: {lineup.leverage_score:.1f}")
            print(f"Uniqueness: {lineup.uniqueness_score:.2%}")
            
            salary_remaining = self.constraints.salary_cap - lineup.total_salary
            print(f"\n💰 Salary: ${lineup.total_salary:,} (${salary_remaining:,} left)")
            print(f"👥 Ownership: {lineup.total_ownership:.1f}%")
            
            self._display_lineup_players(lineup)
    
    @abstractmethod
    def _display_lineup_players(self, lineup: BaseLineup):
        """Display lineup players - sport-specific formatting."""
        pass