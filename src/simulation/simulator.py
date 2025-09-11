"""Monte Carlo simulation engine for DFS lineups."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..models import Player, Lineup, Contest
from .variance_model import VarianceModel

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from lineup simulation."""
    lineup: Lineup
    scores: np.ndarray
    mean_score: float
    std_dev: float
    percentiles: Dict[int, float]
    min_score: float
    max_score: float
    
    def get_percentile(self, p: int) -> float:
        """Get score at specific percentile."""
        return self.percentiles.get(p, np.percentile(self.scores, p))
    
    def get_top_heavy_score(self) -> float:
        """Get average of 95th-99th percentile (GPP target)."""
        top_scores = [self.get_percentile(p) for p in range(95, 100)]
        return np.mean(top_scores)


class Simulator:
    """Main simulation engine for DFS."""
    
    def __init__(self, n_simulations: int = 10000, 
                 variance_model: Optional[VarianceModel] = None,
                 correlation_matrix: Optional[np.ndarray] = None):
        """
        Initialize simulator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            variance_model: Model for player variance
            correlation_matrix: Player correlation matrix
        """
        self.n_simulations = n_simulations
        self.variance_model = variance_model or VarianceModel()
        self.correlation_matrix = correlation_matrix
        
        # Cache for player simulations
        self._player_cache: Dict[str, np.ndarray] = {}
    
    def simulate_player(self, player: Player, use_cache: bool = True) -> np.ndarray:
        """
        Simulate a single player's scores.
        
        Args:
            player: Player to simulate
            use_cache: Whether to use cached simulations
            
        Returns:
            Array of simulated scores
        """
        # Check cache
        if use_cache and player.player_id in self._player_cache:
            return self._player_cache[player.player_id]
        
        # Generate simulations
        scores = self.variance_model.simulate_score(player, self.n_simulations)
        
        # Apply game environment adjustments
        if player.game_total > 0:
            scores = self.variance_model.add_game_variance(
                scores, player.game_total
            )
        
        # Apply ownership leverage for GPP
        if player.ownership > 0:
            scores = self.variance_model.add_ownership_leverage(
                scores, player.ownership
            )
        
        # Cache results
        if use_cache:
            self._player_cache[player.player_id] = scores
        
        return scores
    
    def simulate_lineup(self, lineup: Lineup, 
                       use_correlation: bool = True) -> SimulationResult:
        """
        Run full simulation for a lineup.
        
        Args:
            lineup: Lineup to simulate
            use_correlation: Whether to apply correlation matrix
            
        Returns:
            SimulationResult with scores and statistics
        """
        players = lineup.players
        n_players = len(players)
        
        if use_correlation and self.correlation_matrix is not None:
            scores = self._simulate_correlated(players)
        else:
            scores = self._simulate_independent(players)
        
        # Calculate statistics
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        
        # Calculate key percentiles for GPP
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 97, 99]:
            percentiles[p] = np.percentile(scores, p)
        
        # Store in lineup
        lineup.clear_simulations()
        for score in scores:
            lineup.add_simulated_score(score)
        
        return SimulationResult(
            lineup=lineup,
            scores=scores,
            mean_score=mean_score,
            std_dev=std_dev,
            percentiles=percentiles,
            min_score=np.min(scores),
            max_score=np.max(scores)
        )
    
    def _simulate_independent(self, players: List[Player]) -> np.ndarray:
        """Simulate players independently (no correlation)."""
        lineup_scores = np.zeros(self.n_simulations)
        
        for player in players:
            player_scores = self.simulate_player(player)
            lineup_scores += player_scores
        
        return lineup_scores
    
    def _simulate_correlated(self, players: List[Player]) -> np.ndarray:
        """Simulate players with correlation matrix."""
        n_players = len(players)
        
        # Get player indices in correlation matrix
        player_indices = []
        for player in players:
            # This would need a mapping of player_id to matrix index
            # For now, simulate independently
            pass
        
        # TODO: Implement correlated simulation using Cholesky decomposition
        logger.warning("Correlation matrix provided but not yet implemented")
        return self._simulate_independent(players)
    
    def simulate_contest_placement(self, lineup_score: float, 
                                  field_scores: np.ndarray) -> Tuple[int, float]:
        """
        Determine contest placement given a score and field.
        
        Args:
            lineup_score: Our lineup's score
            field_scores: Array of opponent scores
            
        Returns:
            Tuple of (placement, percentile)
        """
        # Count how many scores we beat
        better_than = np.sum(field_scores < lineup_score)
        total = len(field_scores)
        
        placement = total - better_than + 1
        percentile = (better_than / total) * 100
        
        return placement, percentile
    
    def calculate_lineup_ev(self, result: SimulationResult, 
                           contest: Contest,
                           field_scores: Optional[np.ndarray] = None) -> float:
        """
        Calculate expected value of a lineup in a contest.
        
        Args:
            result: Simulation result for lineup
            contest: Contest to evaluate
            field_scores: Optional field scores for placement
            
        Returns:
            Expected value in dollars
        """
        if field_scores is None:
            # Use percentiles to estimate cash probability
            cash_percentile = 100 - contest.cash_line
            cash_probability = np.sum(
                result.scores >= result.get_percentile(cash_percentile)
            ) / len(result.scores)
            
            # Estimate win probability (top 1%)
            win_probability = np.sum(
                result.scores >= result.get_percentile(99)
            ) / len(result.scores)
        else:
            # Calculate actual placement probabilities
            cash_count = 0
            win_count = 0
            
            for score in result.scores:
                placement, _ = self.simulate_contest_placement(score, field_scores)
                if placement <= contest.places_paid:
                    cash_count += 1
                if placement == 1:
                    win_count += 1
            
            cash_probability = cash_count / len(result.scores)
            win_probability = win_count / len(result.scores)
        
        return contest.expected_value(win_probability, cash_probability)
    
    def find_optimal_lineup_score(self, result: SimulationResult,
                                 target_percentile: int = 97) -> float:
        """
        Find the target score for GPP optimization.
        
        Args:
            result: Simulation result
            target_percentile: Target percentile for GPP (95-99)
            
        Returns:
            Target score to optimize for
        """
        # For GPPs, optimize for 95th-99th percentile average
        if target_percentile >= 95:
            return result.get_top_heavy_score()
        else:
            return result.get_percentile(target_percentile)
    
    def clear_cache(self):
        """Clear the player simulation cache."""
        self._player_cache.clear()
        logger.info("Cleared simulation cache")