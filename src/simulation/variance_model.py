"""Variance modeling for player projections."""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..models.player import Player, Position


class DistributionType(Enum):
    """Types of statistical distributions for simulation."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"


@dataclass
class VarianceModel:
    """Model variance for player projections based on position and context."""
    
    # Position-based variance multipliers (higher = more volatile)
    position_variance: Dict[Position, float] = None
    
    # Distribution types by position
    position_distributions: Dict[Position, DistributionType] = None
    
    # Minimum floor as percentage of projection
    floor_percentage: float = 0.5
    
    # Maximum ceiling as percentage of projection  
    ceiling_percentage: float = 2.0
    
    def __post_init__(self):
        """Set default variance by position if not provided."""
        if self.position_variance is None:
            self.position_variance = {
                # NFL
                Position.QB: 0.20,
                Position.RB: 0.25,
                Position.WR: 0.30,
                Position.TE: 0.35,
                Position.DST: 0.40,
                
                # NBA
                Position.PG: 0.22,
                Position.SG: 0.24,
                Position.SF: 0.23,
                Position.PF: 0.25,
                Position.C: 0.20,
                
                # MLB
                Position.P: 0.35,
                Position.C_MLB: 0.28,
                Position.FB: 0.26,
                Position.SB: 0.27,
                Position.TB: 0.27,
                Position.SS: 0.28,
                Position.OF: 0.26,
                
                # NHL
                Position.C_NHL: 0.25,
                Position.W: 0.27,
                Position.D: 0.30,
                Position.G_NHL: 0.35,
                
                # Single position sports
                Position.GOLFER: 0.40,
                Position.DRIVER: 0.45,
                Position.FIGHTER: 0.70,  # Extreme variance for MMA (KOs, subs, decisions)
            }
        
        if self.position_distributions is None:
            # Most positions use normal distribution
            self.position_distributions = {}
            for pos in Position:
                # High-variance positions use lognormal for long tail
                if pos in [Position.TE, Position.DST, Position.GOLFER, Position.DRIVER, Position.FIGHTER]:
                    self.position_distributions[pos] = DistributionType.LOGNORMAL
                else:
                    self.position_distributions[pos] = DistributionType.NORMAL
    
    def simulate_score(self, player: Player, n_sims: int = 1) -> np.ndarray:
        """
        Simulate player scores based on projection and variance.
        
        Args:
            player: Player to simulate
            n_sims: Number of simulations
            
        Returns:
            Array of simulated scores
        """
        # Get base parameters
        mean = player.adjusted_projection
        
        # Use player's std_dev if available, otherwise calculate from position
        if player.std_dev > 0:
            std_dev = player.adjusted_std_dev
        else:
            variance_mult = self.position_variance.get(player.position, 0.25)
            std_dev = mean * variance_mult
        
        # Sport-specific variance adjustments
        if player.position == Position.FIGHTER:
            # MMA: Apply fight competitiveness variance adjustment
            ml_odds = getattr(player, 'ml_odds', 0)
            opponent_ml_odds = getattr(player, 'opponent_ml_odds', 0)

            if ml_odds != 0 and opponent_ml_odds != 0:
                odds_spread = abs(ml_odds - opponent_ml_odds)
                if odds_spread < 100:  # Close fight
                    competitiveness_mult = 0.9  # Lower variance (more predictable)
                else:  # Mismatch
                    competitiveness_mult = 1.1  # Higher variance (finish-likely)

                std_dev *= competitiveness_mult

        elif player.position == Position.DRIVER:
            # 🔥 ENHANCED NASCAR VARIANCE MODEL USING ALL NEW FEATURES 🔥
            if hasattr(player, 'metadata'):
                track_type = player.metadata.get('track_type', 'intermediate')
                starting_position = player.metadata.get('starting_position', 20)

                # Base track type variance multiplier
                if track_type == 'superspeedway':
                    std_dev *= 1.3  # Much higher variance (big wrecks, pack racing)
                elif track_type == 'road':
                    std_dev *= 0.9  # Lower variance (skill-based)

                # 🔥 NEW: Position Advancement Consistency (using avg_pass_diff)
                avg_pass_diff = player.metadata.get('avg_pass_diff', 0)
                if abs(avg_pass_diff) > 5:  # Highly volatile position changes
                    std_dev *= 1.15  # +15% variance for boom/bust drivers
                elif abs(avg_pass_diff) < 2:  # Very consistent position
                    std_dev *= 0.85  # -15% variance for steady drivers

                # 🔥 NEW: Practice Performance Variance (using practice_lap_time)
                practice_time = player.metadata.get('practice_lap_time', 15.5)
                avg_practice_time = 15.5  # Approximate average for Bristol
                if practice_time > avg_practice_time + 0.2:  # Struggled in practice
                    std_dev *= 1.25  # +25% variance (setup issues, mechanical problems)
                elif practice_time < avg_practice_time - 0.1:  # Fast in practice
                    std_dev *= 0.90  # -10% variance (good setup, more predictable)

                # 🔥 NEW: Track Specialist Adjustment (using bristol_avg_finish vs season avg)
                bristol_avg = player.metadata.get('bristol_avg_finish', 20.0)
                season_avg = player.metadata.get('avg_finish', 20.0)
                track_advantage = season_avg - bristol_avg  # Positive = better at this track
                if track_advantage > 3:  # Track specialist (3+ positions better)
                    std_dev *= 0.80  # -20% variance (more predictable at specialty track)
                elif track_advantage < -3:  # Track struggle (3+ positions worse)
                    std_dev *= 1.30  # +30% variance (unpredictable at weak tracks)

                # 🔥 NEW: Quality/Consistency Factor (using quality_passes_per_race)
                quality_passes = player.metadata.get('quality_passes_per_race', 50)
                if quality_passes > 65:  # High-quality, consistent driver
                    std_dev *= 0.88  # -12% variance
                elif quality_passes < 35:  # Inconsistent, erratic driver
                    std_dev *= 1.18  # +18% variance

                # 🔥 NEW: Speed vs Consistency Balance (using fastest_laps_per_race)
                fastest_laps = player.metadata.get('fastest_laps_per_race', 5)
                if fastest_laps > 10:  # Speed-focused driver (high risk/reward)
                    std_dev *= 1.10  # +10% variance (goes for speed, more crashes)
                elif fastest_laps < 3:  # Conservative driver
                    std_dev *= 0.92  # -8% variance (plays it safe)

                # Original starting position variance (enhanced)
                if starting_position <= 5:
                    std_dev *= 0.90  # Front runners more consistent
                elif starting_position >= 25:
                    std_dev *= 1.20  # Back of pack more volatile
        
        # Get distribution type
        dist_type = self.position_distributions.get(player.position, DistributionType.NORMAL)
        
        # Generate samples based on distribution
        if dist_type == DistributionType.NORMAL:
            scores = np.random.normal(mean, std_dev, n_sims)
            
        elif dist_type == DistributionType.LOGNORMAL:
            # Convert mean/std to lognormal parameters
            variance = std_dev ** 2
            mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
            sigma = np.sqrt(np.log(1 + variance / mean ** 2))
            scores = np.random.lognormal(mu, sigma, n_sims)
            
        else:  # BETA
            # Map to beta distribution (bounded between floor and ceiling)
            floor = player.floor if player.floor > 0 else mean * self.floor_percentage
            ceiling = player.ceiling if player.ceiling > 0 else mean * self.ceiling_percentage
            
            # Calculate alpha and beta parameters
            normalized_mean = (mean - floor) / (ceiling - floor)
            variance_factor = 0.1  # Controls spread
            alpha = normalized_mean * ((1 - normalized_mean) / variance_factor - 1)
            beta_param = alpha * (1 - normalized_mean) / normalized_mean
            
            # Generate and scale
            normalized_scores = np.random.beta(alpha, beta_param, n_sims)
            scores = floor + normalized_scores * (ceiling - floor)
        
        # Apply floor and ceiling constraints
        if player.floor > 0:
            scores = np.maximum(scores, player.floor)
        if player.ceiling > 0:
            scores = np.minimum(scores, player.ceiling)
        
        # Ensure non-negative
        scores = np.maximum(scores, 0)
        
        return scores
    
    def simulate_lineup(self, players: List[Player], n_sims: int = 1) -> np.ndarray:
        """
        Simulate total scores for a lineup without correlation.
        
        Args:
            players: List of players in lineup
            n_sims: Number of simulations
            
        Returns:
            Array of simulated lineup totals
        """
        # Simulate each player independently
        player_scores = np.zeros((len(players), n_sims))
        
        for i, player in enumerate(players):
            player_scores[i] = self.simulate_score(player, n_sims)
        
        # Sum for lineup totals
        return np.sum(player_scores, axis=0)
    
    def add_game_variance(self, scores: np.ndarray, game_total: float, 
                         expected_total: float = 50.0) -> np.ndarray:
        """
        Adjust scores based on game environment (high/low scoring).
        
        Args:
            scores: Base simulated scores
            game_total: Vegas game total
            expected_total: League average game total
            
        Returns:
            Adjusted scores
        """
        # Scale scores based on game environment
        game_multiplier = game_total / expected_total
        
        # Add some randomness to the multiplier
        random_factor = np.random.normal(1, 0.05, len(scores))
        
        return scores * game_multiplier * random_factor
    
    def add_ownership_leverage(self, scores: np.ndarray, ownership: float) -> np.ndarray:
        """
        Add ownership-based variance (low-owned players more volatile).
        
        Args:
            scores: Base simulated scores
            ownership: Ownership percentage (0-100)
            
        Returns:
            Adjusted scores with ownership variance
        """
        # Lower ownership = higher variance
        ownership_variance = 1 + (20 - ownership) / 100  # Max 1.2x at 0% ownership
        
        # Add noise based on ownership
        noise = np.random.normal(1, ownership_variance * 0.1, len(scores))
        
        return scores * noise