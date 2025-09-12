"""MMA DFS lineup optimizer using base architecture."""

import numpy as np
import pandas as pd
from typing import List, Optional
import random
from dataclasses import dataclass

from ..models.player import Player, Position
from .base_optimizer import BaseOptimizer, SportConstraints, BaseLineup
from .field_generator import MMAFieldGenerator, BaseFieldGenerator


@dataclass
class MMALineup(BaseLineup):
    """MMA-specific lineup with additional metrics."""
    # MMA-specific flags
    has_main_event_pair: bool = False
    num_prelim_fighters: int = 0
    num_leverage_plays: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        
        # Calculate MMA-specific metrics
        self.num_leverage_plays = sum(1 for p in self.players if p.ownership < 10)
        
        # Check for main event pair
        sorted_by_salary = sorted(self.players, key=lambda x: x.salary, reverse=True)
        if len(sorted_by_salary) >= 2:
            top_2_salaries = [p.salary for p in sorted_by_salary[:2]]
            if all(s >= 9000 for s in top_2_salaries):
                fighter1, fighter2 = sorted_by_salary[0], sorted_by_salary[1]
                if (fighter1.opponent == fighter2.name or 
                    fighter2.opponent == fighter1.name):
                    self.has_main_event_pair = True
        
        # Count prelim fighters
        self.num_prelim_fighters = sum(1 for p in self.players if p.salary <= 7500)


class MMAOptimizer(BaseOptimizer):
    """MMA-specific DFS optimizer."""
    
    def _get_sport_constraints(self) -> SportConstraints:
        """Get MMA-specific constraints."""
        return SportConstraints(
            salary_cap=50000,
            roster_size=6,
            max_salary_remaining=1200,
            min_salary_remaining=300,
            max_lineup_ownership=140.0,
            min_leverage_plays=2,
            max_lineup_overlap=0.33,
            sport_rules={
                'max_players_high_owned': 2,
                'min_players_very_low_owned': 1,
                'no_opponents': True,
            }
        )
    
    def _create_field_generator(self) -> BaseFieldGenerator:
        """Create MMA field generator."""
        return MMAFieldGenerator(self.players)
    
    def _has_opponents(self, players: List[Player]) -> bool:
        """Check if any players in the list are opponents of each other."""
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players):
                if i >= j:
                    continue
                if p1.opponent and (p1.opponent.upper() == p2.name.upper() or
                                   p1.opponent.upper() in p2.name.upper()):
                    return True
                if p2.opponent and p2.opponent.upper() == p1.name.upper():
                    return True
        return False
    
    def _validate_lineup(self, players: List[Player]) -> bool:
        """Validate MMA lineup meets sport-specific rules."""
        if len(players) != 6:
            return False
        
        # Check salary constraints
        total_salary = sum(p.salary for p in players)
        if total_salary > 50000 or total_salary < 48800:
            return False
        
        # Check no opponents rule
        if self._has_opponents(players):
            return False
        
        # Check ownership constraints
        total_ownership = sum(p.ownership for p in players)
        if total_ownership > 140.0:
            return False
        
        high_owned = sum(1 for p in players if p.ownership >= 25)
        if high_owned > 2:
            return False
        
        leverage_plays = sum(1 for p in players if p.ownership < 15)
        if leverage_plays < 2:
            return False
        
        return True
    
    def _create_lineup(self, players: List[Player]) -> BaseLineup:
        """Create MMA lineup object."""
        return MMALineup(players=players)
    
    def _create_player_from_row(self, row: pd.Series) -> Player:
        """Create Player object from CSV row."""
        player = Player(
            player_id=int(row['player_id']),
            name=row['name'],
            position=Position.FIGHTER,
            team=row.get('team', ''),
            salary=int(row['salary']),
            projection=float(row['updated_projection']),
            floor=float(row['updated_floor']),
            ceiling=float(row['updated_ceiling']),
            std_dev=float(row.get('std_dev', 25.0)),
            ownership=float(row['updated_ownership']),
            value=float(row.get('value', 0.0)),
            opponent=row.get('opponent', None)
        )
        
        # Add MMA-specific metadata
        player.metadata = {
            'ml_odds': float(row.get('ml_odds', 0)),
            'itd_probability': float(row.get('itd_probability', 0.35)),
            'itd_adjusted_ceiling': float(row.get('itd_adjusted_ceiling', player.ceiling)),
            'newsletter_signal': row.get('newsletter_signal', 'neutral'),
            'newsletter_confidence': float(row.get('newsletter_confidence', 0.5)),
            'base_ownership': float(row.get('original_ownership', player.ownership))
        }
        
        return player
    
    def _generate_single_lineup(self) -> Optional[BaseLineup]:
        """Generate a single MMA lineup candidate."""
        max_attempts = 1000
        
        # Categorize players
        chalk = [p for p in self.players if p.ownership >= 25]
        moderate = [p for p in self.players if 10 <= p.ownership < 25]
        leverage = [p for p in self.players if p.ownership < 10]
        
        for attempt in range(max_attempts):
            players = []
            used_ids = set()
            
            # Strategy selection
            strategy = random.choice(['leverage', 'balanced', 'ceiling'])
            
            if strategy == 'leverage':
                # Heavy leverage focus
                target_pools = [
                    (leverage, 3, 4),
                    (moderate, 1, 2),
                    (chalk, 0, 1)
                ]
            elif strategy == 'ceiling':
                # High ceiling regardless of ownership
                high_ceiling = sorted(self.players, key=lambda x: x.ceiling, reverse=True)[:15]
                target_pools = [
                    (high_ceiling[:8], 3, 4),
                    (high_ceiling[8:], 2, 3)
                ]
            else:  # balanced
                target_pools = [
                    (chalk, 1, 2),
                    (moderate, 2, 3),
                    (leverage, 2, 3)
                ]
            
            # Build lineup
            for pool, min_picks, max_picks in target_pools:
                if len(players) >= 6:
                    break
                
                available = [p for p in pool if p.player_id not in used_ids]
                if not available:
                    continue
                
                num_picks = min(random.randint(min_picks, max_picks), 
                               min(len(available), 6 - len(players)))
                
                for _ in range(num_picks):
                    if not available or len(players) >= 6:
                        break
                    
                    # Weight by ceiling for selection
                    weights = [p.ceiling for p in available]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probs = [w/total_weight for w in weights]
                        fighter = np.random.choice(available, p=probs)
                    else:
                        fighter = random.choice(available)
                    
                    # Check for opponents before adding
                    if not self._has_opponents(players + [fighter]):
                        players.append(fighter)
                        used_ids.add(fighter.player_id)
                        available.remove(fighter)
            
            # Fill remaining spots
            while len(players) < 6:
                available = [p for p in self.players if p.player_id not in used_ids]
                if not available:
                    break
                
                fighter = random.choice(available)
                
                # Check for opponents
                valid = True
                for existing in players:
                    if (fighter.opponent and fighter.opponent.upper() == existing.name.upper()) or \
                       (existing.opponent and existing.opponent.upper() == fighter.name.upper()):
                        valid = False
                        break
                
                if valid:
                    players.append(fighter)
                    used_ids.add(fighter.player_id)
            
            # Validate final lineup
            if len(players) == 6 and self._validate_lineup(players):
                return self._create_lineup(players)
        
        return None
    
    def _score_lineup_gpp(self, lineup: BaseLineup) -> float:
        """Score MMA lineup for GPP success."""
        # Run simulation if needed
        if lineup.simulated_scores is None:
            scores = self.simulator._simulate_correlated(lineup.players)
            lineup.simulated_scores = scores
            lineup.percentile_95 = np.percentile(scores, 95)
            lineup.percentile_99 = np.percentile(scores, 99)
        
        # 1. Ceiling score (40% weight)
        ceiling_score = lineup.percentile_95 * 0.3 + lineup.percentile_99 * 0.1
        
        # 2. Leverage score (25% weight)  
        lineup.leverage_score = self._calculate_leverage_score(lineup)
        leverage_score = lineup.leverage_score * 0.25
        
        # 3. Uniqueness vs field (20% weight)
        lineup.uniqueness_score = self._calculate_lineup_uniqueness(lineup)
        uniqueness_score = lineup.uniqueness_score * 100 * 0.20
        
        # 4. Diversity vs our lineups (10% weight)
        diversity_score = self._calculate_lineup_diversity(lineup) * 100 * 0.10
        
        # 5. Ownership penalty (5% weight)
        ownership_penalty = max(0, lineup.total_ownership - 100) * 0.5
        
        # Calculate base GPP score
        gpp_score = ceiling_score + leverage_score + uniqueness_score + diversity_score - ownership_penalty
        
        # MMA-specific bonuses
        if isinstance(lineup, MMALineup):
            if lineup.num_prelim_fighters >= 2:
                gpp_score += 5  # Prelim exposure bonus
            
            if lineup.num_leverage_plays >= 3:
                gpp_score += 10  # Extreme leverage bonus
            
            if lineup.has_main_event_pair:
                gpp_score -= 15  # Penalty for main event chalk
        
        return gpp_score
    
    def _calculate_leverage_score(self, lineup: BaseLineup) -> float:
        """Calculate leverage score for MMA lineup."""
        leverage = 0
        for player in lineup.players:
            if player.ownership < 15:
                ceiling_upside = player.ceiling - player.projection
                ownership_factor = (100 - player.ownership) / 100
                
                if player.ownership < 5:
                    # Extreme leverage plays get bonus
                    leverage += ceiling_upside * ownership_factor * 3
                else:
                    leverage += ceiling_upside * ownership_factor * 2
        
        return leverage
    
    def _display_lineup_players(self, lineup: BaseLineup):
        """Display MMA lineup players."""
        print(f"\n👊 Fighters:")
        for player in lineup.players:
            signal = player.metadata.get('newsletter_signal', 'neutral')
            signal_icon = {'target': '🎯', 'avoid': '⛔', 'volatile': '⚡'}.get(signal, '  ')
            
            ml_odds = player.metadata.get('ml_odds', 0)
            itd_prob = player.metadata.get('itd_probability', 0.35)
            
            print(f"  {signal_icon} {player.name:18} ${player.salary:5,} | "
                  f"{player.projection:5.1f}pts | {player.ownership:4.1f}% | "
                  f"ITD:{itd_prob:.2f} | ML:{ml_odds:+4.0f}")
        
        # Display MMA-specific stats
        if isinstance(lineup, MMALineup):
            print(f"\n📊 MMA Stats:")
            print(f"  Leverage plays: {lineup.num_leverage_plays}")
            print(f"  Prelim fighters: {lineup.num_prelim_fighters}")
            print(f"  Main event pair: {lineup.has_main_event_pair}")