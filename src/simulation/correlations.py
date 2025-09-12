"""Correlation matrix builder for DFS simulation."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..models import Player


@dataclass 
class CorrelationRule:
    """Simple correlation rule between two players."""
    player1_id: int
    player2_id: int
    correlation: float
    rule_type: str


class BaseCorrelationBuilder(ABC):
    """Base class for correlation matrix builders."""
    
    def __init__(self, players: List[Player]):
        self.players = players
    
    @abstractmethod
    def build_matrix(self) -> np.ndarray:
        """Build correlation matrix."""
        pass
    
    @abstractmethod 
    def get_rules(self) -> List[CorrelationRule]:
        """Get correlation rules."""
        pass


class MMACorrelationBuilder(BaseCorrelationBuilder):
    """Enhanced MMA correlation builder with ITD and ownership insights.
    
    Rules:
    1. Direct opponents: Strong negative correlation (-0.85)
    2. Similar odds favorites: Weak positive correlation (+0.15) 
    3. Dog vs chalk: Weak negative correlation (-0.10)
    4. ITD fighters: Positive correlation with volatility (+0.20)
    """
    
    def build_matrix(self) -> np.ndarray:
        """Build enhanced MMA correlation matrix."""
        n = len(self.players)
        matrix = np.eye(n)  # Identity matrix
        
        # 1. Set opponent correlations (strongest relationship)
        for i, player1 in enumerate(self.players):
            opponent_name = getattr(player1, 'opponent', '').upper()
            if not opponent_name:
                continue
                
            # Find opponent (try exact match first, then partial match)
            for j, player2 in enumerate(self.players):
                if i != j:
                    player2_name = player2.name.upper()
                    # Try exact match, last name match, or if opponent name is in player name
                    if (opponent_name == player2_name or 
                        opponent_name == player2_name.split()[-1] or
                        opponent_name in player2_name):
                        matrix[i][j] = -0.85  # Strong negative
                        matrix[j][i] = -0.85
                        break
        
        # 2. Favorites correlation (similar ML favorites often win together)
        for i, player1 in enumerate(self.players):
            ml_odds1 = getattr(player1, 'ml_odds', 0)
            ownership1 = player1.ownership
            
            if ml_odds1 < -150:  # Strong favorite
                for j, player2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                        continue
                        
                    ml_odds2 = getattr(player2, 'ml_odds', 0)
                    
                    # Similar favorites have slight positive correlation
                    if ml_odds2 < -150 and abs(ml_odds1 - ml_odds2) < 100:
                        matrix[i][j] = 0.15
                        matrix[j][i] = 0.15
        
        # 3. Ownership anti-correlation (chalk vs contrarian)
        for i, player1 in enumerate(self.players):
            ownership1 = player1.ownership
            
            for j, player2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                    continue
                
                ownership2 = player2.ownership
                
                # High ownership vs low ownership: negative correlation
                if ownership1 > 25 and ownership2 < 10:
                    matrix[i][j] = -0.10
                    matrix[j][i] = -0.10
                elif ownership1 < 10 and ownership2 > 25:
                    matrix[i][j] = -0.10
                    matrix[j][i] = -0.10
        
        # 4. ITD/finishing correlation (high variance fighters)
        for i, player1 in enumerate(self.players):
            itd_prob1 = getattr(player1, 'itd_probability', 0)
            
            if itd_prob1 > 0.4:  # High finishing potential
                for j, player2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:
                        continue
                        
                    itd_prob2 = getattr(player2, 'itd_probability', 0)
                    
                    # High ITD fighters create volatility correlation
                    if itd_prob2 > 0.4:
                        matrix[i][j] = 0.20
                        matrix[j][i] = 0.20
        
        return matrix
    
    def get_rules(self) -> List[CorrelationRule]:
        """Get MMA correlation rules."""
        rules = []
        processed = set()
        
        for player1 in self.players:
            if player1.player_id in processed:
                continue
                
            opponent_name = getattr(player1, 'opponent', '').upper()
            if not opponent_name:
                continue
            
            # Find opponent
            for player2 in self.players:
                if player2.player_id not in processed:
                    player2_name = player2.name.upper()
                    # Try exact match, last name match, or if opponent name is in player name
                    if (opponent_name == player2_name or 
                        opponent_name == player2_name.split()[-1] or
                        opponent_name in player2_name):
                        
                        rules.append(CorrelationRule(
                            player1_id=player1.player_id,
                            player2_id=player2.player_id,
                            correlation=-0.8,
                            rule_type='opponent'
                        ))
                        
                        processed.add(player1.player_id)
                        processed.add(player2.player_id)
                        break
        
        return rules


def build_correlation_matrix(sport: str, players: List[Player]) -> Tuple[np.ndarray, List[CorrelationRule]]:
    """Build correlation matrix for any sport."""
    if sport.lower() == 'mma':
        builder = MMACorrelationBuilder(players)
        matrix = builder.build_matrix()
        rules = builder.get_rules()
        return matrix, rules
    else:
        # Default: no correlations
        n = len(players)
        return np.eye(n), []