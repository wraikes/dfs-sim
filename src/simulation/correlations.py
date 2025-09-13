"""Correlation matrix builder for DFS simulation."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..models.player import Player


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


class NASCARCorrelationBuilder(BaseCorrelationBuilder):
    """NASCAR correlation builder with manufacturer and track-specific rules.

    Rules:
    1. Manufacturer correlations: Same manufacturer drivers correlate (+0.15 to +0.25)
    2. Starting position correlations: Nearby starters correlate (+0.10)
    3. Team correlations: Teammates have higher correlation (+0.30)
    4. Track type adjustments: Superspeedways increase all correlations
    5. Dominator vs field: Front runners negatively correlate with back markers
    """

    def build_matrix(self) -> np.ndarray:
        """Build NASCAR correlation matrix."""
        n = len(self.players)
        matrix = np.eye(n)  # Identity matrix

        # Detect track type
        track_type = 'intermediate'  # Default
        for player in self.players:
            if hasattr(player, 'metadata') and player.metadata.get('track_type'):
                track_type = player.metadata['track_type'].lower()
                break

        # Track type multiplier
        track_multiplier = 1.2 if track_type == 'superspeedway' else 1.0

        # 1. Manufacturer correlations (teammates and same brand)
        for i, driver1 in enumerate(self.players):
            mfg1 = driver1.metadata.get('manufacturer', '') if hasattr(driver1, 'metadata') else ''
            team1 = driver1.team if driver1.team else ''

            for j, driver2 in enumerate(self.players):
                if i >= j:
                    continue

                mfg2 = driver2.metadata.get('manufacturer', '') if hasattr(driver2, 'metadata') else ''
                team2 = driver2.team if driver2.team else ''

                # Same team: strong positive correlation
                if team1 and team1 == team2:
                    matrix[i][j] = 0.30 * track_multiplier
                    matrix[j][i] = 0.30 * track_multiplier
                # Same manufacturer: moderate positive correlation
                elif mfg1 and mfg1 == mfg2 and mfg1 in ['Chevrolet', 'Ford', 'Toyota']:
                    matrix[i][j] = 0.15 * track_multiplier
                    matrix[j][i] = 0.15 * track_multiplier

        # 2. Starting position correlations
        for i, driver1 in enumerate(self.players):
            start1 = driver1.metadata.get('starting_position', 99) if hasattr(driver1, 'metadata') else 99

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                    continue

                start2 = driver2.metadata.get('starting_position', 99) if hasattr(driver2, 'metadata') else 99

                # Drivers starting near each other correlate (pack racing)
                if abs(start1 - start2) <= 3:
                    matrix[i][j] = 0.10 * track_multiplier
                    matrix[j][i] = 0.10 * track_multiplier

        # 3. Dominator vs field correlations (front vs back)
        for i, driver1 in enumerate(self.players):
            start1 = driver1.metadata.get('starting_position', 99) if hasattr(driver1, 'metadata') else 99

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                    continue

                start2 = driver2.metadata.get('starting_position', 99) if hasattr(driver2, 'metadata') else 99

                # Front runners vs back markers: negative correlation (can't both dominate)
                if start1 <= 5 and start2 >= 25:
                    matrix[i][j] = -0.15
                    matrix[j][i] = -0.15
                elif start2 <= 5 and start1 >= 25:
                    matrix[i][j] = -0.15
                    matrix[j][i] = -0.15

        # 4. Superspeedway-specific: increase pack racing correlations
        if track_type == 'superspeedway':
            # All drivers have some correlation (pack racing)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(matrix[i][j]) < 0.05:  # If not already correlated
                        matrix[i][j] = 0.05
                        matrix[j][i] = 0.05

        # 5. Road course-specific: skill-based correlations
        if track_type == 'road':
            for i, driver1 in enumerate(self.players):
                track_history1 = driver1.metadata.get('track_avg_finish', 20) if hasattr(driver1, 'metadata') else 20

                for j, driver2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:
                        continue

                    track_history2 = driver2.metadata.get('track_avg_finish', 20) if hasattr(driver2, 'metadata') else 20

                    # Road course specialists correlate
                    if track_history1 < 10 and track_history2 < 10:
                        matrix[i][j] = 0.25
                        matrix[j][i] = 0.25

        # 🔥 ENHANCED CORRELATIONS USING NEW FEATURES 🔥

        # 6. Practice Performance Correlations (using practice_lap_time)
        for i, driver1 in enumerate(self.players):
            practice_time1 = driver1.metadata.get('practice_lap_time', 16.0) if hasattr(driver1, 'metadata') else 16.0

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.15:  # Skip if already strongly correlated
                    continue

                practice_time2 = driver2.metadata.get('practice_lap_time', 16.0) if hasattr(driver2, 'metadata') else 16.0

                # Similar practice times = similar car performance/setup
                time_diff = abs(practice_time1 - practice_time2)
                if time_diff < 0.05:  # Within 0.05 seconds - very similar cars
                    matrix[i][j] = max(matrix[i][j], 0.18 * track_multiplier)
                    matrix[j][i] = max(matrix[j][i], 0.18 * track_multiplier)
                elif time_diff < 0.1:  # Within 0.1 seconds - similar pace
                    matrix[i][j] = max(matrix[i][j], 0.10 * track_multiplier)
                    matrix[j][i] = max(matrix[j][i], 0.10 * track_multiplier)

        # 7. Position Advancement Pattern Correlations (using avg_pass_diff)
        for i, driver1 in enumerate(self.players):
            pass_diff1 = driver1.metadata.get('avg_pass_diff', 0) if hasattr(driver1, 'metadata') else 0

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.15:  # Skip if already strongly correlated
                    continue

                pass_diff2 = driver2.metadata.get('avg_pass_diff', 0) if hasattr(driver2, 'metadata') else 0

                # Similar advancement patterns correlate (both gainers or both losers)
                if abs(pass_diff1 - pass_diff2) < 3.0:  # Similar pass differential patterns
                    correlation_strength = 0.12 if abs(pass_diff1) > 2 else 0.08  # Stronger for aggressive drivers
                    matrix[i][j] = max(matrix[i][j], correlation_strength * track_multiplier)
                    matrix[j][i] = max(matrix[j][i], correlation_strength * track_multiplier)

        # 8. Track Specialist Correlations (using bristol_avg_finish vs season avg)
        for i, driver1 in enumerate(self.players):
            bristol_avg1 = driver1.metadata.get('bristol_avg_finish', 20.0) if hasattr(driver1, 'metadata') else 20.0
            season_avg1 = driver1.metadata.get('avg_finish', 20.0) if hasattr(driver1, 'metadata') else 20.0
            track_advantage1 = season_avg1 - bristol_avg1  # Positive = better at this track

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.15:  # Skip if already strongly correlated
                    continue

                bristol_avg2 = driver2.metadata.get('bristol_avg_finish', 20.0) if hasattr(driver2, 'metadata') else 20.0
                season_avg2 = driver2.metadata.get('avg_finish', 20.0) if hasattr(driver2, 'metadata') else 20.0
                track_advantage2 = season_avg2 - bristol_avg2

                # Similar track advantage/disadvantage patterns correlate
                if abs(track_advantage1 - track_advantage2) < 2.0:  # Similar track performance
                    correlation_strength = 0.15 if track_advantage1 > 3 else 0.10  # Stronger for track specialists
                    matrix[i][j] = max(matrix[i][j], correlation_strength)
                    matrix[j][i] = max(matrix[j][i], correlation_strength)

        # 9. Consistency vs Volatility Anti-Correlation (using quality_passes_per_race)
        for i, driver1 in enumerate(self.players):
            quality_passes1 = driver1.metadata.get('quality_passes_per_race', 50) if hasattr(driver1, 'metadata') else 50

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.10:  # Skip if already correlated
                    continue

                quality_passes2 = driver2.metadata.get('quality_passes_per_race', 50) if hasattr(driver2, 'metadata') else 50

                # Consistent drivers (high quality passes) vs inconsistent drivers: negative correlation
                if quality_passes1 > 65 and quality_passes2 < 35:  # High vs low quality passes
                    matrix[i][j] = min(matrix[i][j], -0.06)  # Mild negative correlation
                    matrix[j][i] = min(matrix[j][i], -0.06)
                elif quality_passes1 < 35 and quality_passes2 > 65:
                    matrix[i][j] = min(matrix[i][j], -0.06)
                    matrix[j][i] = min(matrix[j][i], -0.06)

        return matrix

    def get_rules(self) -> List[CorrelationRule]:
        """Get NASCAR correlation rules."""
        rules = []

        # Track manufacturer correlations
        for i, driver1 in enumerate(self.players):
            mfg1 = driver1.metadata.get('manufacturer', '') if hasattr(driver1, 'metadata') else ''

            for j, driver2 in enumerate(self.players):
                if i >= j:
                    continue

                mfg2 = driver2.metadata.get('manufacturer', '') if hasattr(driver2, 'metadata') else ''

                if mfg1 and mfg1 == mfg2:
                    rules.append(CorrelationRule(
                        player1_id=driver1.player_id,
                        player2_id=driver2.player_id,
                        correlation=0.15,
                        rule_type='manufacturer'
                    ))

        return rules


class MMACorrelationBuilder(BaseCorrelationBuilder):
    """Enhanced MMA correlation builder with ITD and ownership insights.
    
    Rules:
    1. Direct opponents: Strong negative correlation (-0.95)
    2. Similar odds favorites: Weak positive correlation (+0.15) 
    3. Dog vs chalk: Weak negative correlation (-0.10)
    4. ITD fighters: Positive correlation with volatility (+0.20)
    5. Fighting styles: Similar styles correlate (wrestlers, KO artists, grapplers)
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
                        matrix[i][j] = -0.95  # Nearly exclusive (can't both win)
                        matrix[j][i] = -0.95
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
            itd_prob1 = player1.metadata.get('itd_probability', 0.35)
            
            if itd_prob1 > 0.4:  # High finishing potential
                for j, player2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:
                        continue
                        
                    itd_prob2 = player2.metadata.get('itd_probability', 0.35)
                    
                    # High ITD fighters create volatility correlation
                    if itd_prob2 > 0.4:
                        matrix[i][j] = 0.20
                        matrix[j][i] = 0.20
        
        # 5. Fighting style correlations
        for i, player1 in enumerate(self.players):
            metadata1 = player1.metadata or {}
            itd_prob1 = metadata1.get('itd_probability', 0.35)
            ml_odds1 = metadata1.get('ml_odds', 0)
            
            # Categorize fighting style based on stats
            if itd_prob1 > 0.6:
                style1 = 'finisher'  # KO/Sub artists
            elif itd_prob1 < 0.25:
                style1 = 'decision'  # Point fighters/wrestlers
            else:
                style1 = 'balanced'
            
            for j, player2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:
                    continue
                
                metadata2 = player2.metadata or {}
                itd_prob2 = metadata2.get('itd_probability', 0.35)
                ml_odds2 = metadata2.get('ml_odds', 0)
                
                # Categorize second fighter
                if itd_prob2 > 0.6:
                    style2 = 'finisher'
                elif itd_prob2 < 0.25:
                    style2 = 'decision'
                else:
                    style2 = 'balanced'
                
                # Apply style-based correlations
                if style1 == 'finisher' and style2 == 'finisher':
                    # KO artists tend to correlate (exciting cards)
                    matrix[i][j] = 0.30
                    matrix[j][i] = 0.30
                elif style1 == 'decision' and style2 == 'decision':
                    # Grinders correlate (boring cards)
                    matrix[i][j] = 0.25
                    matrix[j][i] = 0.25
                elif (style1 == 'finisher' and style2 == 'decision') or \
                     (style1 == 'decision' and style2 == 'finisher'):
                    # Opposite styles have slight negative correlation
                    matrix[i][j] = -0.05
                    matrix[j][i] = -0.05
        
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
    elif sport.lower() == 'nascar':
        builder = NASCARCorrelationBuilder(players)
        matrix = builder.build_matrix()
        rules = builder.get_rules()
        return matrix, rules
    else:
        # Default: no correlations
        n = len(players)
        return np.eye(n), []