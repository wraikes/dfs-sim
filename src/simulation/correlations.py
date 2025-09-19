"""Correlation matrix builder for DFS simulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config.correlation_config import (
    MMA_CORRELATION_CONFIG,
    NASCAR_CORRELATION_CONFIG,
    NFL_CORRELATION_CONFIG,
    CorrelationType,
    FightingStyle,
    TrackType,
)
from ..models.player import Player, Position


@dataclass
class CorrelationRule:
    """Simple correlation rule between two players."""
    player1_id: int
    player2_id: int
    correlation: float
    rule_type: CorrelationType


class BaseCorrelationBuilder(ABC):
    """Base class for correlation matrix builders."""

    def __init__(self, players: list[Player]):
        self.players = players

    @abstractmethod
    def build_matrix(self) -> np.ndarray:
        """Build correlation matrix."""
        pass

    @abstractmethod
    def get_rules(self) -> list[CorrelationRule]:
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
        track_type = TrackType.INTERMEDIATE  # Default
        for player in self.players:
            if hasattr(player, 'metadata') and player.metadata.get('track_type'):
                track_type_str = player.metadata['track_type'].lower()
                try:
                    track_type = TrackType(track_type_str)
                except ValueError:
                    track_type = TrackType.INTERMEDIATE
                break

        # Track type multiplier
        track_multiplier = (NASCAR_CORRELATION_CONFIG.SUPERSPEEDWAY_MULTIPLIER
                          if track_type == TrackType.SUPERSPEEDWAY
                          else NASCAR_CORRELATION_CONFIG.DEFAULT_MULTIPLIER)

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
                    correlation = NASCAR_CORRELATION_CONFIG.SAME_TEAM_CORRELATION * track_multiplier
                    matrix[i][j] = correlation
                    matrix[j][i] = correlation
                # Same manufacturer: moderate positive correlation
                elif (mfg1 and mfg1 == mfg2 and
                      mfg1 in NASCAR_CORRELATION_CONFIG.VALID_MANUFACTURERS):
                    correlation = NASCAR_CORRELATION_CONFIG.SAME_MANUFACTURER_CORRELATION * track_multiplier
                    matrix[i][j] = correlation
                    matrix[j][i] = correlation

        # 2. Starting position correlations
        for i, driver1 in enumerate(self.players):
            start1 = driver1.metadata.get('starting_position', 99) if hasattr(driver1, 'metadata') else 99

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                    continue

                start2 = driver2.metadata.get('starting_position', 99) if hasattr(driver2, 'metadata') else 99

                # Drivers starting near each other correlate (pack racing)
                if abs(start1 - start2) <= NASCAR_CORRELATION_CONFIG.STARTING_POSITION_THRESHOLD:
                    correlation = NASCAR_CORRELATION_CONFIG.STARTING_POSITION_CORRELATION * track_multiplier
                    matrix[i][j] = correlation
                    matrix[j][i] = correlation

        # 3. Dominator vs field correlations (front vs back)
        for i, driver1 in enumerate(self.players):
            start1 = driver1.metadata.get('starting_position', 99) if hasattr(driver1, 'metadata') else 99

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                    continue

                start2 = driver2.metadata.get('starting_position', 99) if hasattr(driver2, 'metadata') else 99

                # Front runners vs back markers: negative correlation (can't both dominate)
                if ((start1 <= NASCAR_CORRELATION_CONFIG.FRONT_RUNNER_THRESHOLD and
                     start2 >= NASCAR_CORRELATION_CONFIG.BACK_MARKER_THRESHOLD) or
                    (start2 <= NASCAR_CORRELATION_CONFIG.FRONT_RUNNER_THRESHOLD and
                     start1 >= NASCAR_CORRELATION_CONFIG.BACK_MARKER_THRESHOLD)):
                    matrix[i][j] = NASCAR_CORRELATION_CONFIG.FRONT_VS_BACK_CORRELATION
                    matrix[j][i] = NASCAR_CORRELATION_CONFIG.FRONT_VS_BACK_CORRELATION

        # 4. Superspeedway-specific: increase pack racing correlations
        if track_type == TrackType.SUPERSPEEDWAY:
            # All drivers have some correlation (pack racing)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(matrix[i][j]) < NASCAR_CORRELATION_CONFIG.PACK_RACING_CORRELATION:
                        matrix[i][j] = NASCAR_CORRELATION_CONFIG.PACK_RACING_CORRELATION
                        matrix[j][i] = NASCAR_CORRELATION_CONFIG.PACK_RACING_CORRELATION

        # 5. Road course-specific: skill-based correlations
        if track_type == TrackType.ROAD:
            for i, driver1 in enumerate(self.players):
                track_history1 = driver1.metadata.get('track_avg_finish', 20) if hasattr(driver1, 'metadata') else 20

                for j, driver2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:
                        continue

                    track_history2 = driver2.metadata.get('track_avg_finish', 20) if hasattr(driver2, 'metadata') else 20

                    # Road course specialists correlate
                    if (track_history1 < NASCAR_CORRELATION_CONFIG.ROAD_SPECIALIST_THRESHOLD and
                        track_history2 < NASCAR_CORRELATION_CONFIG.ROAD_SPECIALIST_THRESHOLD):
                        matrix[i][j] = NASCAR_CORRELATION_CONFIG.ROAD_SPECIALIST_CORRELATION
                        matrix[j][i] = NASCAR_CORRELATION_CONFIG.ROAD_SPECIALIST_CORRELATION

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
                if time_diff < NASCAR_CORRELATION_CONFIG.PRACTICE_VERY_SIMILAR_TIME:
                    correlation = NASCAR_CORRELATION_CONFIG.PRACTICE_VERY_SIMILAR_CORRELATION * track_multiplier
                    matrix[i][j] = max(matrix[i][j], correlation)
                    matrix[j][i] = max(matrix[j][i], correlation)
                elif time_diff < NASCAR_CORRELATION_CONFIG.PRACTICE_SIMILAR_TIME:
                    correlation = NASCAR_CORRELATION_CONFIG.PRACTICE_SIMILAR_CORRELATION * track_multiplier
                    matrix[i][j] = max(matrix[i][j], correlation)
                    matrix[j][i] = max(matrix[j][i], correlation)

        # 7. Position Advancement Pattern Correlations (using avg_pass_diff)
        for i, driver1 in enumerate(self.players):
            pass_diff1 = driver1.metadata.get('avg_pass_diff', 0) if hasattr(driver1, 'metadata') else 0

            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.15:  # Skip if already strongly correlated
                    continue

                pass_diff2 = driver2.metadata.get('avg_pass_diff', 0) if hasattr(driver2, 'metadata') else 0

                # Similar advancement patterns correlate (both gainers or both losers)
                if abs(pass_diff1 - pass_diff2) < NASCAR_CORRELATION_CONFIG.PASS_DIFF_THRESHOLD:
                    correlation_strength = (NASCAR_CORRELATION_CONFIG.AGGRESSIVE_CORRELATION
                                           if abs(pass_diff1) > NASCAR_CORRELATION_CONFIG.AGGRESSIVE_DRIVER_THRESHOLD
                                           else NASCAR_CORRELATION_CONFIG.NORMAL_CORRELATION)
                    correlation = correlation_strength * track_multiplier
                    matrix[i][j] = max(matrix[i][j], correlation)
                    matrix[j][i] = max(matrix[j][i], correlation)

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
                if abs(track_advantage1 - track_advantage2) < NASCAR_CORRELATION_CONFIG.TRACK_ADVANTAGE_THRESHOLD:
                    correlation_strength = (NASCAR_CORRELATION_CONFIG.SPECIALIST_CORRELATION
                                           if track_advantage1 > NASCAR_CORRELATION_CONFIG.SPECIALIST_ADVANTAGE_THRESHOLD
                                           else NASCAR_CORRELATION_CONFIG.NORMAL_TRACK_CORRELATION)
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
                if ((quality_passes1 > NASCAR_CORRELATION_CONFIG.HIGH_QUALITY_PASSES_THRESHOLD and
                     quality_passes2 < NASCAR_CORRELATION_CONFIG.LOW_QUALITY_PASSES_THRESHOLD) or
                    (quality_passes1 < NASCAR_CORRELATION_CONFIG.LOW_QUALITY_PASSES_THRESHOLD and
                     quality_passes2 > NASCAR_CORRELATION_CONFIG.HIGH_QUALITY_PASSES_THRESHOLD)):
                    matrix[i][j] = min(matrix[i][j], NASCAR_CORRELATION_CONFIG.CONSISTENCY_ANTI_CORRELATION)
                    matrix[j][i] = min(matrix[j][i], NASCAR_CORRELATION_CONFIG.CONSISTENCY_ANTI_CORRELATION)

        return matrix

    def get_rules(self) -> list[CorrelationRule]:
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
                        correlation=NASCAR_CORRELATION_CONFIG.SAME_MANUFACTURER_CORRELATION,
                        rule_type=CorrelationType.MANUFACTURER
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
                        matrix[i][j] = MMA_CORRELATION_CONFIG.OPPONENT_CORRELATION
                        matrix[j][i] = MMA_CORRELATION_CONFIG.OPPONENT_CORRELATION
                        break

        # 2. Favorites correlation (similar ML favorites often win together)
        for i, player1 in enumerate(self.players):
            ml_odds1 = getattr(player1, 'ml_odds', 0)
            ownership1 = player1.ownership

            if ml_odds1 < MMA_CORRELATION_CONFIG.STRONG_FAVORITE_THRESHOLD:
                for j, player2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                        continue

                    ml_odds2 = getattr(player2, 'ml_odds', 0)

                    # Similar favorites have slight positive correlation
                    if (ml_odds2 < MMA_CORRELATION_CONFIG.STRONG_FAVORITE_THRESHOLD and
                        abs(ml_odds1 - ml_odds2) < MMA_CORRELATION_CONFIG.FAVORITE_ODDS_DIFF_THRESHOLD):
                        matrix[i][j] = MMA_CORRELATION_CONFIG.FAVORITE_CORRELATION
                        matrix[j][i] = MMA_CORRELATION_CONFIG.FAVORITE_CORRELATION

        # 3. Ownership anti-correlation (chalk vs contrarian)
        for i, player1 in enumerate(self.players):
            ownership1 = player1.ownership

            for j, player2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:  # Skip if already correlated
                    continue

                ownership2 = player2.ownership

                # High ownership vs low ownership: negative correlation
                if ((ownership1 > MMA_CORRELATION_CONFIG.HIGH_OWNERSHIP_THRESHOLD and
                     ownership2 < MMA_CORRELATION_CONFIG.LOW_OWNERSHIP_THRESHOLD) or
                    (ownership1 < MMA_CORRELATION_CONFIG.LOW_OWNERSHIP_THRESHOLD and
                     ownership2 > MMA_CORRELATION_CONFIG.HIGH_OWNERSHIP_THRESHOLD)):
                    matrix[i][j] = MMA_CORRELATION_CONFIG.OWNERSHIP_ANTI_CORRELATION
                    matrix[j][i] = MMA_CORRELATION_CONFIG.OWNERSHIP_ANTI_CORRELATION

        # 4. ITD/finishing correlation (high variance fighters)
        for i, player1 in enumerate(self.players):
            itd_prob1 = player1.metadata.get('itd_probability', MMA_CORRELATION_CONFIG.DEFAULT_ITD_PROBABILITY)

            if itd_prob1 > MMA_CORRELATION_CONFIG.HIGH_ITD_THRESHOLD:
                for j, player2 in enumerate(self.players):
                    if i >= j or abs(matrix[i][j]) > 0.1:
                        continue

                    itd_prob2 = player2.metadata.get('itd_probability', MMA_CORRELATION_CONFIG.DEFAULT_ITD_PROBABILITY)

                    # High ITD fighters create volatility correlation
                    if itd_prob2 > MMA_CORRELATION_CONFIG.HIGH_ITD_THRESHOLD:
                        matrix[i][j] = MMA_CORRELATION_CONFIG.ITD_CORRELATION
                        matrix[j][i] = MMA_CORRELATION_CONFIG.ITD_CORRELATION

        # 5. Fighting style correlations
        for i, player1 in enumerate(self.players):
            metadata1 = player1.metadata or {}
            itd_prob1 = metadata1.get('itd_probability', MMA_CORRELATION_CONFIG.DEFAULT_ITD_PROBABILITY)
            ml_odds1 = metadata1.get('ml_odds', 0)

            # Categorize fighting style based on stats
            if itd_prob1 > MMA_CORRELATION_CONFIG.VERY_HIGH_ITD_THRESHOLD:
                style1 = FightingStyle.FINISHER
            elif itd_prob1 < MMA_CORRELATION_CONFIG.DECISION_FIGHTER_THRESHOLD:
                style1 = FightingStyle.DECISION
            else:
                style1 = FightingStyle.BALANCED

            for j, player2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:
                    continue

                metadata2 = player2.metadata or {}
                itd_prob2 = metadata2.get('itd_probability', MMA_CORRELATION_CONFIG.DEFAULT_ITD_PROBABILITY)
                ml_odds2 = metadata2.get('ml_odds', 0)

                # Categorize second fighter
                if itd_prob2 > MMA_CORRELATION_CONFIG.VERY_HIGH_ITD_THRESHOLD:
                    style2 = FightingStyle.FINISHER
                elif itd_prob2 < MMA_CORRELATION_CONFIG.DECISION_FIGHTER_THRESHOLD:
                    style2 = FightingStyle.DECISION
                else:
                    style2 = FightingStyle.BALANCED

                # Apply style-based correlations
                if style1 == FightingStyle.FINISHER and style2 == FightingStyle.FINISHER:
                    # KO artists tend to correlate (exciting cards)
                    matrix[i][j] = MMA_CORRELATION_CONFIG.FINISHER_CORRELATION
                    matrix[j][i] = MMA_CORRELATION_CONFIG.FINISHER_CORRELATION
                elif style1 == FightingStyle.DECISION and style2 == FightingStyle.DECISION:
                    # Grinders correlate (boring cards)
                    matrix[i][j] = MMA_CORRELATION_CONFIG.DECISION_CORRELATION
                    matrix[j][i] = MMA_CORRELATION_CONFIG.DECISION_CORRELATION
                elif ((style1 == FightingStyle.FINISHER and style2 == FightingStyle.DECISION) or
                      (style1 == FightingStyle.DECISION and style2 == FightingStyle.FINISHER)):
                    # Opposite styles have slight negative correlation
                    matrix[i][j] = MMA_CORRELATION_CONFIG.STYLE_ANTI_CORRELATION
                    matrix[j][i] = MMA_CORRELATION_CONFIG.STYLE_ANTI_CORRELATION

        return matrix

    def get_rules(self) -> list[CorrelationRule]:
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
                            correlation=MMA_CORRELATION_CONFIG.OPPONENT_CORRELATION,
                            rule_type=CorrelationType.OPPONENT
                        ))

                        processed.add(player1.player_id)
                        processed.add(player2.player_id)
                        break

        return rules


class NFLCorrelationBuilder(BaseCorrelationBuilder):
    """NFL correlation builder with stacking and game correlations.

    Rules:
    1. QB-pass catcher stacks: Same team QB + WR/TE/RB correlate (+0.6 to +0.8)
    2. Game stacks: Players in same game correlate (+0.15 to +0.25)
    3. Negative correlations: Opposing DSTs vs offensive players (-0.3)
    4. RB-DST same team: Positive correlation (+0.4)
    5. Weather correlations: Bad weather boosts RB/DST, hurts passing
    """

    def build_matrix(self) -> np.ndarray:
        """Build NFL correlation matrix with stacking emphasis."""
        n = len(self.players)
        if n == 0:
            return np.eye(0)

        # Start with identity matrix
        matrix = np.eye(n, dtype=np.float32)

        for i in range(n):
            for j in range(i+1, n):
                player1 = self.players[i]
                player2 = self.players[j]

                # Get correlation coefficient
                corr = self._get_correlation(player1, player2)
                if corr != 0:
                    matrix[i, j] = corr
                    matrix[j, i] = corr  # Symmetric

        return matrix

    def _get_correlation(self, p1: Player, p2: Player) -> float:
        """Calculate correlation between two NFL players."""
        # Same team stacking correlations
        if p1.team == p2.team and p1.team:
            # QB + pass catchers (primary stacks)
            if (p1.position == Position.QB and p2.position.value in NFL_CORRELATION_CONFIG.PASS_CATCHERS) or \
               (p2.position == Position.QB and p1.position.value in NFL_CORRELATION_CONFIG.PASS_CATCHERS):
                return NFL_CORRELATION_CONFIG.QB_PASS_CATCHER_CORRELATION

            # QB + RB (weaker but positive)
            if (p1.position == Position.QB and p2.position == Position.RB) or \
               (p2.position == Position.QB and p1.position == Position.RB):
                return NFL_CORRELATION_CONFIG.QB_RB_CORRELATION

            # RB + DST same team (positive game script)
            if (p1.position == Position.RB and p2.position == Position.DST) or \
               (p2.position == Position.RB and p1.position == Position.DST):
                return NFL_CORRELATION_CONFIG.RB_DST_CORRELATION

            # WR + WR same team (bring-back stacks)
            if p1.position == Position.WR and p2.position == Position.WR:
                return NFL_CORRELATION_CONFIG.WR_WR_CORRELATION

            # TE + WR same team
            if (p1.position == Position.TE and p2.position == Position.WR) or \
               (p2.position == Position.TE and p1.position == Position.WR):
                return NFL_CORRELATION_CONFIG.TE_WR_CORRELATION

        # Same game correlations (different teams)
        game1 = p1.metadata.get('game_id', '')
        game2 = p2.metadata.get('game_id', '')
        if game1 and game2 and game1 == game2 and p1.team != p2.team:
            # Bring-back stacks (different teams, same game)
            if (p1.position.value in NFL_CORRELATION_CONFIG.OFFENSIVE_POSITIONS and
                p2.position.value in NFL_CORRELATION_CONFIG.OFFENSIVE_POSITIONS):
                return NFL_CORRELATION_CONFIG.GAME_STACK_CORRELATION

        # Negative correlations
        if p1.team != p2.team and p1.team and p2.team:
            # DST vs opposing offense
            if (p1.position == Position.DST and p2.position.value in NFL_CORRELATION_CONFIG.OFFENSIVE_POSITIONS):
                if p1.opponent == p2.team:
                    return NFL_CORRELATION_CONFIG.DST_OPPOSING_OFFENSE_CORRELATION
            if (p2.position == Position.DST and p1.position.value in NFL_CORRELATION_CONFIG.OFFENSIVE_POSITIONS):
                if p2.opponent == p1.team:
                    return NFL_CORRELATION_CONFIG.DST_OPPOSING_OFFENSE_CORRELATION

        return 0.0  # No correlation

    def get_rules(self) -> list[CorrelationRule]:
        """Generate NFL correlation rules."""
        rules = []

        for i, player1 in enumerate(self.players):
            for j, player2 in enumerate(self.players[i+1:], start=i+1):
                corr = self._get_correlation(player1, player2)
                if corr != 0:
                    rule_type = self._get_rule_type(player1, player2, corr)
                    rules.append(CorrelationRule(
                        player1_id=i,
                        player2_id=j,
                        correlation=corr,
                        rule_type=rule_type
                    ))

        return rules

    def _get_rule_type(self, p1: Player, p2: Player, corr: float) -> CorrelationType:
        """Determine the type of correlation rule."""
        if p1.team == p2.team:
            if p1.position == Position.QB or p2.position == Position.QB:
                return CorrelationType.QB_STACK
            elif p1.position == Position.DST or p2.position == Position.DST:
                return CorrelationType.DEFENSE_STACK
            else:
                return CorrelationType.SAME_TEAM
        elif corr > 0:
            return CorrelationType.GAME_STACK
        else:
            return CorrelationType.NEGATIVE_CORRELATION


# Sport enum for consistency
class Sport(Enum):
    """Supported sports."""
    MMA = "mma"
    NASCAR = "nascar"
    NFL = "nfl"


def build_correlation_matrix(sport: str, players: list[Player]) -> tuple[np.ndarray, list[CorrelationRule]]:
    """Build correlation matrix for any sport."""
    try:
        sport_enum = Sport(sport.lower())
    except ValueError:
        # Default: no correlations for unsupported sports
        n = len(players)
        return np.eye(n), []

    if sport_enum == Sport.MMA:
        builder = MMACorrelationBuilder(players)
    elif sport_enum == Sport.NASCAR:
        builder = NASCARCorrelationBuilder(players)
    elif sport_enum == Sport.NFL:
        builder = NFLCorrelationBuilder(players)
    else:
        # Default: no correlations
        n = len(players)
        return np.eye(n), []

    matrix = builder.build_matrix()
    rules = builder.get_rules()
    return matrix, rules
