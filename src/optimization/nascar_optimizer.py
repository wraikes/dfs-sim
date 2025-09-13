"""NASCAR DFS lineup optimizer using base architecture."""

import numpy as np
import pandas as pd
from typing import List, Optional
import random
from dataclasses import dataclass

from ..models.player import Player, Position
from .base_optimizer import BaseOptimizer, SportConstraints, BaseLineup
from .field_generator import BaseFieldGenerator


@dataclass
class NASCARLineup(BaseLineup):
    """NASCAR-specific lineup with additional metrics."""
    # NASCAR-specific metrics
    num_dominators: int = 0  # Drivers starting in top 5
    num_value_plays: int = 0  # Drivers starting outside top 20
    manufacturer_diversity: int = 0  # Number of unique manufacturers
    position_differential_upside: float = 0.0  # Total PD upside

    def __post_init__(self):
        super().__post_init__()

        # Calculate NASCAR-specific metrics
        self.num_dominators = sum(1 for p in self.players if p.metadata.get('starting_position', 99) <= 5)
        self.num_value_plays = sum(1 for p in self.players if p.metadata.get('starting_position', 99) > 20)

        # Manufacturer diversity
        manufacturers = set(p.metadata.get('manufacturer', 'Unknown') for p in self.players)
        self.manufacturer_diversity = len(manufacturers)

        # Position differential upside (drivers starting outside top 15 have PD upside)
        self.position_differential_upside = sum(
            max(0, p.metadata.get('starting_position', 15) - 15)
            for p in self.players
        )


class NASCARFieldGenerator(BaseFieldGenerator):
    """NASCAR-specific field generator for modeling opponent lineups."""

    def __init__(self, players: List[Player]):
        super().__init__(players)
        self.salary_cap = 50000
        self.roster_size = 6

    def _has_invalid_combinations(self, players: List[Player]) -> bool:
        """Check for NASCAR-specific invalid combinations (simple - no direct restrictions)."""
        # NASCAR has manufacturer/team limits but we'll keep this simple
        # Skip constraints if using placeholder data

        mfg_counts = {}
        team_counts = {}

        for player in players:
            mfg = player.metadata.get('manufacturer', 'Unknown')
            # Skip check if all manufacturers are "Unknown" (placeholder data)
            if mfg != 'Unknown':
                mfg_counts[mfg] = mfg_counts.get(mfg, 0) + 1
                if mfg_counts[mfg] > 3:  # Very loose limit for field generation
                    return True

            team = player.team if player.team else 'Unknown'
            if team != 'Unknown' and team != '':
                team_counts[team] = team_counts.get(team, 0) + 1
                if team_counts[team] > 3:  # Very loose limit for field generation
                    return True

        return False

    def _select_by_ownership_probability(self, available: List[Player], n: int = 1) -> List[Player]:
        """Select drivers weighted by ownership probability."""
        if not available or n <= 0:
            return []

        # Normalize ownership to probabilities
        ownerships = np.array([p.ownership for p in available])

        # Add small epsilon to avoid zero probability
        ownerships = np.maximum(ownerships, 0.1)

        # Convert to selection probabilities
        probabilities = ownerships / ownerships.sum()

        # Select without replacement
        n = min(n, len(available))
        selected_indices = np.random.choice(
            len(available),
            size=n,
            replace=False,
            p=probabilities
        )

        return [available[i] for i in selected_indices]

    def generate_field(self, n_lineups: int = 10000) -> List:
        """Generate realistic field of NASCAR opponent lineups."""
        field = []

        for lineup_id in range(n_lineups):
            # Generate a single field lineup
            lineup = self._generate_single_lineup()
            if lineup:
                from .field_generator import FieldLineup
                field_lineup = FieldLineup(players=lineup, lineup_id=lineup_id)
                field.append(field_lineup)

        return field

    def _generate_single_lineup(self) -> Optional[List[Player]]:
        """Generate single NASCAR lineup for field simulation."""
        max_attempts = 50

        for attempt in range(max_attempts):
            selected = []

            try:
                # Simple strategy: ownership-weighted selection with position balance

                # Get 1-2 dominators (P1-P12) based on ownership
                dominators = [p for p in self.players if p.metadata.get('starting_position', 99) <= 12]
                if dominators:
                    num_dominators = random.choice([1, 2])
                    selected.extend(self._select_by_ownership_probability(dominators, num_dominators))

                # Fill rest with ownership-weighted selection from remaining drivers
                remaining_slots = 6 - len(selected)
                if remaining_slots > 0:
                    remaining = [p for p in self.players if p not in selected]
                    if remaining:
                        selected.extend(self._select_by_ownership_probability(remaining, remaining_slots))

                # Validate basic constraints
                if len(selected) == 6:
                    total_salary = sum(p.salary for p in selected)
                    if 48000 <= total_salary <= 50000:
                        if not self._has_invalid_combinations(selected):
                            return selected

            except (ValueError, IndexError):
                continue

        return None


class NASCAROptimizer(BaseOptimizer):
    """NASCAR-specific DFS optimizer."""

    def __init__(self, players: List[Player], sport: str = 'nascar', field_size: int = 10000):
        super().__init__(players, sport, field_size)

    def _get_sport_constraints(self) -> SportConstraints:
        """Get NASCAR-specific constraints based on track type."""
        # Detect if superspeedway based on track type in metadata
        is_superspeedway = any(p.metadata.get('track_type', '').lower() == 'superspeedway' for p in self.players)

        if is_superspeedway:
            # Superspeedway rules: more chaos, more underdogs
            return SportConstraints(
                salary_cap=50000,
                roster_size=6,
                max_salary_remaining=2000,  # Can leave more on superspeedways
                min_salary_remaining=300,
                max_lineup_ownership=120.0,  # Lower ownership cap
                min_leverage_plays=2,
                max_lineup_overlap=0.4,
                sport_rules={
                    'track_type': 'superspeedway',
                    'max_from_top12': 1,  # Max 1 driver from P1-P12
                    'min_from_back': 4,  # At least 4 drivers from P23+
                    'max_manufacturer_count': 2,
                    'max_team_count': 2,
                }
            )
        else:
            # Non-superspeedway (intermediate/road course) rules
            return SportConstraints(
                salary_cap=50000,
                roster_size=6,
                max_salary_remaining=1500,
                min_salary_remaining=300,
                max_lineup_ownership=135.0,
                min_leverage_plays=2,  # At least 2 drivers ≤12% owned
                max_lineup_overlap=0.5,
                sport_rules={
                    'track_type': 'intermediate',
                    'max_dominators': 2,  # 1-2 dominators from P1-P12
                    'min_pd_plays': 3,  # At least 3 PD drivers from P18+
                    'max_from_top12': 2,  # Max 2 drivers from P1-P12 together
                    'max_manufacturer_count': 2,
                    'max_team_count': 2,
                }
            )

    def _create_field_generator(self) -> BaseFieldGenerator:
        """Create NASCAR field generator."""
        return NASCARFieldGenerator(self.players)

    def _validate_lineup(self, players: List[Player]) -> bool:
        """Validate NASCAR lineup meets sport-specific rules."""
        if len(players) != 6:
            return False

        # Check salary constraints
        total_salary = sum(p.salary for p in players)
        salary_remaining = self.constraints.salary_cap - total_salary
        if (salary_remaining > self.constraints.max_salary_remaining or
            salary_remaining < self.constraints.min_salary_remaining):
            return False

        # Check manufacturer/team diversity (skip if data unavailable - using placeholders)
        manufacturer_counts = {}
        team_counts = {}
        for player in players:
            mfg = player.metadata.get('manufacturer', 'Unknown')
            # Skip manufacturer constraint if all are "Unknown" (placeholder data)
            if mfg != 'Unknown':
                manufacturer_counts[mfg] = manufacturer_counts.get(mfg, 0) + 1
                if manufacturer_counts[mfg] > self.constraints.sport_rules.get('max_manufacturer_count', 2):
                    return False

            # Also check team counts (skip if data unavailable)
            team = player.team if player.team else 'Unknown'
            if team != 'Unknown' and team != '':
                team_counts[team] = team_counts.get(team, 0) + 1
                if team_counts[team] > self.constraints.sport_rules.get('max_team_count', 2):
                    return False

        # Get starting positions
        starting_positions = [p.metadata.get('starting_position', 20) for p in players]

        # Track-specific validation
        track_type = self.constraints.sport_rules.get('track_type', 'intermediate')

        if track_type == 'superspeedway':
            # Superspeedway rules
            # Max 1 driver from P1-P12
            top12_count = sum(1 for pos in starting_positions if pos <= 12)
            if top12_count > self.constraints.sport_rules.get('max_from_top12', 1):
                return False

            # At least 4 drivers from P23+
            back_count = sum(1 for pos in starting_positions if pos >= 23)
            if back_count < self.constraints.sport_rules.get('min_from_back', 4):
                return False

        else:
            # Non-superspeedway rules
            # Max 2 drivers from P1-P12
            top12_count = sum(1 for pos in starting_positions if pos <= 12)
            if top12_count > self.constraints.sport_rules.get('max_from_top12', 2):
                return False

            # At least 3 PD drivers from P18+
            pd_plays = sum(1 for pos in starting_positions if pos >= 18)
            if pd_plays < self.constraints.sport_rules.get('min_pd_plays', 3):
                return False

        # Check ownership constraints
        total_ownership = sum(p.ownership for p in players)
        if total_ownership > self.constraints.max_lineup_ownership:
            return False

        # Check leverage plays requirement
        # NASCAR rule: At least 2 drivers ≤12% owned
        low_owned = sum(1 for p in players if p.ownership <= 12)
        if low_owned < self.constraints.min_leverage_plays:
            return False

        # Max 3 drivers ≥25% owned
        high_owned = sum(1 for p in players if p.ownership >= 25)
        if high_owned > 3:
            return False

        return True

    def _create_lineup(self, players: List[Player]) -> BaseLineup:
        """Create NASCAR lineup object."""
        return NASCARLineup(players=players)

    def _create_player_from_row(self, row: pd.Series) -> Player:
        """Create Player object from CSV row."""
        player = Player(
            player_id=int(row['player_id']),
            name=row['name'],
            position=Position.DRIVER,
            team=row.get('team', ''),
            opponent='',  # NASCAR doesn't have direct opponents
            salary=int(row['salary']),
            projection=float(row.get('updated_projection', row['projection'])),
            ownership=float(row.get('updated_ownership', row['ownership'])),
            floor=float(row.get('updated_floor', row['floor'])),
            ceiling=float(row.get('updated_ceiling', row['ceiling'])),
            std_dev=float(row.get('std_dev', 35.0)),  # Higher variance for NASCAR
            game_total=0.0,  # Not applicable for NASCAR
            team_total=0.0,  # Not applicable for NASCAR
            spread=0.0,      # Not applicable for NASCAR
            value=float(row.get('value', 0.0))
        )

        # Add NASCAR-specific metadata
        player.metadata = {
            'starting_position': int(row.get('starting_position', 20)),
            'manufacturer': row.get('manufacturer', 'Unknown'),
            'win_odds': float(row.get('win_odds', 0)),
            'practice_speed': float(row.get('practice_speed', 0)),
            'avg_finish': float(row.get('avg_finish', 20)),
            'top5_pct': float(row.get('top5_pct', 0)),
            'top10_pct': float(row.get('top10_pct', 0)),
            'dnf_pct': float(row.get('dnf_pct', 0)),
            'laps_led_avg': float(row.get('laps_led_avg', 0)),
            'dominator_score': float(row.get('dominator_score', 0)),
            'pd_upside': float(row.get('pd_upside', 0)),
            'track_type': row.get('track_type', 'intermediate'),
            'newsletter_signal': row.get('newsletter_signal', 'neutral'),
            'newsletter_confidence': float(row.get('newsletter_confidence', 0.5)),
        }

        return player

    def _generate_single_lineup(self) -> Optional[BaseLineup]:
        """Generate a single NASCAR lineup candidate based on track type."""
        attempts = 0
        max_attempts = 100

        # Detect track type
        track_type = self.constraints.sport_rules.get('track_type', 'intermediate')

        while attempts < max_attempts:
            attempts += 1
            selected = []

            try:
                if track_type == 'superspeedway':
                    # Superspeedway strategy: chaos lineup with mostly back-of-pack drivers
                    front_runners = [p for p in self.players if p.metadata.get('starting_position', 99) <= 12]
                    mid_pack = [p for p in self.players if 13 <= p.metadata.get('starting_position', 99) <= 22]
                    back_pack = [p for p in self.players if p.metadata.get('starting_position', 99) >= 23]

                    # Max 1 from P1-P12
                    if front_runners and random.random() < 0.7:  # 70% chance to include 1 front runner
                        selected.append(random.choice(front_runners))

                    # 1-2 from mid-pack
                    num_mid = random.choice([1, 2])
                    if mid_pack:
                        selected.extend(random.sample(mid_pack, min(num_mid, len(mid_pack))))

                    # Fill rest with back-of-pack (need at least 4 from P23+)
                    remaining_slots = 6 - len(selected)
                    if back_pack and remaining_slots > 0:
                        # Prioritize low-owned back markers
                        back_sorted = sorted(back_pack, key=lambda x: x.ownership)
                        selected.extend(random.sample(back_sorted, min(remaining_slots, len(back_sorted))))

                else:
                    # Non-superspeedway strategy: balanced with dominators and PD plays
                    dominators = [p for p in self.players if p.metadata.get('starting_position', 99) <= 12]
                    mid_tier = [p for p in self.players if 13 <= p.metadata.get('starting_position', 99) <= 17]
                    pd_plays = [p for p in self.players if p.metadata.get('starting_position', 99) >= 18]

                    # 1-2 dominators from P1-P12, prioritizing dominator score + ownership leverage
                    num_dominators = random.choice([1, 2])
                    if dominators:
                        # Sort by combined dominator score and ownership leverage
                        # Higher dominator score is better, lower ownership is better
                        dom_sorted = sorted(dominators, key=lambda x: (-x.metadata.get('dominator_score', 0), x.ownership))

                        # Select from top dominator candidates with some randomization for diversity
                        if random.random() < 0.7:  # 70% chance for pure dominator score selection
                            top_dominators = dom_sorted[:min(6, len(dom_sorted))]  # Top 6 dominator candidates
                        else:  # 30% chance for ownership-first selection (more contrarian)
                            top_dominators = sorted(dominators, key=lambda x: (x.ownership, -x.metadata.get('dominator_score', 0)))[:6]

                        selected.extend(random.sample(top_dominators, min(num_dominators, len(top_dominators))))

                    # 1-2 mid-tier drivers
                    num_mid = random.choice([1, 2])
                    if mid_tier:
                        selected.extend(random.sample(mid_tier, min(num_mid, len(mid_tier))))

                    # Fill rest with PD plays (need at least 3 from P18+) - prioritize upside + leverage
                    remaining_slots = 6 - len(selected)
                    if pd_plays and remaining_slots > 0:
                        # Sort by PD upside potential and ownership leverage
                        # Higher PD upside is better, lower ownership is better
                        pd_sorted = sorted(pd_plays, key=lambda x: (-x.metadata.get('pd_upside', 0), x.ownership))

                        # Strategy: Mix high-upside with some randomization for diversity
                        if remaining_slots >= 2 and len(pd_sorted) >= 2:
                            # 80% chance to take the best PD upside play
                            if random.random() < 0.8:
                                selected.append(pd_sorted[0])  # Best PD upside play
                                remaining_slots -= 1

                            # Select from expanded PD candidates for diversity
                            pd_candidates = pd_sorted[:min(12, len(pd_sorted))]  # Top 12 PD plays
                            remaining_needed = min(remaining_slots, len([p for p in pd_candidates if p not in selected]))
                            if remaining_needed > 0:
                                available_pd = [p for p in pd_candidates if p not in selected]
                                selected.extend(random.sample(available_pd, remaining_needed))
                        else:
                            # Select from expanded PD upside candidates for more diversity
                            top_pd = pd_sorted[:min(10, len(pd_sorted))]
                            selected.extend(random.sample(top_pd, min(remaining_slots, len(top_pd))))

                # Fill to 6 drivers if needed
                while len(selected) < 6:
                    remaining = [p for p in self.players if p not in selected]
                    if remaining:
                        # Prefer low-owned drivers with good value
                        remaining_sorted = sorted(remaining, key=lambda x: (x.ownership, -x.value))
                        selected.append(remaining_sorted[0])
                    else:
                        break

                if len(selected) == 6:
                    lineup = self._create_lineup(selected)
                    if self._validate_lineup(selected):
                        return lineup

            except (ValueError, IndexError):
                continue

        return None

    def _score_lineup_gpp(self, lineup: BaseLineup) -> float:
        """Score NASCAR lineup for GPP success using sport rules constraints."""
        # Run simulation if needed
        if lineup.simulated_scores is None:
            scores = self.simulator._simulate_correlated(lineup.players)
            lineup.simulated_scores = scores
            lineup.percentile_25 = np.percentile(scores, 25)
            lineup.percentile_50 = np.percentile(scores, 50)
            lineup.percentile_75 = np.percentile(scores, 75)
            lineup.percentile_95 = np.percentile(scores, 95)
            lineup.percentile_99 = np.percentile(scores, 99)

        # Base score from 95th percentile (main GPP target)
        base_score = lineup.percentile_95

        # Ownership leverage bonus (lower ownership = higher leverage)
        ownership_leverage = max(0, self.constraints.max_lineup_ownership - lineup.total_ownership) / self.constraints.max_lineup_ownership * 25

        # Position differential bonus based on sport rules
        pd_bonus = 0
        dominators = 0
        for player in lineup.players:
            starting_pos = player.metadata.get('starting_position', 20)

            # Count dominators (P1-P12)
            if starting_pos <= 12:
                dominators += 1

            # PD play bonus (P18+)
            if starting_pos >= 18:
                pd_bonus += 5  # Flat bonus for each PD driver
            if starting_pos >= 25:  # Deep PD play
                pd_bonus += 3  # Extra bonus for extreme PD

        # Dominator balance based on track type rules
        track_type = self.constraints.sport_rules.get('track_type', 'intermediate')
        dominator_bonus = 0

        if track_type == 'superspeedway':
            max_from_top12 = self.constraints.sport_rules.get('max_from_top12', 1)
            if dominators <= max_from_top12:
                dominator_bonus = 10  # Reward staying within superspeedway limits
            else:
                dominator_bonus = -15  # Penalty for violating superspeedway rules
        else:
            max_dominators = self.constraints.sport_rules.get('max_dominators', 2)
            if dominators <= max_dominators:
                dominator_bonus = 8 if dominators == 1 else 5  # Prefer 1 but allow 2
            else:
                dominator_bonus = -10  # Penalty for too many dominators

        # Track-specific bonuses using sport rules
        track_bonus = 0
        if track_type == 'superspeedway':
            # Reward chaos lineups meeting min_from_back requirement
            min_from_back = self.constraints.sport_rules.get('min_from_back', 4)
            back_count = sum(1 for p in lineup.players if p.metadata.get('starting_position', 20) >= 23)
            if back_count >= min_from_back:
                track_bonus += 15  # Big bonus for meeting chaos requirements
        else:
            # Reward balanced approach meeting min_pd_plays
            min_pd_plays = self.constraints.sport_rules.get('min_pd_plays', 3)
            pd_count = sum(1 for p in lineup.players if p.metadata.get('starting_position', 20) >= 18)
            if pd_count >= min_pd_plays:
                track_bonus += 10  # Bonus for meeting PD requirements

        # Manufacturer/team diversity using sport rules limits
        max_mfg_count = self.constraints.sport_rules.get('max_manufacturer_count', 2)
        max_team_count = self.constraints.sport_rules.get('max_team_count', 2)

        mfg_counts = {}
        team_counts = {}
        for player in lineup.players:
            mfg = player.metadata.get('manufacturer', 'Unknown')
            mfg_counts[mfg] = mfg_counts.get(mfg, 0) + 1

            team = player.team if player.team else 'Unknown'
            team_counts[team] = team_counts.get(team, 0) + 1

        concentration_penalty = 0
        for count in mfg_counts.values():
            if count > max_mfg_count:
                concentration_penalty -= 10  # Penalty for violating manufacturer limits

        for count in team_counts.values():
            if count > max_team_count:
                concentration_penalty -= 8  # Penalty for violating team limits

        return base_score + ownership_leverage + pd_bonus + dominator_bonus + track_bonus + concentration_penalty

    def _display_lineup_players(self, lineup: BaseLineup):
        """Display NASCAR lineup players."""
        print(f"\n🏁 Drivers:")
        for player in lineup.players:
            signal = player.metadata.get('newsletter_signal', 'neutral')
            signal_icon = {'target': '🎯', 'avoid': '⛔', 'volatile': '⚡'}.get(signal, '  ')

            starting_pos = player.metadata.get('starting_position', 20)
            win_odds = player.metadata.get('win_odds', 0)

            # Position category
            if starting_pos <= 5:
                pos_category = "DOM"
            elif starting_pos <= 12:
                pos_category = "MID"
            elif starting_pos >= 18:
                pos_category = "PD "
            else:
                pos_category = "   "

            print(f"  {signal_icon} {player.name:18} ${player.salary:5,} | "
                  f"P{starting_pos:2d} {pos_category} | "
                  f"{player.projection:5.1f}pts ({player.ownership:4.1f}%) | "
                  f"Win: ${win_odds:+4.0f}")