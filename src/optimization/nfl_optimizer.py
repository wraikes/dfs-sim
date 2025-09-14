"""NFL DFS lineup optimizer with stacking logic and position constraints."""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import random
from dataclasses import dataclass
from collections import defaultdict

from ..models.player import Player, Position
from .base_optimizer import BaseOptimizer, SportConstraints, BaseLineup
from .field_generator import BaseFieldGenerator


@dataclass
class NFLLineup(BaseLineup):
    """NFL-specific lineup with stacking analysis."""
    # NFL roster constraints: QB, RB, RB, WR, WR, WR, TE, FLEX, DST
    qb: Optional[Player] = None
    rb1: Optional[Player] = None
    rb2: Optional[Player] = None
    wr1: Optional[Player] = None
    wr2: Optional[Player] = None
    wr3: Optional[Player] = None
    te: Optional[Player] = None
    flex: Optional[Player] = None  # RB/WR/TE
    dst: Optional[Player] = None

    # Stacking metrics
    primary_stack_size: int = 0      # QB + pass catchers from same team
    game_stack_teams: int = 0        # Number of teams in same game
    bring_back_stack: bool = False   # Opposing team players in QB's game
    total_target_share: float = 0.0  # Combined target share of pass catchers

    def __post_init__(self):
        super().__post_init__()
        self._assign_positions()
        self._calculate_stacking_metrics()

    def _assign_positions(self):
        """Assign players to specific NFL roster positions."""
        qbs = [p for p in self.players if p.position == Position.QB]
        rbs = [p for p in self.players if p.position == Position.RB]
        wrs = [p for p in self.players if p.position == Position.WR]
        tes = [p for p in self.players if p.position == Position.TE]
        dsts = [p for p in self.players if p.position == Position.DST]

        # Assign required positions
        self.qb = qbs[0] if qbs else None
        self.dst = dsts[0] if dsts else None

        # Sort skill positions by salary (highest first)
        rbs.sort(key=lambda x: x.salary, reverse=True)
        wrs.sort(key=lambda x: x.salary, reverse=True)
        tes.sort(key=lambda x: x.salary, reverse=True)

        # Assign RBs
        self.rb1 = rbs[0] if len(rbs) >= 1 else None
        self.rb2 = rbs[1] if len(rbs) >= 2 else None

        # Assign WRs
        self.wr1 = wrs[0] if len(wrs) >= 1 else None
        self.wr2 = wrs[1] if len(wrs) >= 2 else None
        self.wr3 = wrs[2] if len(wrs) >= 3 else None

        # Assign TE
        self.te = tes[0] if tes else None

        # Determine FLEX (remaining RB/WR/TE)
        remaining_flex = []
        if len(rbs) >= 3:
            remaining_flex.extend(rbs[2:])
        if len(wrs) >= 4:
            remaining_flex.extend(wrs[3:])
        if len(tes) >= 2:
            remaining_flex.extend(tes[1:])

        self.flex = remaining_flex[0] if remaining_flex else None

    def _calculate_stacking_metrics(self):
        """Calculate NFL stacking correlation metrics."""
        if not self.qb:
            return

        qb_team = self.qb.team
        qb_game_id = getattr(self.qb, 'game_id', '')

        # Find pass catchers on QB's team (primary stack)
        same_team_catchers = []
        all_pass_catchers = [self.wr1, self.wr2, self.wr3, self.te]
        if self.flex and getattr(self.flex, 'position', None) in [Position.WR, Position.TE]:
            all_pass_catchers.append(self.flex)

        for player in all_pass_catchers:
            if player and player.team == qb_team:
                same_team_catchers.append(player)

        self.primary_stack_size = len(same_team_catchers)

        # Calculate total target share of pass catchers
        self.total_target_share = sum(
            getattr(p, 'targets_per_game', 0) for p in all_pass_catchers if p
        )

        # Check for bring-back stack (opponent players in same game)
        game_teams = set()
        for player in self.players:
            if getattr(player, 'game_id', '') == qb_game_id:
                game_teams.add(player.team)

        self.game_stack_teams = len(game_teams)
        self.bring_back_stack = self.game_stack_teams >= 2 and qb_team in game_teams


class NFLFieldGenerator(BaseFieldGenerator):
    """Generate opponent field for NFL contests."""

    def __init__(self, players: List[Player]):
        super().__init__(players)
        self.players_by_position = self._group_players_by_position()

    def _group_players_by_position(self) -> Dict[Position, List[Player]]:
        """Group players by position for efficient sampling."""
        groups = defaultdict(list)
        for player in self.players:
            groups[player.position].append(player)
        return groups

    def generate_field(self, field_size: int) -> List['NFLLineup']:
        """Generate field of opponent lineups."""
        print(f"Generating field of {field_size:,} lineups...")

        field = []
        max_attempts = field_size * 10  # Allow more attempts for complex NFL constraints
        attempts = 0

        while len(field) < field_size and attempts < max_attempts:
            attempts += 1
            lineup = self.generate_lineup()
            if lineup:
                field.append(lineup)

        print(f"✅ Generated {len(field):,} total lineups")
        return field

    def generate_lineup(self) -> Optional['NFLLineup']:
        """Generate a single opponent lineup."""
        try:
            # Sample players by position based on ownership
            qb = self._sample_by_ownership(self.players_by_position[Position.QB], 1)[0]
            rbs = self._sample_by_ownership(self.players_by_position[Position.RB], 3)[:2]  # Take top 2
            wrs = self._sample_by_ownership(self.players_by_position[Position.WR], 4)[:3]  # Take top 3
            te = self._sample_by_ownership(self.players_by_position[Position.TE], 1)[0]
            dst = self._sample_by_ownership(self.players_by_position[Position.DST], 1)[0]

            # Sample FLEX from remaining skill positions
            flex_pool = (
                self.players_by_position[Position.RB] +
                self.players_by_position[Position.WR] +
                self.players_by_position[Position.TE]
            )
            flex_candidates = [p for p in flex_pool if p not in rbs + wrs + [te]]
            flex = self._sample_by_ownership(flex_candidates, 1)[0] if flex_candidates else None

            players = [qb] + rbs + wrs + [te, flex, dst]
            players = [p for p in players if p is not None]

            # Validate lineup meets constraints
            if len(players) != 9:  # NFL requires exactly 9 players
                return None

            if sum(p.salary for p in players) > 50000:  # Salary cap check
                return None

            return NFLLineup(players=players)

        except (IndexError, ValueError):
            return None

    def _sample_by_ownership(self, players: List[Player], count: int) -> List[Player]:
        """Sample players weighted by ownership percentage."""
        if not players or count <= 0:
            return []

        # Weight by ownership (higher ownership = more likely to be selected)
        weights = [max(p.ownership, 0.1) for p in players]  # Minimum 0.1% weight

        try:
            sampled = np.random.choice(
                players,
                size=min(count, len(players)),
                replace=False,
                p=np.array(weights) / sum(weights)
            )
            return list(sampled)
        except ValueError:
            # Fallback to random sampling
            return random.sample(players, min(count, len(players)))

    def _select_by_ownership_probability(self, players: List[Player], count: int) -> List[Player]:
        """Select players by ownership probability (required by base class)."""
        return self._sample_by_ownership(players, count)

    def _has_invalid_combinations(self, lineup) -> bool:
        """Check for invalid player combinations (required by base class)."""
        # NFL doesn't have invalid combinations like opposing fighters
        return False

    def get_player_ids(self) -> set:
        """Get set of player IDs in this lineup (for uniqueness calculation)."""
        return {p.player_id for p in self.players}


class NFLOptimizer(BaseOptimizer):
    """NFL-specific DFS optimizer with stacking logic."""

    def __init__(self, players: List[Player], sport: str = 'nfl', field_size: int = 10000):
        super().__init__(players, sport, field_size)

    def _get_sport_constraints(self) -> SportConstraints:
        """Get NFL-specific constraints."""
        return SportConstraints(
            salary_cap=50000,           # DraftKings NFL salary cap
            roster_size=9,              # QB + 2RB + 3WR + TE + FLEX + DST
            max_salary_remaining=1000,  # Reasonable remaining salary
            min_salary_remaining=0,     # Can spend full cap
            max_lineup_ownership=200.0, # Higher ownership tolerance for NFL
            min_leverage_plays=1,       # At least 1 contrarian play
            max_lineup_overlap=0.40,    # Allow more overlap for stacking
            sport_rules={
                'min_stack_size': 2,    # QB + at least 1 pass catcher
                'max_same_team': 4,     # Max 4 players from same team
                'max_same_game': 6,     # Max 6 players from same game
                'require_qb_stack': True, # QB must have at least 1 teammate
                'salary_ownership_correlation': True,  # Apply salary-ownership guidelines
            }
        )

    def _create_field_generator(self) -> BaseFieldGenerator:
        """Create NFL-specific field generator."""
        return NFLFieldGenerator(self.players)

    def _validate_lineup(self, players: List[Player]) -> bool:
        """Validate NFL lineup meets position and stacking constraints."""
        if len(players) != 9:
            return False

        # Check salary cap and minimum usage
        total_salary = sum(p.salary for p in players)
        if total_salary > self.constraints.salary_cap:
            return False


        # Check position requirements
        position_counts = {}
        for player in players:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # NFL position requirements: 1 QB, 2-3 RB, 3-4 WR, 1-2 TE, 1 DST, 1 FLEX
        required_positions = {
            Position.QB: (1, 1),     # Exactly 1 QB
            Position.RB: (2, 3),     # 2-3 RBs (2 base + potential FLEX)
            Position.WR: (3, 4),     # 3-4 WRs (3 base + potential FLEX)
            Position.TE: (1, 2),     # 1-2 TEs (1 base + potential FLEX)
            Position.DST: (1, 1)     # Exactly 1 DST
        }

        for pos, (min_count, max_count) in required_positions.items():
            actual_count = position_counts.get(pos, 0)
            if not (min_count <= actual_count <= max_count):
                return False

        # Check team stacking constraints
        team_counts = {}
        game_counts = {}
        for player in players:
            team = player.team
            game_id = player.metadata.get('game_id', '')

            team_counts[team] = team_counts.get(team, 0) + 1
            if game_id:
                game_counts[game_id] = game_counts.get(game_id, 0) + 1

        # Enforce max players per team/game
        max_same_team = self.constraints.sport_rules.get('max_same_team', 4)
        max_same_game = self.constraints.sport_rules.get('max_same_game', 6)

        if any(count > max_same_team for count in team_counts.values()):
            return False

        if any(count > max_same_game for count in game_counts.values()):
            return False

        # Check QB stacking requirement
        if self.constraints.sport_rules.get('require_qb_stack', True):
            qbs = [p for p in players if p.position == Position.QB]
            if qbs:
                qb_team = qbs[0].team
                teammates = [p for p in players if p.team == qb_team and p.position != Position.QB]
                min_stack = self.constraints.sport_rules.get('min_stack_size', 2)
                if len(teammates) < min_stack - 1:  # -1 because QB counts as 1
                    return False

        return True

    def _create_lineup(self, players: List[Player]) -> BaseLineup:
        """Create NFL-specific lineup object."""
        return NFLLineup(players=players)

    def _generate_single_lineup(self) -> Optional[BaseLineup]:
        """Generate a single candidate NFL lineup with stacking logic."""
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                # Step 1: Select QB (weighted by projection and value)
                qbs = [p for p in self.players if p.position == Position.QB]
                if not qbs:
                    continue

                # Weight QBs by projection and ceiling (not value)
                qb_weights = []
                for qb in qbs:
                    projection_score = qb.projection
                    ceiling_score = getattr(qb, 'ceiling', qb.projection)
                    weight = projection_score + ceiling_score * 0.5  # Favor projection with ceiling upside
                    qb_weights.append(weight)

                qb_probs = np.array(qb_weights) / sum(qb_weights)
                qb = np.random.choice(qbs, p=qb_probs)

                # Step 2: Build primary stack (QB + pass catchers from same team)
                stack_candidates = [
                    p for p in self.players
                    if p.team == qb.team and p.position in [Position.WR, Position.TE, Position.RB]
                ]

                # Select 1-3 stack partners based on target share and value
                stack_size = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]  # Weighted toward 2-player stacks
                stack_partners = []

                if stack_candidates:
                    # Weight by targets per game and projection (not value)
                    stack_weights = []
                    for player in stack_candidates:
                        targets = player.metadata.get('targets_per_game', 0)
                        projection_score = player.projection
                        ceiling_score = getattr(player, 'ceiling', player.projection)
                        # Target share and projection, with ceiling upside
                        weight = (targets * 3) + projection_score + (ceiling_score * 0.3)
                        stack_weights.append(max(weight, 0.1))

                    stack_probs = np.array(stack_weights) / sum(stack_weights)

                    # Sample stack partners
                    available_partners = stack_candidates.copy()
                    for _ in range(min(stack_size, len(available_partners))):
                        if available_partners:
                            partner_probs = np.array([
                                stack_probs[stack_candidates.index(p)] for p in available_partners
                            ])
                            partner_probs = partner_probs / partner_probs.sum()

                            partner = np.random.choice(available_partners, p=partner_probs)
                            stack_partners.append(partner)
                            available_partners.remove(partner)

                # Step 3: Fill remaining positions with optimal players
                selected_players = [qb] + stack_partners
                selected_ids = {p.player_id for p in selected_players}

                remaining_budget = self.constraints.salary_cap - sum(p.salary for p in selected_players)

                # Get remaining position requirements
                selected_positions = [p.position for p in selected_players]
                needed_positions = self._get_remaining_positions(selected_positions)

                # Fill remaining positions with value-based selection
                for pos in needed_positions:
                    candidates = [
                        p for p in self.players
                        if p.position == pos and p.player_id not in selected_ids
                        and p.salary <= remaining_budget
                    ]

                    if not candidates:
                        break

                    # Weight by projection and matchup (not value)
                    weights = []
                    for player in candidates:
                        projection_score = player.projection
                        ceiling_score = getattr(player, 'ceiling', player.projection)
                        matchup_bonus = player.metadata.get('matchup_fppg_allowed', 15.0) / 20.0
                        # Focus on projection + ceiling + matchup
                        weight = projection_score + (ceiling_score * 0.3) + matchup_bonus
                        weights.append(max(weight, 0.1))

                    probs = np.array(weights) / sum(weights)
                    selected = np.random.choice(candidates, p=probs)

                    selected_players.append(selected)
                    selected_ids.add(selected.player_id)
                    remaining_budget -= selected.salary

                if len(selected_players) == 9 and self._validate_lineup(selected_players):
                    return self._create_lineup(selected_players)

            except (ValueError, IndexError, ZeroDivisionError):
                continue

        return None

    def _get_remaining_positions(self, selected_positions: List[Position]) -> List[Position]:
        """Determine remaining positions needed to complete NFL lineup."""
        required = [
            Position.QB,  # 1
            Position.RB, Position.RB,  # 2
            Position.WR, Position.WR, Position.WR,  # 3
            Position.TE,  # 1
            Position.DST  # 1
        ]

        # Add FLEX position (RB, WR, or TE)
        flex_options = [Position.RB, Position.WR, Position.TE]
        required.append(random.choice(flex_options))

        # Remove already selected positions
        remaining = required.copy()
        for pos in selected_positions:
            if pos in remaining:
                remaining.remove(pos)

        return remaining

    def _score_lineup_gpp(self, lineup: BaseLineup) -> float:
        """Score NFL lineup for GPP success with stacking bonuses."""
        if not isinstance(lineup, NFLLineup):
            return 0.0

        # Base scoring: 70% simulation results + 30% other factors
        sim_score = lineup.percentile_95 * 0.7

        # Stacking bonuses (critical for NFL GPP success)
        stack_bonus = 0
        if lineup.primary_stack_size >= 2:
            stack_bonus += 20  # Base QB stack bonus
        if lineup.primary_stack_size >= 3:
            stack_bonus += 15  # Large stack bonus
        if lineup.bring_back_stack:
            stack_bonus += 10  # Bring-back correlation bonus

        # Target share bonus (high-target players in stacks)
        target_bonus = min(lineup.total_target_share * 2, 30)  # Cap at 30 points

        # Leverage scoring (contrarian ownership)
        leverage_bonus = self._calculate_lineup_value_leverage(lineup)

        # Ceiling potential (95th percentile focus)
        ceiling_bonus = (lineup.percentile_95 - lineup.percentile_50) * 0.3

        # Combine all scoring components
        total_score = sim_score + stack_bonus + target_bonus + (leverage_bonus * 0.3) + ceiling_bonus

        return max(total_score, 0)

    def _display_lineup_players(self, lineup: BaseLineup):
        """Display NFL lineup with position assignments and stacking info."""
        if not isinstance(lineup, NFLLineup):
            return

        print(f"\n🏈 NFL LINEUP:")

        # Display by position with stacking indicators
        positions = [
            ("QB", lineup.qb),
            ("RB1", lineup.rb1),
            ("RB2", lineup.rb2),
            ("WR1", lineup.wr1),
            ("WR2", lineup.wr2),
            ("WR3", lineup.wr3),
            ("TE", lineup.te),
            ("FLEX", lineup.flex),
            ("DST", lineup.dst)
        ]

        for pos_name, player in positions:
            if player:
                # Check if player is in primary stack
                stack_indicator = "🔗" if (lineup.qb and player.team == lineup.qb.team and
                                         player.position != Position.QB) else "  "

                targets = getattr(player, 'targets_per_game', 0)
                target_info = f" ({targets:.1f}T)" if targets > 0 else ""

                print(f"   {stack_indicator} {pos_name:4}: {player.name:20} ${player.salary:4,} "
                      f"{player.projection:4.1f}pts {player.ownership:4.1f}%{target_info}")

        # Display stacking summary
        if lineup.primary_stack_size > 1:
            print(f"\n📊 STACK ANALYSIS:")
            print(f"   Primary Stack: {lineup.primary_stack_size} players ({lineup.qb.team})")
            print(f"   Total Targets: {lineup.total_target_share:.1f}/game")
            if lineup.bring_back_stack:
                print(f"   Bring-back: {lineup.game_stack_teams} teams in game")

    def _create_player_from_row(self, row: pd.Series) -> Player:
        """Create Player object from processed CSV row."""
        player = Player(
            player_id=str(row['player_id']),
            name=row['name'],
            position=self._get_position_from_string(row['position']),
            team=row['team'],
            opponent=row.get('opponent', ''),
            salary=int(row['salary']),
            projection=float(row['updated_projection']),
            ownership=float(row['updated_ownership']),
            floor=float(row['updated_floor']),
            ceiling=float(row['updated_ceiling']),
            std_dev=float(row.get('std_dev', 25.0)),
            game_total=float(row.get('game_total', 45.0)),
            team_total=float(row.get('team_total', 22.5)),
            spread=float(row.get('spread', 0.0))
        )

        # Add NFL-specific metadata
        player.metadata = {
            'game_id': row.get('game_id', ''),
            'targets_per_game': float(row.get('targets_per_game', 0.0)),
            'snap_pct': float(row.get('snap_pct', 50.0)),
            'rz_targets': float(row.get('rz_targets', 0.0)),
            'matchup_fppg_allowed': float(row.get('matchup_fppg_allowed', 15.0)),
            'speed_advantage': float(row.get('speed_advantage', 0.0)),
            'shadow_coverage': bool(row.get('shadow_coverage', False))
        }

        return player

    def _get_position_from_string(self, pos_str: str) -> Position:
        """Convert position string to Position enum."""
        pos_map = {
            'QB': Position.QB,
            'RB': Position.RB,
            'WR': Position.WR,
            'TE': Position.TE,
            'K': Position.K,
            'DST': Position.DST,
            'D': Position.DST,
            'DEF': Position.DST
        }
        return pos_map.get(pos_str.upper(), Position.FLEX)