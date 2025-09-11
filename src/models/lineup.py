"""Lineup model for DFS simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

from .player import Player, Position
from .site import Site


@dataclass
class Lineup:
    """Represents a DFS lineup with roster constraints."""
    
    players: List[Player]
    site: Site
    
    # Calculated fields
    _total_salary: Optional[int] = field(default=None, init=False)
    _total_projection: Optional[float] = field(default=None, init=False)
    _total_ownership: Optional[float] = field(default=None, init=False)
    _simulated_scores: List[float] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Validate lineup and calculate totals."""
        self._validate()
        self._calculate_totals()
    
    def _validate(self):
        """Validate lineup meets site requirements."""
        is_valid, error_msg = self.site.validate_lineup(self.players)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Check unique players
        if len(set(self.players)) != len(self.players):
            raise ValueError("Lineup contains duplicate players")
    
    def _get_locked_players(self) -> Set[Player]:
        """Get players locked into non-FLEX positions."""
        locked = set()
        for position, required in self.site.roster_slots.items():
            if position not in [Position.FLEX, Position.UTIL, Position.G, Position.F]:
                position_players = [p for p in self.players if p.position == position]
                locked.update(position_players[:required])
        return locked
    
    def _calculate_totals(self):
        """Calculate lineup totals."""
        self._total_salary = sum(p.salary for p in self.players)
        self._total_projection = sum(p.adjusted_projection for p in self.players)
        self._total_ownership = sum(p.ownership for p in self.players) / len(self.players)
    
    @property
    def total_salary(self) -> int:
        """Get total lineup salary."""
        if self._total_salary is None:
            self._calculate_totals()
        return self._total_salary
    
    @property
    def total_projection(self) -> float:
        """Get total lineup projection."""
        if self._total_projection is None:
            self._calculate_totals()
        return self._total_projection
    
    @property
    def total_ownership(self) -> float:
        """Get average lineup ownership."""
        if self._total_ownership is None:
            self._calculate_totals()
        return self._total_ownership
    
    @property
    def salary_remaining(self) -> int:
        """Get remaining salary under cap."""
        return self.site.salary_cap - self.total_salary
    
    def get_stacks(self) -> Dict[str, List[Player]]:
        """Get players grouped by team."""
        stacks = {}
        for player in self.players:
            if player.team not in stacks:
                stacks[player.team] = []
            stacks[player.team].append(player)
        return {team: players for team, players in stacks.items() if len(players) > 1}
    
    def get_game_stacks(self) -> List[Set[Player]]:
        """Get players from same game."""
        games = {}
        for player in self.players:
            game_key = tuple(sorted([player.team, player.opponent]))
            if game_key not in games:
                games[game_key] = set()
            games[game_key].add(player)
        return [players for players in games.values() if len(players) > 1]
    
    def similarity_score(self, other: "Lineup") -> float:
        """Calculate similarity to another lineup (0-1)."""
        if not isinstance(other, Lineup):
            raise TypeError("Can only compare to another Lineup")
        
        common_players = set(self.players) & set(other.players)
        return len(common_players) / len(self.players)
    
    def get_percentile_score(self, percentile: float) -> float:
        """Get score at given percentile from simulations."""
        if not self._simulated_scores:
            raise ValueError("No simulated scores available")
        return np.percentile(self._simulated_scores, percentile)
    
    def add_simulated_score(self, score: float):
        """Add a simulated score to the lineup."""
        self._simulated_scores.append(score)
    
    def clear_simulations(self):
        """Clear simulated scores."""
        self._simulated_scores.clear()
    
    def __hash__(self) -> int:
        """Make lineup hashable."""
        return hash(frozenset(self.players))
    
    def __eq__(self, other) -> bool:
        """Check lineup equality based on players."""
        if not isinstance(other, Lineup):
            return False
        return set(self.players) == set(other.players)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Lineup(${self.total_salary}/{self.site.salary_cap}, "
            f"{self.total_projection:.1f}pts, {self.total_ownership:.1f}% own)"
        )
    
    def to_csv_row(self) -> List[str]:
        """Convert lineup to CSV row for DFS site upload."""
        position_order = self.site.get_csv_upload_format()
        if not position_order:
            # Fallback to player IDs in order
            return [p.player_id for p in self.players]
        
        row = []
        used_players = set()
        
        # Map position strings to Position enums
        for pos_str in position_order:
            position = Position(pos_str)
            for player in self.players:
                if player not in used_players and player.is_eligible_for(position):
                    row.append(player.player_id)
                    used_players.add(player)
                    break
        
        return row