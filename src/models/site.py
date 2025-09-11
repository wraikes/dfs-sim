"""DFS site configuration model."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from .player import Position


class SiteName(Enum):
    """Supported DFS sites."""
    DRAFTKINGS = "DraftKings"
    FANDUEL = "FanDuel"  # Future support


@dataclass
class Site:
    """Configuration for a specific DFS site."""
    
    name: SiteName
    sport: str
    salary_cap: int
    roster_slots: Dict[Position, int]
    max_players_per_team: int = 999  # Some sites limit players from same team
    
    # Scoring settings
    scoring_settings: Dict[str, float] = None
    
    def __post_init__(self):
        """Set site-specific defaults."""
        if self.name == SiteName.DRAFTKINGS:
            self._set_draftkings_defaults()
    
    def _set_draftkings_defaults(self):
        """Set DraftKings-specific configurations."""
        if self.sport == 'nfl':
            self.salary_cap = 50000
            self.roster_slots = {
                Position.QB: 1,
                Position.RB: 2,
                Position.WR: 3,
                Position.TE: 1,
                Position.FLEX: 1,  # RB/WR/TE
                Position.DST: 1
            }
            self.max_players_per_team = 999  # No limit for NFL
            
        elif self.sport == 'nba':
            self.salary_cap = 50000
            self.roster_slots = {
                Position.PG: 1,
                Position.SG: 1,
                Position.SF: 1,
                Position.PF: 1,
                Position.C: 1,
                Position.G: 1,    # Guard (PG/SG)
                Position.F: 1,    # Forward (SF/PF)
                Position.UTIL: 1  # Any position
            }
            self.max_players_per_team = 999
            
        elif self.sport == 'mlb':
            self.salary_cap = 50000
            self.roster_slots = {
                Position.P: 2,
                Position.C_MLB: 1,
                Position.FB: 1,
                Position.SB: 1,
                Position.TB: 1,
                Position.SS: 1,
                Position.OF: 3
            }
            self.max_players_per_team = 5  # MLB has team limits
            
        elif self.sport == 'pga':
            self.salary_cap = 50000
            self.roster_slots = {
                Position.GOLFER: 6
            }
            self.max_players_per_team = 999
            
        elif self.sport == 'nascar':
            self.salary_cap = 50000
            self.roster_slots = {
                Position.DRIVER: 6
            }
            self.max_players_per_team = 999
            
        elif self.sport == 'mma':
            self.salary_cap = 50000
            self.roster_slots = {
                Position.FIGHTER: 6
            }
            self.max_players_per_team = 999
    
    def validate_lineup(self, players: List) -> tuple[bool, str]:
        """
        Validate a lineup meets site requirements.
        
        Args:
            players: List of Player objects
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check salary cap
        total_salary = sum(p.salary for p in players)
        if total_salary > self.salary_cap:
            return False, f"Salary ${total_salary} exceeds cap ${self.salary_cap}"
        
        # Check roster requirements
        position_counts = {}
        for player in players:
            position_counts[player.position] = position_counts.get(player.position, 0) + 1
        
        for position, required in self.roster_slots.items():
            eligible_count = sum(1 for p in players if p.is_eligible_for(position))
            if eligible_count < required:
                return False, f"Need {required} {position.value}, have {eligible_count}"
        
        # Check team limits
        team_counts = {}
        for player in players:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
        
        for team, count in team_counts.items():
            if count > self.max_players_per_team:
                return False, f"Too many players from {team}: {count} (max: {self.max_players_per_team})"
        
        return True, "Valid lineup"
    
    def get_csv_upload_format(self) -> List[str]:
        """Get the position order for CSV upload to this site."""
        if self.name == SiteName.DRAFTKINGS:
            if self.sport == 'nfl':
                return ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
            elif self.sport == 'nba':
                return ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            # Add other sports as needed
        
        return []
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Site({self.name.value}, {self.sport}, ${self.salary_cap} cap)"


# Preset configurations
DRAFTKINGS_NFL = Site(
    name=SiteName.DRAFTKINGS,
    sport='nfl',
    salary_cap=50000,
    roster_slots={}  # Will be set in __post_init__
)

DRAFTKINGS_NBA = Site(
    name=SiteName.DRAFTKINGS,
    sport='nba',
    salary_cap=50000,
    roster_slots={}  # Will be set in __post_init__
)