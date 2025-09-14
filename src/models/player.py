"""Player model for DFS simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class Position(Enum):
    """DFS player positions across all sports."""
    
    # NFL
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"  # Kicker
    DST = "DST"
    FLEX = "FLEX"  # RB/WR/TE
    SUPER_FLEX = "SUPER_FLEX"  # Any position including QB
    
    # NBA
    PG = "PG"
    SG = "SG"
    SF = "SF"
    PF = "PF"
    C = "C"
    G = "G"  # Guard (PG/SG)
    F = "F"  # Forward (SF/PF)
    UTIL = "UTIL"  # Any NBA position
    
    # MLB
    P = "P"  # Pitcher
    C_MLB = "C"  # Catcher (aliased to avoid conflict)
    FB = "1B"  # First Base
    SB = "2B"  # Second Base
    TB = "3B"  # Third Base
    SS = "SS"  # Shortstop
    OF = "OF"  # Outfield
    
    # NHL
    C_NHL = "C"  # Center (aliased to avoid conflict)
    W = "W"  # Wing
    D = "D"  # Defense
    G_NHL = "G"  # Goalie (aliased to avoid conflict)
    SKATER = "SKATER"  # Any non-goalie
    
    # PGA
    GOLFER = "GOLFER"
    
    # NASCAR
    DRIVER = "DRIVER"
    
    # MMA
    FIGHTER = "FIGHTER"


@dataclass
class Player:
    """Represents a DFS player with projections and metadata."""
    
    player_id: str
    name: str
    position: Position
    team: str
    opponent: str
    salary: int
    projection: float
    ownership: float
    floor: float
    ceiling: float
    std_dev: float
    game_total: float
    team_total: float
    spread: float
    
    # Optional newsletter adjustments
    target_multiplier: float = 1.0
    fade_multiplier: float = 1.0
    volatile: bool = False
    
    # Sport-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Calculated fields
    value: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.value = self.projection / (self.salary / 1000) if self.salary > 0 else 0
        
    @property
    def adjusted_projection(self) -> float:
        """Get projection after newsletter adjustments."""
        return self.projection * self.target_multiplier * self.fade_multiplier
    
    @property
    def adjusted_std_dev(self) -> float:
        """Get standard deviation adjusted for volatility."""
        return self.std_dev * (1.5 if self.volatile else 1.0)
    
    def is_eligible_for(self, position: Position) -> bool:
        """Check if player is eligible for a roster position."""
        if position == self.position:
            return True
        
        # NFL FLEX positions
        if position == Position.FLEX and self.position in [Position.RB, Position.WR, Position.TE]:
            return True
        if position == Position.SUPER_FLEX and self.position in [Position.QB, Position.RB, Position.WR, Position.TE]:
            return True
        
        # NBA utility positions
        if position == Position.G and self.position in [Position.PG, Position.SG]:
            return True
        if position == Position.F and self.position in [Position.SF, Position.PF]:
            return True
        if position == Position.UTIL and self.position in [Position.PG, Position.SG, Position.SF, Position.PF, Position.C]:
            return True
        
        # NHL SKATER position
        if position == Position.SKATER and self.position in [Position.C_NHL, Position.W, Position.D]:
            return True
        
        return False
    
    def __hash__(self) -> int:
        """Make player hashable for use in sets."""
        return hash(self.player_id)
    
    def __eq__(self, other) -> bool:
        """Check equality based on player_id."""
        if not isinstance(other, Player):
            return False
        return self.player_id == other.player_id
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Player({self.name}, {self.position.value}, ${self.salary}, {self.projection:.1f}pts)"