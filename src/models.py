"""Core data models for DFS simulation."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Player:
    """Represents a DFS player with projections and metadata."""
    
    # Core identification
    player_id: int
    name: str
    position: str
    team: str
    
    # DFS specifics
    salary: int
    projection: float
    floor: float
    ceiling: float
    std_dev: float
    ownership: float
    value: float
    
    # Additional attributes (can be sport-specific)
    opponent: Optional[str] = None
    game_info: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # Newsletter adjustments
    target_multiplier: float = 1.0
    fade_multiplier: float = 1.0
    volatile: bool = False
    
    def __post_init__(self):
        """Initialize metadata dict if not provided."""
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def points_per_dollar(self) -> float:
        """Calculate points per dollar efficiency."""
        return self.projection / self.salary * 1000 if self.salary > 0 else 0.0
    
    @property
    def ceiling_upside(self) -> float:
        """Calculate ceiling upside vs projection."""
        return self.ceiling - self.projection
        
    @property
    def floor_downside(self) -> float:
        """Calculate floor downside vs projection.""" 
        return self.projection - self.floor