"""Newsletter processor for DFS player adjustments."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from ..models import Player

logger = logging.getLogger(__name__)


@dataclass
class PlayerModifier:
    """Represents a modifier for a specific player."""
    name: str
    modifier_type: str  # 'boost', 'fade', 'lock', 'avoid'
    value: float  # Multiplier for projections (e.g., 1.2 for 20% boost)
    reason: str  # Why this modifier is applied
    source: str  # Newsletter source


class NewsletterProcessor:
    """Process newsletter content and apply modifiers to players."""
    
    def __init__(self, sport: str):
        """
        Initialize processor for specific sport.
        
        Args:
            sport: Sport name
        """
        self.sport = sport.lower()
        self.modifiers: List[PlayerModifier] = []
        
        # Default modifier values by type
        self.modifier_defaults = {
            'lock': 1.3,     # 30% boost for must-play recommendations
            'boost': 1.15,   # 15% boost for positive mentions
            'fade': 0.85,    # 15% reduction for concerns
            'avoid': 0.7,    # 30% reduction for avoid recommendations
        }
    
    def add_modifier(self, player_name: str, modifier_type: str, 
                    value: Optional[float] = None, reason: str = "", 
                    source: str = "newsletter"):
        """
        Add a modifier for a specific player.
        
        Args:
            player_name: Player's name
            modifier_type: Type of modifier (boost, fade, lock, avoid)
            value: Custom multiplier value (uses default if None)
            reason: Explanation for the modifier
            source: Source of the recommendation
        """
        if modifier_type not in self.modifier_defaults:
            logger.warning(f"Unknown modifier type '{modifier_type}'. Using 'boost'.")
            modifier_type = 'boost'
        
        if value is None:
            value = self.modifier_defaults[modifier_type]
        
        modifier = PlayerModifier(
            name=player_name.lower().strip(),
            modifier_type=modifier_type,
            value=value,
            reason=reason,
            source=source
        )
        
        self.modifiers.append(modifier)
        logger.info(f"Added {modifier_type} modifier for {player_name}: {value:.2f}x")
    
    def parse_newsletter_text(self, newsletter_text: str, source: str = "newsletter"):
        """
        Parse newsletter text and extract player recommendations.
        
        Args:
            newsletter_text: Raw newsletter text
            source: Newsletter source name
        """
        if not newsletter_text:
            return
        
        lines = newsletter_text.strip().split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Parse different recommendation formats
            if 'lock:' in line_lower or 'core play:' in line_lower:
                player = self._extract_player_name(line)
                if player:
                    self.add_modifier(player, 'lock', source=source, reason=line.strip())
            
            elif 'fade:' in line_lower or 'avoid:' in line_lower:
                player = self._extract_player_name(line)
                if player:
                    modifier_type = 'avoid' if 'avoid:' in line_lower else 'fade'
                    self.add_modifier(player, modifier_type, source=source, reason=line.strip())
            
            elif 'boost:' in line_lower or 'target:' in line_lower:
                player = self._extract_player_name(line)
                if player:
                    self.add_modifier(player, 'boost', source=source, reason=line.strip())
    
    def _extract_player_name(self, line: str) -> Optional[str]:
        """Extract player name from a line of text."""
        # Simple extraction - look for name after colon
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) > 1:
                # Clean up the name part
                name = parts[1].strip()
                # Remove common prefixes/suffixes
                name = name.replace('(', '').replace(')', '')
                name = name.split('-')[0].strip()  # Take first part if dash-separated
                name = name.split(',')[0].strip()  # Take first part if comma-separated
                
                if len(name) > 2 and name.replace(' ', '').replace('.', '').isalpha():
                    return name
        
        return None
    
    def apply_modifiers(self, players: List[Player]) -> List[Player]:
        """
        Apply newsletter modifiers to players.
        
        Args:
            players: List of Player objects
            
        Returns:
            List of Player objects with modifiers applied
        """
        if not self.modifiers:
            logger.info("No modifiers to apply")
            return players
        
        # Create lookup for quick matching
        modifier_map = {}
        for modifier in self.modifiers:
            modifier_map[modifier.name] = modifier
        
        modified_count = 0
        for player in players:
            player_key = player.name.lower().strip()
            
            # Try exact match first
            if player_key in modifier_map:
                modifier = modifier_map[player_key]
                player.target_multiplier = modifier.value
                player.metadata['modifier_type'] = modifier.modifier_type
                player.metadata['modifier_reason'] = modifier.reason
                modified_count += 1
                continue
            
            # Try partial matching
            for mod_name, modifier in modifier_map.items():
                if (len(mod_name) > 3 and mod_name in player_key) or \
                   (len(player_key) > 3 and player_key in mod_name):
                    player.target_multiplier = modifier.value
                    player.metadata['modifier_type'] = modifier.modifier_type
                    player.metadata['modifier_reason'] = modifier.reason
                    modified_count += 1
                    break
        
        logger.info(f"Applied modifiers to {modified_count} players")
        return players
    
    def get_summary(self) -> str:
        """Get summary of all modifiers."""
        if not self.modifiers:
            return "No modifiers applied"
        
        summary = f"Newsletter Modifiers ({len(self.modifiers)} total):\n"
        
        by_type = {}
        for modifier in self.modifiers:
            if modifier.modifier_type not in by_type:
                by_type[modifier.modifier_type] = []
            by_type[modifier.modifier_type].append(modifier)
        
        for mod_type, mods in by_type.items():
            summary += f"\n{mod_type.upper()} ({len(mods)}):\n"
            for mod in mods:
                summary += f"  - {mod.name} ({mod.value:.2f}x)\n"
        
        return summary