"""Simple CSV loader for DFS data."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

from ..models import Player, Position

logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def map_position(pos_str: str, sport: str = 'nfl') -> Optional[Position]:
    """Map position string to Position enum."""
    pos_str = pos_str.upper().strip()
    
    # Direct mapping
    try:
        return Position(pos_str)
    except ValueError:
        pass
    
    # Common aliases
    aliases = {
        'D/ST': Position.DST,
        'DEF': Position.DST,
        'DST': Position.DST,
    }
    
    if pos_str in aliases:
        return aliases[pos_str]
    
    logger.warning(f"Unknown position: {pos_str}")
    return None


def load_draftkings_csv(filepath: str, sport: str = 'nfl') -> List[Player]:
    """
    Load DraftKings CSV export.
    
    Expected columns:
    - Name
    - Position
    - Salary
    - TeamAbbrev
    - AvgPointsPerGame (or similar projection column)
    """
    df = load_csv(filepath)
    
    # Find projection column (DraftKings uses various names)
    proj_columns = ['AvgPointsPerGame', 'Projection', 'FPPG', 'Points']
    proj_col = None
    for col in proj_columns:
        if col in df.columns:
            proj_col = col
            break
    
    if not proj_col:
        logger.warning("No projection column found, using 0")
    
    players = []
    for _, row in df.iterrows():
        position = map_position(row.get('Position', ''), sport)
        if not position:
            continue
        
        # Extract opponent from game info if available
        game_info = row.get('Game Info', '')
        opponent = ''
        if '@' in game_info:
            parts = game_info.split('@')
            team = row.get('TeamAbbrev', '')
            opponent = parts[1].split()[0] if team in parts[0] else parts[0].split()[0]
        
        player = Player(
            player_id=row.get('ID', row.get('Name', '')),
            name=row.get('Name', ''),
            position=position,
            team=row.get('TeamAbbrev', ''),
            opponent=opponent,
            salary=int(row.get('Salary', 0)),
            projection=float(row.get(proj_col, 0)) if proj_col else 0,
            ownership=0,  # Will be loaded separately
            floor=0,  # Will be calculated
            ceiling=0,  # Will be calculated
            std_dev=0,  # Will be calculated
            game_total=0,
            team_total=0,
            spread=0
        )
        
        # Auto-calculate floor/ceiling/std_dev if not provided
        if player.projection > 0:
            player.floor = player.projection * 0.7
            player.ceiling = player.projection * 1.3
            player.std_dev = player.projection * 0.2
        
        players.append(player)
    
    logger.info(f"Loaded {len(players)} players from DraftKings CSV")
    return players


def load_projections_csv(filepath: str, sport: str = 'nfl') -> List[Player]:
    """
    Load a generic projections CSV.
    
    Expected columns (flexible):
    - name or player_name
    - position or pos
    - salary
    - projection or proj or points
    - team
    - opponent or opp
    - ownership or own (optional)
    """
    df = load_csv(filepath)
    
    # Map column names flexibly
    column_map = {
        'name': ['name', 'player_name', 'player', 'Name', 'Player'],
        'position': ['position', 'pos', 'Position', 'Pos'],
        'salary': ['salary', 'sal', 'Salary'],
        'projection': ['projection', 'proj', 'points', 'fpts', 'Projection'],
        'team': ['team', 'tm', 'Team'],
        'opponent': ['opponent', 'opp', 'vs', 'Opponent'],
        'ownership': ['ownership', 'own', 'own%', 'Ownership'],
    }
    
    # Find actual column names
    found_cols = {}
    for key, options in column_map.items():
        for col in df.columns:
            if col in options:
                found_cols[key] = col
                break
    
    if 'name' not in found_cols:
        raise ValueError("No name column found")
    
    players = []
    for _, row in df.iterrows():
        position = map_position(row.get(found_cols.get('position', ''), ''), sport)
        if not position:
            continue
        
        player = Player(
            player_id=str(row.get(found_cols['name'])),
            name=row.get(found_cols['name'], ''),
            position=position,
            team=row.get(found_cols.get('team', ''), ''),
            opponent=row.get(found_cols.get('opponent', ''), ''),
            salary=int(row.get(found_cols.get('salary', 0), 0)),
            projection=float(row.get(found_cols.get('projection', 0), 0)),
            ownership=float(row.get(found_cols.get('ownership', 0), 0)),
            floor=0,
            ceiling=0,
            std_dev=0,
            game_total=0,
            team_total=0,
            spread=0
        )
        
        # Auto-calculate if needed
        if player.projection > 0:
            if player.floor == 0:
                player.floor = player.projection * 0.7
            if player.ceiling == 0:
                player.ceiling = player.projection * 1.3
            if player.std_dev == 0:
                player.std_dev = player.projection * 0.2
        
        players.append(player)
    
    logger.info(f"Loaded {len(players)} players from projections CSV")
    return players


def load_ownership_csv(filepath: str, players: List[Player]) -> None:
    """
    Load ownership projections and update players.
    
    Expected columns:
    - Name or Player
    - Ownership or Own% or Projected%
    """
    df = load_csv(filepath)
    
    # Find columns
    name_col = None
    own_col = None
    
    for col in df.columns:
        if 'name' in col.lower() or 'player' in col.lower():
            name_col = col
        elif 'own' in col.lower() or 'percent' in col.lower():
            own_col = col
    
    if not name_col or not own_col:
        logger.warning("Could not identify name/ownership columns")
        return
    
    # Create lookup dict
    ownership_map = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).lower().strip()
        ownership = float(row[own_col])
        # Handle percentage vs decimal
        if ownership > 1:
            ownership = ownership / 100
        ownership_map[name] = ownership * 100  # Store as percentage
    
    # Update players
    updated = 0
    for player in players:
        name_key = player.name.lower().strip()
        if name_key in ownership_map:
            player.ownership = ownership_map[name_key]
            updated += 1
    
    logger.info(f"Updated ownership for {updated} players")