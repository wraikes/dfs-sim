"""JSON extractor for DFS data (MMA, NASCAR, NFL focus)."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..models import Player, Position

logger = logging.getLogger(__name__)


class JsonExtractor:
    """Extract key fields from JSON data to Player objects."""
    
    def __init__(self, sport: str):
        """
        Initialize extractor for specific sport.
        
        Args:
            sport: Sport name (mma, nascar, nfl)
        """
        self.sport = sport.lower()
        
        # Key fields to extract by sport
        self.field_configs = {
            'mma': {
                'key_stats': [
                    'Vegas Odds', 'ITD Odds', 'SS Landed/Min', 'Strike Acc%',
                    'Takedowns/Min', 'SS Taken/Min', 'Takedown Def%',
                    'Win%', 'FPPG', 'Finish Rate'
                ],
                'matchup_sections': {
                    'vegas odds': 'vegas',
                    'last 5 fights (off)': 'l5_off',
                    'last 5 fights (def)': 'l5_def',
                    'fight': 'fight_info'
                }
            },
            'nascar': {
                'key_stats': [
                    'Vegas Odds', 'Q', 'Best Lap', 'Practice Speed',
                    'Avg Finish', 'Top 5s', 'Laps Led/Race', 'FPPG',
                    'Start Pos', 'Proj Finish'
                ],
                'matchup_sections': {
                    'driver season': 'season',
                    'practice & qualifying': 'practice',
                    'track type': 'track',
                    'vegas odds': 'vegas'
                }
            },
            'nfl': {
                'key_stats': [
                    'Targets/G', 'Red Zone Tgt', 'FPPG Allowed',
                    'Team Total', 'Game Total', 'FPPG'
                ],
                'matchup_sections': {
                    'last 4 games': 'recent4',
                    'last 5 games': 'recent5',
                    'last 3 games': 'recent3',
                    'vs opponent': 'vs_opp',
                    'vegas': 'vegas'
                }
            }
        }
        
        self.config = self.field_configs.get(sport, {})
    
    def extract_from_json(self, json_data: Dict) -> List[Player]:
        """
        Extract player data from JSON.
        
        Args:
            json_data: Raw JSON data
            
        Returns:
            List of Player objects
        """
        if not json_data or 'SalaryContainerJson' not in json_data:
            logger.warning("Invalid or empty JSON data")
            return []
        
        # Extract base player data
        players_df = self._extract_player_data(json_data)
        
        # Extract ownership data
        ownership_df = self._extract_ownership_data(json_data)
        
        # Extract matchup/stat data
        stats_df = self._extract_matchup_data(json_data)
        
        # Merge data
        final_df = players_df
        if not ownership_df.empty:
            ownership_merge = ownership_df.drop(columns=['salary_id'], errors='ignore')
            final_df = final_df.merge(ownership_merge, on='player_id', how='left')
        if not stats_df.empty:
            final_df = final_df.merge(stats_df, on='player_id', how='left')
        
        # Convert to Player objects
        return self._dataframe_to_players(final_df)
    
    def _extract_player_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract base player information."""
        try:
            salary_data = json.loads(json_data['SalaryContainerJson'])
            players = salary_data.get('Salaries', [])
            
            player_records = []
            for player in players:
                record = {
                    'player_id': player.get('PID'),
                    'name': player.get('PN') or player.get('Name'),
                    'position': player.get('PO') or player.get('POS'),
                    'salary': player.get('SAL') or player.get('S'),
                    'points_scored': player.get('PS', 0),
                    'salary_id': player.get('Id'),
                    'team': player.get('TN') or player.get('PTEAM', ''),
                    'opponent': player.get('OP') or player.get('OTEAM', ''),
                    'game_time': player.get('GT', ''),
                    'projection': player.get('Projection', 0),
                }
                player_records.append(record)
            
            return pd.DataFrame(player_records)
            
        except Exception as e:
            logger.error(f"Error extracting player data: {e}")
            return pd.DataFrame()
    
    def _extract_ownership_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract ownership projections."""
        try:
            ownership_section = json_data.get('Ownership', {})
            projected_data = ownership_section.get('Projected', {})
            
            if not projected_data:
                return pd.DataFrame()
            
            # Get first slate's ownership data
            slate_data = None
            for slate_id, data in projected_data.items():
                slate_data = data
                break
            
            if not slate_data:
                return pd.DataFrame()
            
            ownership_records = []
            for entry in slate_data:
                record = {
                    'player_id': entry.get('PlayerId'),
                    'salary_id': entry.get('SalaryId'),
                    'ownership': entry.get('Owned', 0),
                    'doubleup': entry.get('DoubleUp', 0)
                }
                ownership_records.append(record)
            
            return pd.DataFrame(ownership_records)
            
        except Exception as e:
            logger.error(f"Error extracting ownership data: {e}")
            return pd.DataFrame()
    
    def _extract_matchup_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract matchup and statistical data."""
        if 'MatchupData' not in json_data:
            return pd.DataFrame()
        
        stats_by_player = {}
        
        for table in json_data['MatchupData']:
            section_name = table.get("Name", "").lower().strip()
            col_names = table.get('Columns', [])
            
            # Get section suffix
            suffix = self.config.get('matchup_sections', {}).get(
                section_name, 'other'
            )
            
            # Extract player stats
            for player in table.get('PlayerMatchups', []):
                player_id = player.get('PlayerId')
                if not player_id:
                    continue
                
                if player_id not in stats_by_player:
                    stats_by_player[player_id] = {'player_id': player_id}
                
                # Extract values
                values = player.get('Values', [])
                for col, val in zip(col_names, values):
                    # Store key stats based on sport config
                    if self.config and 'key_stats' in self.config:
                        if any(key in col for key in self.config['key_stats']):
                            col_name = f"{col}_{suffix}".replace(' ', '_').replace('/', '_')
                            stats_by_player[player_id][col_name] = val
                    else:
                        # Take all stats if no config
                        col_name = f"{col}_{suffix}".replace(' ', '_').replace('/', '_')
                        stats_by_player[player_id][col_name] = val
        
        return pd.DataFrame(list(stats_by_player.values()))
    
    def _dataframe_to_players(self, df: pd.DataFrame) -> List[Player]:
        """Convert DataFrame to Player objects."""
        players = []
        
        for _, row in df.iterrows():
            # Map position
            position = self._map_position(row.get('position', ''))
            if not position:
                continue
            
            # Extract key stats for metadata
            metadata = {}
            for col in df.columns:
                if col not in ['player_id', 'name', 'position', 'salary', 
                              'team', 'opponent', 'projection', 'ownership']:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[col] = val
            
            # Create Player object
            player = Player(
                player_id=str(row.get('player_id', row.get('name', ''))),
                name=row.get('name', ''),
                position=position,
                team=row.get('team', ''),
                opponent=row.get('opponent', ''),
                salary=int(row.get('salary', 0)),
                projection=float(row.get('projection', row.get('points_scored', 0))),
                ownership=float(row.get('ownership', 0)),
                floor=0,  # Will be calculated
                ceiling=0,  # Will be calculated
                std_dev=0,  # Will be calculated
                game_total=float(metadata.get('Game_Total_vegas', 0)),
                team_total=float(metadata.get('Team_Total_vegas', 0)),
                spread=float(metadata.get('Spread_vegas', 0)),
                metadata=metadata
            )
            
            # Auto-calculate floor/ceiling/std_dev
            if player.projection > 0:
                # Sport-specific variance
                if self.sport == 'mma':
                    variance_mult = 0.5  # High variance
                elif self.sport == 'nascar':
                    variance_mult = 0.45
                else:  # NFL
                    variance_mult = 0.25
                
                player.floor = player.projection * 0.6
                player.ceiling = player.projection * 1.5
                player.std_dev = player.projection * variance_mult
            
            players.append(player)
        
        logger.info(f"Extracted {len(players)} players from JSON")
        return players
    
    def _map_position(self, pos_str: str) -> Optional[Position]:
        """Map position string to Position enum."""
        pos_str = str(pos_str).upper().strip()
        
        # Sport-specific mappings
        if self.sport == 'mma':
            return Position.FIGHTER
        elif self.sport == 'nascar':
            return Position.DRIVER
        elif self.sport == 'nfl':
            # NFL position mapping
            mappings = {
                'QB': Position.QB,
                'RB': Position.RB,
                'WR': Position.WR,
                'TE': Position.TE,
                'DST': Position.DST,
                'D/ST': Position.DST,
                'DEF': Position.DST,
            }
            return mappings.get(pos_str)
        
        return None
    
    def load_json_file(self, filepath: str) -> List[Player]:
        """
        Load JSON file and extract players.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of Player objects
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        with open(path, 'r') as f:
            json_data = json.load(f)
        
        return self.extract_from_json(json_data)