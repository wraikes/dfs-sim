"""JSON extractor for DFS data with sport-specific subclasses."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re
from abc import ABC, abstractmethod

from ..models import Player, Position

logger = logging.getLogger(__name__)


class BaseJsonExtractor(ABC):
    """Abstract base class for sport-specific JSON extractors."""
    
    def __init__(self):
        """Initialize base extractor."""
        pass
    
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
        
        # Merge data (sport-specific logic)
        final_df = self._merge_dataframes(players_df, ownership_df, stats_df)
        
        # Convert to Player objects
        return self._dataframe_to_players(final_df)
    
    @abstractmethod
    def _extract_player_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract base player information. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _extract_matchup_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract matchup data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _merge_dataframes(self, players_df: pd.DataFrame, 
                         ownership_df: pd.DataFrame, 
                         stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merge dataframes using sport-specific logic."""
        pass
    
    @abstractmethod
    def _map_position(self, pos_str: str) -> Optional[Position]:
        """Map position string to Position enum."""
        pass
    
    @abstractmethod
    def _create_player(self, row: pd.Series, position: Position, metadata: Dict) -> Player:
        """Create Player object with sport-specific logic."""
        pass
    
    def _extract_ownership_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract ownership projections (common implementation)."""
        try:
            ownership_section = json_data.get('Ownership', {})
            projected_data = ownership_section.get('Projected', {})
            
            if not projected_data:
                return pd.DataFrame()
            
            # Get first slate's ownership data
            slate_data = None
            for data in projected_data.values():
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
    
    def _dataframe_to_players(self, df: pd.DataFrame) -> List[Player]:
        """Convert DataFrame to Player objects (common implementation)."""
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
                              'team', 'opponent', 'projection', 'ownership',
                              'floor', 'ceiling', 'points_scored']:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[col] = val
            
            # Create Player object (sport-specific logic)
            player = self._create_player(row, position, metadata)
            players.append(player)
        
        logger.info(f"Extracted {len(players)} players from JSON")
        return players
    
    def load_json_file(self, filepath: str) -> List[Player]:
        """Load JSON file and extract players."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        with open(path, 'r') as f:
            json_data = json.load(f)
        
        return self.extract_from_json(json_data)
    
    def extract_and_save_csv(self, filepath: str, output_path: Optional[str] = None) -> str:
        """Extract players from JSON and save to CSV."""
        # Load and extract players
        players = self.load_json_file(filepath)
        
        # Convert to DataFrame
        player_data = []
        for player in players:
            # Flatten player data and metadata
            row = {
                'player_id': player.player_id,
                'name': player.name,
                'position': player.position.value,
                'team': player.team,
                'opponent': player.opponent,
                'salary': player.salary,
                'projection': player.projection,
                'floor': player.floor,
                'ceiling': player.ceiling,
                'ownership': player.ownership,
                'std_dev': player.std_dev,
                'value': player.value,
            }
            # Add all metadata
            row.update(player.metadata)
            player_data.append(row)
        
        df = pd.DataFrame(player_data)
        
        # Determine output path - save to /csv/ folder if JSON is in organized structure
        if output_path is None:
            input_path = Path(filepath)
            
            # Check if JSON is in /json/ folder and save CSV to /csv/ folder
            if input_path.parent.name == 'json':
                csv_dir = input_path.parent.parent / 'csv'
                csv_dir.mkdir(exist_ok=True)
                output_path = csv_dir / f"{input_path.stem}_extracted.csv"
            else:
                # Fallback to same directory
                output_path = input_path.parent / f"{input_path.stem}_extracted.csv"
        
        # Save CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(players)} players to {output_path}")
        
        return str(output_path)


class MMAJsonExtractor(BaseJsonExtractor):
    """MMA-specific JSON extractor."""
    
    def _extract_player_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract MMA fighter data."""
        try:
            salary_data = json.loads(json_data['SalaryContainerJson'])
            players = salary_data.get('Salaries', [])
            
            player_records = []
            for player in players:
                record = self._extract_mma_player(player)
                player_records.append(record)
            
            return pd.DataFrame(player_records)
            
        except Exception as e:
            logger.error(f"Error extracting MMA player data: {e}")
            return pd.DataFrame()
    
    def _extract_mma_player(self, player: Dict) -> Dict:
        """Extract MMA-specific player data."""
        record = {
            'player_id': player.get('PID'),
            'salary_id': player.get('Id'),
            'name': player.get('Name'),
            'position': player.get('POS'),
            'salary': player.get('SAL'),
            'projection': player.get('PP', 0),
            'ceiling': player.get('Ceil', 0),
            'floor': player.get('Floor', 0),
            'confidence': player.get('Conf', 0),
            'stars': player.get('Stars', 0),
            'team': player.get('PTEAM', ''),
            'opponent': player.get('OTEAM', ''),
            'game_info': player.get('GI', ''),
        }
        
        # Parse Notes field for fight stats
        if player.get('Notes'):
            try:
                notes = json.loads(player['Notes'])
                for note in notes:
                    text = note.get('Note', '')
                    # Extract numeric values from notes
                    if 'Sig Strikes Landed/min' in text:
                        match = re.search(r'[\d.]+', text)
                        if match:
                            record['ss_landed_min'] = float(match.group())
                    elif 'Sig Strikes Attempted/min' in text:
                        match = re.search(r'[\d.]+', text)
                        if match:
                            record['ss_attempted_min'] = float(match.group())
                    elif 'Takedowns/min' in text:
                        match = re.search(r'[\d.]+', text)
                        if match:
                            record['takedowns_min'] = float(match.group())
                    elif 'Knockdowns/min' in text:
                        match = re.search(r'[\d.]+', text)
                        if match:
                            record['knockdowns_min'] = float(match.group())
                    elif 'Submissions Attempted' in text:
                        match = re.search(r'[\d.]+', text)
                        if match:
                            record['submissions_min'] = float(match.group())
                    elif 'win chance' in text.lower():
                        match = re.search(r'[\d.]+', text)
                        if match:
                            record['win_pct_notes'] = float(match.group())
            except (json.JSONDecodeError, TypeError):
                pass
        
        return record
    
    def _extract_matchup_data(self, json_data: Dict) -> pd.DataFrame:
        """Extract MMA-specific matchup data using salary ID matching."""
        if 'MatchupData' not in json_data:
            return pd.DataFrame()
            
        stats_by_salary = {}
        
        for table in json_data['MatchupData']:
            section_name = table.get("Name", "").lower().strip()
            col_names = table.get('Columns', [])
            
            # Define section suffixes for MMA
            suffix_map = {
                'fight': 'fight',
                'last 5 fights (off)': 'l5off',
                'last 5 fights (def)': 'l5def',
                'opp last 5 (off)': 'opp_l5off',
                'opp last 5 (def)': 'opp_l5def',
                'career (off)': 'career_off',
                'career (def)': 'career_def',
            }
            suffix = suffix_map.get(section_name, section_name.replace(' ', '_'))
            
            # Extract player stats using SID (salary ID)
            for player in table.get('PlayerMatchups', []):
                salary_id = player.get('SID')  # MMA uses SID for salary ID
                if not salary_id:
                    continue
                
                if salary_id not in stats_by_salary:
                    stats_by_salary[salary_id] = {
                        'salary_id': salary_id,
                        'player_id': player.get('PlayerId')
                    }
                
                # Extract values
                values = player.get('Values', [])
                for col, val in zip(col_names, values):
                    col_name = f"{col}_{suffix}".replace(' ', '_').replace('/', '_').replace('%', 'pct')
                    stats_by_salary[salary_id][col_name] = val
        
        return pd.DataFrame(list(stats_by_salary.values()))
    
    def _merge_dataframes(self, players_df: pd.DataFrame, 
                         ownership_df: pd.DataFrame, 
                         stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merge MMA dataframes using salary_id."""
        final_df = players_df
        
        # MMA: merge on salary_id for both ownership and stats
        if not ownership_df.empty:
            final_df = final_df.merge(ownership_df[['salary_id', 'ownership', 'doubleup']], 
                                    on='salary_id', how='left')
        if not stats_df.empty:
            stats_merge = stats_df.drop(columns=['player_id'], errors='ignore')
            final_df = final_df.merge(stats_merge, on='salary_id', how='left')
        
        return final_df
    
    def _map_position(self, _pos_str: str) -> Optional[Position]:  # noqa: ARG002
        """Map MMA position string to Position enum."""
        # All MMA fighters use same position regardless of pos_str
        return Position.FIGHTER
    
    def _create_player(self, row: pd.Series, position: Position, metadata: Dict) -> Player:
        """Create MMA Player object."""
        player = Player(
            player_id=str(row.get('player_id', row.get('name', ''))),
            name=row.get('name', ''),
            position=position,
            team=row.get('team', ''),
            opponent=row.get('opponent', ''),
            salary=int(row.get('salary', 0)),
            projection=float(row.get('projection', row.get('points_scored', 0))),
            ownership=float(row.get('ownership', 0)),
            floor=float(row.get('floor', 0)),
            ceiling=float(row.get('ceiling', 0)),
            std_dev=0,  # Will be calculated
            game_total=0,  # Not applicable for MMA
            team_total=0,  # Not applicable for MMA
            spread=float(row.get('Vegas_Odds_fight', 0)),  # Use vegas odds as spread
            metadata=metadata
        )
        
        # Calculate std_dev
        if player.projection > 0:
            variance_mult = 0.5  # High variance for MMA
            # Only calculate floor/ceiling if not provided
            if player.floor == 0:
                player.floor = player.projection * 0.4
            if player.ceiling == 0:
                player.ceiling = player.projection * 2.0
            
            player.std_dev = player.projection * variance_mult
        
        return player


# Factory function for backward compatibility
def JsonExtractor(sport: str) -> BaseJsonExtractor:
    """Factory function to create sport-specific extractors."""
    if sport.lower() == 'mma':
        return MMAJsonExtractor()
    else:
        raise ValueError(f"Sport '{sport}' not yet implemented. Available: mma")