"""JSON extractor for DFS data with sport-specific subclasses."""

import json
import glob
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
    
    def extract_and_save_csv(self, filepath: str, output_path: Optional[str] = None, apply_newsletter: bool = True) -> str:
        """Extract players from JSON and save to CSV with optional newsletter signals."""
        # Load and extract players
        players = self.load_json_file(filepath)
        
        # Apply newsletter signals if requested and signals file exists
        if apply_newsletter:
            players = self._apply_newsletter_signals(players, filepath)
        
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
                'std_dev': player.std_dev,
                'value': player.value,
            }
            # Add all metadata (including newsletter signals)
            row.update(player.metadata)
            player_data.append(row)
        
        df = pd.DataFrame(player_data)
        
        # Add newsletter columns if they don't exist (for players without signals)
        newsletter_cols = ['newsletter_signal', 'newsletter_confidence', 'newsletter_reason', 
                          'ceiling_delta_applied', 'ownership_delta_applied', 
                          'newsletter_target_mult', 'newsletter_fade_mult',
                          'original_projection', 'projection_modifier', 'updated_projection',
                          'original_floor', 'floor_modifier', 'updated_floor',
                          'original_ceiling', 'ceiling_modifier', 'updated_ceiling',
                          'original_ownership', 'ownership_modifier', 'updated_ownership']
        
        for col in newsletter_cols:
            if col not in df.columns:
                if 'mult' in col:
                    df[col] = 1.0
                elif 'delta' in col or 'modifier' in col:
                    df[col] = 0.0
                elif col == 'newsletter_confidence':
                    df[col] = 0.0
                elif col.startswith('original_'):
                    # Map original columns to their base fields
                    base_field = col.replace('original_', '')
                    df[col] = df.get(base_field, 0.0)
                elif col.startswith('updated_'):
                    # Updated columns default to same as base field if no signals
                    base_field = col.replace('updated_', '')
                    df[col] = df.get(base_field, 0.0)
                else:
                    df[col] = 'neutral' if col == 'newsletter_signal' else ''
        
        # Fill NaN values for newsletter columns
        df['newsletter_signal'] = df['newsletter_signal'].fillna('neutral')
        df['newsletter_confidence'] = df['newsletter_confidence'].fillna(0.0)
        df['newsletter_reason'] = df['newsletter_reason'].fillna('')
        df['ceiling_delta_applied'] = df['ceiling_delta_applied'].fillna(0.0)
        df['ownership_delta_applied'] = df['ownership_delta_applied'].fillna(0.0)
        df['newsletter_target_mult'] = df['newsletter_target_mult'].fillna(1.0)
        df['newsletter_fade_mult'] = df['newsletter_fade_mult'].fillna(1.0)
        
        # Handle transparency columns - for players without signals, original = updated
        transparency_fields = ['projection', 'floor', 'ceiling', 'ownership']
        
        for field in transparency_fields:
            original_col = f'original_{field}'
            modifier_col = f'{field}_modifier'
            updated_col = f'updated_{field}'
            
            if field in df.columns:
                df[original_col] = df[original_col].fillna(df[field])
                df[modifier_col] = df[modifier_col].fillna(0.0)
                df[updated_col] = df[updated_col].fillna(df[original_col])
            else:
                df[original_col] = df[original_col].fillna(0.0)
                df[modifier_col] = df[modifier_col].fillna(0.0)
                df[updated_col] = df[updated_col].fillna(df[original_col])
        
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
        
        # Add MMA-specific calculated columns (after all other processing)
        if hasattr(self, '_calculate_itd_probability'):  # MMA extractor
            # Add ML odds column from Vegas_Odds_fight if available
            if 'Vegas_Odds_fight' in df.columns:
                df['ml_odds'] = df['Vegas_Odds_fight'].copy()
            else:
                df['ml_odds'] = 0
            
            # Calculate ITD-adjusted ceiling using sport_rules.py formula
            # adj = CEIL * (1 - 0.20 * ownership) * (1 + 0.4 * ITD_prob)
            df['itd_adjusted_ceiling'] = 0.0
            
            for idx, row in df.iterrows():
                ownership_decimal = row.get('updated_ownership', row.get('ownership', 0)) / 100.0
                itd_prob = row.get('itd_probability', 0.35)  # Default to UFC average
                base_ceiling = row.get('updated_ceiling', row.get('ceiling', 0))
                
                ownership_factor = max(0, 1 - 0.20 * ownership_decimal)
                itd_factor = 1 + 0.4 * itd_prob
                df.at[idx, 'itd_adjusted_ceiling'] = base_ceiling * ownership_factor * itd_factor
        
        # Save CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(players)} players to {output_path}")
        
        return str(output_path)
    
    def _apply_newsletter_signals(self, players: List[Player], json_filepath: str) -> List[Player]:
        """Apply newsletter signals to players if signals file exists."""
        try:
            from .llm_newsletter_processor import LLMNewsletterProcessor
            
            # Find signals JSON file
            json_path = Path(json_filepath)
            if json_path.parent.name == 'json':
                # Look for signals file in same /json/ folder
                signals_pattern = json_path.parent / "*signals.json"
                signals_files = glob.glob(str(signals_pattern))
                
                if not signals_files:
                    logger.info("No newsletter signals file found")
                    return players
                
                # Use first signals file found
                signals_path = signals_files[0]
                logger.info(f"Applying newsletter signals from: {signals_path}")
                
                # Load and apply signals
                with open(signals_path, 'r') as f:
                    signals_data = json.load(f)
                
                # Apply signals to each player
                modified_count = 0
                for player in players:
                    for player_signal in signals_data.get('players', []):
                        if self._names_match(player.name, player_signal.get('name', '')):
                            # Apply adjustments
                            signal = player_signal.get('signal', 'neutral')
                            confidence = player_signal.get('confidence', 0.5)
                            ceiling_delta = player_signal.get('ceiling_delta', 0.0)
                            ownership_delta = player_signal.get('ownership_delta', 0.0)
                            reason = player_signal.get('reason', '')
                            
                            # Store original values before any modifications
                            original_projection = player.projection
                            original_floor = player.floor
                            original_ceiling = player.ceiling
                            original_ownership = player.ownership
                            
                            # Calculate projection modifier from newsletter signal (aggressive 20% scaling)
                            projection_modifier = 0.0
                            if signal == 'target':
                                projection_modifier = original_projection * (confidence * 0.20)  # Up to 20% boost
                            elif signal == 'avoid':
                                projection_modifier = original_projection * -(confidence * 0.20)  # Up to 20% reduction
                            
                            # Apply projection modifier to projection, floor, AND ceiling (shift entire distribution)
                            updated_projection = original_projection + projection_modifier
                            updated_floor = original_floor + projection_modifier
                            updated_ceiling = original_ceiling + projection_modifier
                            
                            # Update player object
                            player.projection = updated_projection
                            player.floor = updated_floor
                            player.ceiling = updated_ceiling
                            
                            # Apply ownership adjustment
                            ownership_modifier = ownership_delta * 100  # Convert to percentage points
                            updated_ownership = max(0, min(100, original_ownership + ownership_modifier))
                            
                            # Store all transparency values in metadata
                            player.metadata['original_projection'] = original_projection
                            player.metadata['projection_modifier'] = projection_modifier
                            player.metadata['updated_projection'] = updated_projection
                            
                            player.metadata['original_floor'] = original_floor
                            player.metadata['floor_modifier'] = projection_modifier  # Same as projection
                            player.metadata['updated_floor'] = updated_floor
                            
                            player.metadata['original_ceiling'] = original_ceiling
                            player.metadata['ceiling_modifier'] = projection_modifier  # Same as projection
                            player.metadata['updated_ceiling'] = updated_ceiling
                            
                            player.metadata['original_ownership'] = original_ownership
                            player.metadata['ownership_modifier'] = ownership_modifier
                            player.metadata['updated_ownership'] = updated_ownership
                            
                            # Apply target/fade multipliers based on signal (aggressive weighting)
                            if signal == 'target':
                                target_mult = 1.0 + (confidence * 0.4)  # Up to 40% boost
                                player.metadata['newsletter_target_mult'] = target_mult
                                player.metadata['newsletter_fade_mult'] = 1.0
                            elif signal == 'avoid':
                                fade_mult = 1.0 - (confidence * 0.4)  # Up to 40% reduction  
                                player.metadata['newsletter_fade_mult'] = fade_mult
                                player.metadata['newsletter_target_mult'] = 1.0
                            else:
                                player.metadata['newsletter_target_mult'] = 1.0
                                player.metadata['newsletter_fade_mult'] = 1.0
                            
                            # Store newsletter metadata
                            player.metadata['newsletter_signal'] = signal
                            player.metadata['newsletter_confidence'] = confidence
                            player.metadata['newsletter_reason'] = reason
                            player.metadata['ceiling_delta_applied'] = ceiling_delta
                            player.metadata['ownership_delta_applied'] = ownership_delta
                            
                            modified_count += 1
                            break
                
                logger.info(f"Applied newsletter signals to {modified_count} players")
                
        except ImportError:
            logger.warning("LLM newsletter processor not available")
        except Exception as e:
            logger.error(f"Error applying newsletter signals: {e}")
        
        return players
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same player."""
        n1_parts = set(name1.lower().split())
        n2_parts = set(name2.lower().split())
        
        # If names share at least 2 words (typically first + last name)
        if len(n1_parts & n2_parts) >= 2:
            return True
        
        # Check if one name is contained in the other (for partial matches)
        if len(n1_parts) >= 2 and len(n2_parts) >= 2:
            n1_clean = name1.lower().replace('.', '').replace(',', '')
            n2_clean = name2.lower().replace('.', '').replace(',', '')
            
            return n1_clean in n2_clean or n2_clean in n1_clean
        
        return False


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
        
        # Calculate ITD probability based on finishing stats
        record['itd_probability'] = self._calculate_itd_probability(record)
        
        # ML odds will be extracted from matchup data (Vegas_Odds_fight column)
        
        # ITD-adjusted ceiling will be calculated after ownership merge
        
        return record
    
    def _calculate_itd_probability(self, record: Dict) -> float:
        """Calculate ITD (Inside The Distance) probability based on fighter stats.
        
        Uses finishing stats to estimate probability of early finish.
        """
        # Base ITD probability 
        base_itd = 0.35  # UFC average is around 35% ITD
        
        # Get finishing stats (normalized per minute)
        knockdowns_min = record.get('knockdowns_min', 0)
        submissions_min = record.get('submissions_min', 0) 
        ss_landed_min = record.get('ss_landed_min', 0)
        
        # Calculate finishing potential score
        finishing_score = 0
        
        # Knockdown power (high impact on ITD)
        if knockdowns_min > 0.1:  # Above average knockdown rate
            finishing_score += 0.3
        elif knockdowns_min > 0.05:
            finishing_score += 0.15
            
        # Submission threat
        if submissions_min > 0.2:  # Active submission game
            finishing_score += 0.25
        elif submissions_min > 0.1:
            finishing_score += 0.1
            
        # Strike volume (moderate impact)
        if ss_landed_min > 5.0:  # High volume striker
            finishing_score += 0.1
        elif ss_landed_min > 3.0:
            finishing_score += 0.05
            
        # Cap at reasonable maximum (90% ITD is very rare)
        itd_prob = min(0.90, base_itd + finishing_score)
        
        return round(itd_prob, 3)
    
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
            # Fill missing ownership with 0%
            final_df['ownership'] = final_df['ownership'].fillna(0.0)
            final_df['doubleup'] = final_df['doubleup'].fillna(0.0)
        else:
            # If no ownership data, set all to 0%
            final_df['ownership'] = 0.0
            final_df['doubleup'] = 0.0
            
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