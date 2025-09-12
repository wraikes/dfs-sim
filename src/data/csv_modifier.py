"""CSV modifier for applying newsletter signals to extracted player data."""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CSVModifier:
    """Modify player CSV data based on newsletter signals."""
    
    def __init__(self):
        """Initialize CSV modifier."""
        pass
    
    def apply_newsletter_signals(self, csv_path: str, signals_json_path: str, 
                               output_path: Optional[str] = None) -> str:
        """
        Apply newsletter signals to CSV and save modified version.
        
        Args:
            csv_path: Path to extracted player CSV
            signals_json_path: Path to newsletter signals JSON
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to modified CSV file
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        original_count = len(df)
        
        # Load signals
        with open(signals_json_path, 'r') as f:
            signals_data = json.load(f)
        
        # Apply modifications
        modified_count = 0
        for player_signal in signals_data.get('players', []):
            name = player_signal.get('name', '')
            signal = player_signal.get('signal', 'neutral')
            confidence = player_signal.get('confidence', 0.5)
            ceiling_delta = player_signal.get('ceiling_delta', 0.0)
            ownership_delta = player_signal.get('ownership_delta', 0.0)
            reason = player_signal.get('reason', '')
            
            # Find matching player(s) in CSV
            matches = self._find_matching_players(df, name)
            
            for idx in matches:
                # Apply ceiling adjustment
                if ceiling_delta != 0:
                    multiplier = 1.0 + ceiling_delta
                    original_ceiling = df.at[idx, 'ceiling']
                    new_ceiling = max(df.at[idx, 'projection'], original_ceiling * multiplier)
                    df.at[idx, 'ceiling'] = new_ceiling
                
                # Apply ownership adjustment  
                if ownership_delta != 0:
                    original_ownership = df.at[idx, 'ownership']
                    new_ownership = max(0, min(100, original_ownership + (ownership_delta * 100)))
                    df.at[idx, 'ownership'] = new_ownership
                
                # Add newsletter metadata columns
                df.at[idx, 'newsletter_signal'] = signal
                df.at[idx, 'newsletter_confidence'] = confidence
                df.at[idx, 'newsletter_reason'] = reason
                df.at[idx, 'ceiling_delta_applied'] = ceiling_delta
                df.at[idx, 'ownership_delta_applied'] = ownership_delta
                
                # Apply target/fade multipliers based on signal
                if signal == 'target':
                    target_mult = 1.0 + (confidence * 0.2)  # Up to 20% boost
                    df.at[idx, 'newsletter_target_mult'] = target_mult
                elif signal == 'avoid':
                    fade_mult = 1.0 - (confidence * 0.2)  # Up to 20% reduction  
                    df.at[idx, 'newsletter_fade_mult'] = fade_mult
                else:
                    df.at[idx, 'newsletter_target_mult'] = 1.0
                    df.at[idx, 'newsletter_fade_mult'] = 1.0
                
                modified_count += 1
        
        # Fill in default values for players without newsletter signals
        newsletter_cols = ['newsletter_signal', 'newsletter_confidence', 'newsletter_reason', 
                          'ceiling_delta_applied', 'ownership_delta_applied', 
                          'newsletter_target_mult', 'newsletter_fade_mult']
        
        for col in newsletter_cols:
            if col not in df.columns:
                if 'mult' in col:
                    df[col] = 1.0
                elif 'delta' in col:
                    df[col] = 0.0
                elif col == 'newsletter_confidence':
                    df[col] = 0.0
                else:
                    df[col] = 'neutral' if col == 'newsletter_signal' else ''
        
        # Fill NaN values
        df['newsletter_signal'] = df['newsletter_signal'].fillna('neutral')
        df['newsletter_confidence'] = df['newsletter_confidence'].fillna(0.0)
        df['newsletter_reason'] = df['newsletter_reason'].fillna('')
        df['ceiling_delta_applied'] = df['ceiling_delta_applied'].fillna(0.0)
        df['ownership_delta_applied'] = df['ownership_delta_applied'].fillna(0.0)
        df['newsletter_target_mult'] = df['newsletter_target_mult'].fillna(1.0)
        df['newsletter_fade_mult'] = df['newsletter_fade_mult'].fillna(1.0)
        
        # Determine output path - keep in same /csv/ folder
        if output_path is None:
            input_path = Path(csv_path)
            output_path = input_path.parent / f"{input_path.stem}_modified.csv"
        
        # Save modified CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Applied newsletter signals to {modified_count} players")
        logger.info(f"Modified CSV saved to: {output_path}")
        
        return str(output_path)
    
    def _find_matching_players(self, df: pd.DataFrame, target_name: str) -> List[int]:
        """Find player indices that match the target name."""
        matches = []
        target_name_clean = target_name.lower().strip()
        
        for idx, row in df.iterrows():
            player_name = str(row['name']).lower().strip()
            
            # Exact match
            if player_name == target_name_clean:
                matches.append(idx)
                continue
            
            # Partial matching
            if self._names_match(player_name, target_name_clean):
                matches.append(idx)
        
        return matches
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same player."""
        n1_parts = set(name1.split())
        n2_parts = set(name2.split())
        
        # If names share at least 2 words (typically first + last name)
        if len(n1_parts & n2_parts) >= 2:
            return True
        
        # Check if one name is contained in the other (for partial matches)
        if len(n1_parts) >= 2 and len(n2_parts) >= 2:
            n1_clean = name1.replace('.', '').replace(',', '')
            n2_clean = name2.replace('.', '').replace(',', '')
            
            return n1_clean in n2_clean or n2_clean in n1_clean
        
        return False
    
    def get_modification_summary(self, modified_csv_path: str) -> str:
        """Get summary of modifications applied."""
        df = pd.read_csv(modified_csv_path)
        
        # Count players with signals
        signal_counts = df['newsletter_signal'].value_counts()
        
        summary = f"Newsletter Modification Summary:\n"
        summary += f"Total players: {len(df)}\n\n"
        
        for signal_type, count in signal_counts.items():
            if signal_type != 'neutral':
                summary += f"{signal_type.upper()}: {count} players\n"
                
                # Show specific players
                signal_players = df[df['newsletter_signal'] == signal_type]
                for _, player in signal_players.iterrows():
                    name = player['name']
                    confidence = player['newsletter_confidence']
                    ceiling_delta = player['ceiling_delta_applied']
                    ownership_delta = player['ownership_delta_applied']
                    
                    summary += f"  • {name} (conf: {confidence:.2f})\n"
                    if ceiling_delta != 0:
                        summary += f"    └─ Ceiling: {ceiling_delta:+.1%}\n"
                    if ownership_delta != 0:
                        summary += f"    └─ Ownership: {ownership_delta*100:+.1f}%\n"
                summary += "\n"
        
        neutral_count = signal_counts.get('neutral', len(df))
        summary += f"NEUTRAL: {neutral_count} players (no newsletter signal)\n"
        
        return summary