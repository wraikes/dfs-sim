#!/usr/bin/env python3
import sys
sys.path.append('../..')  # Add parent directory to path
"""
⚙️ DFS DATA PROCESSOR  
Transform raw data into analysis-ready CSV format.

Usage:
    python process_data.py --sport mma --pid 466
    python process_data.py --sport nfl --pid week1
    python process_data.py --sport nba --pid 2024-12-15

This script:
    ✅ Loads raw data files (JSON, CSV)
    ✅ Applies newsletter signals & adjustments
    ✅ Calculates sport-specific metrics
    ✅ Generates final extracted CSV
    ✅ Provides data summary for review
"""

import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from src.models.player import Player, Position


class BaseDataProcessor(ABC):
    """Abstract base class for sport-specific data processing."""
    
    def __init__(self, sport: str, pid: str, site: str = 'dk'):
        self.sport = sport.lower()
        self.pid = pid
        self.site = site.lower()
        self.base_path = Path(f"data/{self.sport}/{self.site}/{self.pid}")
        self.raw_path = self.base_path / 'raw'
        self.csv_path = self.base_path / 'csv'
        
        # Ensure directories exist
        self.csv_path.mkdir(exist_ok=True)
    
    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine raw data files into standardized format."""
        pass
    
    @abstractmethod
    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sport-specific metrics (ITD, variance, etc.)."""
        pass
    
    @abstractmethod
    def get_position_from_data(self, row: pd.Series) -> Position:
        """Extract position from raw data."""
        pass
    
    def parse_newsletter_with_llm(self) -> Dict[str, Any]:
        """Parse newsletter content using LLM and create structured signals JSON."""
        # Look for raw newsletter files
        newsletter_files = ['newsletter_signals.json', 'newsletter.json', 'newsletter.txt', 'signals.txt']
        newsletter_content = None
        newsletter_file = None
        
        for filename in newsletter_files:
            file_path = self.raw_path / filename
            if file_path.exists():
                newsletter_file = file_path
                break
        
        if not newsletter_file:
            print("📰 No newsletter file found - creating empty signals")
            return {'targets': [], 'fades': [], 'volatile': []}
        
        print(f"📰 Processing newsletter: {newsletter_file}")
        
        # Read newsletter content
        with open(newsletter_file, 'r') as f:
            if newsletter_file.suffix == '.json':
                # If already JSON, check if it needs LLM processing
                newsletter_data = json.load(f)
                if 'targets' in newsletter_data and 'fades' in newsletter_data:
                    print("📰 Newsletter already in structured format")
                    return newsletter_data
                else:
                    # JSON but unstructured - convert to text for LLM processing
                    newsletter_content = json.dumps(newsletter_data, indent=2)
            else:
                # Plain text content
                newsletter_content = f.read()
        
        if not newsletter_content:
            return {'targets': [], 'fades': [], 'volatile': []}
        
        # Use LLM to parse unstructured newsletter into structured signals
        print("📰 Using LLM to parse newsletter content...")
        try:
            # This is a placeholder for LLM integration
            # In a real implementation, this would call an LLM service
            parsed_signals = self._simulate_llm_newsletter_parsing(newsletter_content)
            
            # Save parsed signals as JSON for future use
            signals_output_path = self.raw_path / 'parsed_newsletter_signals.json'
            with open(signals_output_path, 'w') as f:
                json.dump(parsed_signals, f, indent=2)
            
            print(f"📰 Saved parsed signals to: {signals_output_path}")
            return parsed_signals
            
        except Exception as e:
            print(f"⚠️ LLM newsletter parsing failed: {e}")
            print("📰 Using manual newsletter format fallback")
            return self._try_manual_newsletter_format(newsletter_content)
    
    def _simulate_llm_newsletter_parsing(self, content: str) -> Dict[str, Any]:
        """Simulate LLM parsing of newsletter content (placeholder implementation)."""
        # This is a placeholder that attempts to extract player mentions
        # In a real implementation, this would use an actual LLM service
        
        import re
        
        # Simple pattern matching for common newsletter language
        targets = []
        fades = []
        volatile = []
        
        lines = content.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Look for target indicators
            if any(indicator in line_lower for indicator in ['target', 'like', 'love', 'play', 'favorite']):
                # Extract player names (basic regex)
                names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', line)
                for name in names:
                    targets.append({
                        'name': name,
                        'confidence': 0.7,
                        'reason': line.strip()[:100]
                    })
            
            # Look for fade indicators  
            elif any(indicator in line_lower for indicator in ['fade', 'avoid', 'stay away', 'pass on']):
                names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', line)
                for name in names:
                    fades.append({
                        'name': name,
                        'confidence': 0.6,
                        'reason': line.strip()[:100]
                    })
            
            # Look for volatile indicators
            elif any(indicator in line_lower for indicator in ['boom', 'bust', 'risky', 'upside', 'ceiling']):
                names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', line)
                for name in names:
                    volatile.append({
                        'name': name,
                        'confidence': 0.7,
                        'reason': line.strip()[:100]
                    })
        
        return {
            'targets': targets,
            'fades': fades,
            'volatile': volatile
        }
    
    def _try_manual_newsletter_format(self, content: str) -> Dict[str, Any]:
        """Fallback to try manual newsletter JSON parsing."""
        try:
            # Try to parse as JSON first
            return json.loads(content)
        except:
            # Return empty signals if parsing fails
            return {'targets': [], 'fades': [], 'volatile': []}
    
    def load_newsletter_signals(self) -> Dict[str, Any]:
        """Load newsletter signals, processing with LLM if needed."""
        # First check if we already have parsed signals
        parsed_signals_path = self.raw_path / 'parsed_newsletter_signals.json'
        if parsed_signals_path.exists():
            print(f"📰 Loading parsed signals: {parsed_signals_path}")
            with open(parsed_signals_path, 'r') as f:
                return json.load(f)
        
        # Otherwise parse newsletter content with LLM
        return self.parse_newsletter_with_llm()
    
    def apply_newsletter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply newsletter signals to player data."""
        signals = self.load_newsletter_signals()
        
        # Initialize newsletter columns
        df['newsletter_signal'] = 'neutral'
        df['newsletter_confidence'] = 0.5
        df['newsletter_reason'] = ''
        
        # Apply targets
        for target in signals.get('targets', []):
            name = target['name'].upper()
            mask = df['name'].str.upper().str.contains(name, na=False)
            df.loc[mask, 'newsletter_signal'] = 'target'
            df.loc[mask, 'newsletter_confidence'] = target.get('confidence', 0.7)
            df.loc[mask, 'newsletter_reason'] = target.get('reason', '')
        
        # Apply fades
        for fade in signals.get('fades', []):
            name = fade['name'].upper()
            mask = df['name'].str.upper().str.contains(name, na=False)
            df.loc[mask, 'newsletter_signal'] = 'avoid'
            df.loc[mask, 'newsletter_confidence'] = fade.get('confidence', 0.6)
            df.loc[mask, 'newsletter_reason'] = fade.get('reason', '')
        
        # Apply volatile
        for volatile in signals.get('volatile', []):
            name = volatile['name'].upper()  
            mask = df['name'].str.upper().str.contains(name, na=False)
            df.loc[mask, 'newsletter_signal'] = 'volatile'
            df.loc[mask, 'newsletter_confidence'] = volatile.get('confidence', 0.7)
            df.loc[mask, 'newsletter_reason'] = volatile.get('reason', '')
        
        # Count applied signals
        targets_count = len(df[df['newsletter_signal'] == 'target'])
        fades_count = len(df[df['newsletter_signal'] == 'avoid']) 
        volatile_count = len(df[df['newsletter_signal'] == 'volatile'])
        
        print(f"📰 Applied newsletter signals:")
        print(f"  🎯 {targets_count} targets")
        print(f"  ⛔ {fades_count} fades")
        print(f"  ⚡ {volatile_count} volatile plays")
        
        return df
    
    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply newsletter-based projection adjustments."""
        df = df.copy()
        
        # Store original values
        df['original_projection'] = df['projection']
        df['original_floor'] = df['floor'] 
        df['original_ceiling'] = df['ceiling']
        df['original_ownership'] = df['ownership']
        
        for idx, row in df.iterrows():
            signal = row['newsletter_signal']
            confidence = row['newsletter_confidence']
            
            if signal == 'target':
                # Targets get projection boost
                proj_mult = 1.15 + (0.20 * confidence)  # 1.15-1.35x
                ceil_mult = 1.20 + (0.25 * confidence)  # 1.20-1.45x
                ownership_adj = -5 * confidence         # Reduce perceived ownership
                
                df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * ceil_mult
                df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult
                df.loc[idx, 'updated_ownership'] = max(0.1, row['ownership'] + ownership_adj)
                
            elif signal == 'avoid':
                # Fades get projection reduction
                proj_mult = 0.85 - (0.15 * confidence)  # 0.70-0.85x
                ceil_mult = 0.80 - (0.10 * confidence)  # 0.70-0.80x 
                ownership_adj = 10 * confidence         # Increase perceived ownership
                
                df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * ceil_mult
                df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult
                df.loc[idx, 'updated_ownership'] = row['ownership'] + ownership_adj
                
            elif signal == 'volatile':
                # Volatile plays get ceiling boost, higher variance
                ceil_mult = 1.10 + (0.15 * confidence)  # 1.10-1.25x ceiling
                ownership_adj = -3 * confidence         # Slightly lower ownership
                var_mult = 1.50 + (0.50 * confidence)   # 1.5-2.0x variance
                
                df.loc[idx, 'updated_projection'] = row['projection']  # No proj change
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * ceil_mult
                df.loc[idx, 'updated_floor'] = row['floor']  # Floor unchanged
                df.loc[idx, 'updated_ownership'] = max(0.1, row['ownership'] + ownership_adj)
                df.loc[idx, 'std_dev'] = row['std_dev'] * var_mult
                
            else:
                # Neutral - no changes
                df.loc[idx, 'updated_projection'] = row['projection']
                df.loc[idx, 'updated_ceiling'] = row['ceiling']
                df.loc[idx, 'updated_floor'] = row['floor']
                df.loc[idx, 'updated_ownership'] = row['ownership']
        
        return df
    
    def process_data(self) -> pd.DataFrame:
        """Main data processing pipeline."""
        print(f"\n⚙️ Processing {self.sport.upper()} data for PID: {self.pid}")
        print("=" * 60)
        
        # Load raw data
        print("1️⃣ Loading raw data files...")
        df = self.load_raw_data()
        print(f"   ✅ Loaded {len(df)} players")
        
        # Apply newsletter signals
        print("\n2️⃣ Applying newsletter signals...")
        df = self.apply_newsletter_signals(df)
        
        # Calculate sport-specific metrics
        print(f"\n3️⃣ Calculating {self.sport.upper()} metrics...")
        df = self.calculate_sport_metrics(df)
        
        # Apply projection adjustments
        print("\n4️⃣ Applying projection adjustments...")
        df = self.apply_projection_adjustments(df)
        
        # Calculate final metrics
        print("\n5️⃣ Calculating final metrics...")
        df['value'] = df['updated_projection'] / df['salary'] * 1000
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame) -> Path:
        """Save processed data to CSV."""
        output_file = self.csv_path / f"dk_{self.pid}_extracted.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved processed data: {output_file}")
        return output_file
    
    def display_data_summary(self, df: pd.DataFrame):
        """Display data summary for review."""
        print(f"\n📊 {self.sport.upper()} DATA SUMMARY")
        print("=" * 60)
        
        print(f"Total Players: {len(df)}")
        
        # Salary distribution
        print(f"\n💰 Salary Range: ${df['salary'].min():,} - ${df['salary'].max():,}")
        print(f"   Average: ${df['salary'].mean():,.0f}")
        
        # Projection distribution
        print(f"\n📊 Projections: {df['updated_projection'].min():.1f} - {df['updated_projection'].max():.1f}")
        print(f"   Average: {df['updated_projection'].mean():.1f}")
        
        # Ownership distribution
        print(f"\n👥 Ownership: {df['updated_ownership'].min():.1f}% - {df['updated_ownership'].max():.1f}%")
        print(f"   Average: {df['updated_ownership'].mean():.1f}%")
        
        # Newsletter signal summary
        signal_counts = df['newsletter_signal'].value_counts()
        print(f"\n📰 Newsletter Signals:")
        for signal, count in signal_counts.items():
            icon = {'target': '🎯', 'avoid': '⛔', 'volatile': '⚡', 'neutral': '  '}.get(signal, '  ')
            print(f"   {icon} {signal}: {count}")
        
        # Top players by value
        print(f"\n💎 Top 5 Values:")
        top_values = df.nlargest(5, 'value')[['name', 'salary', 'updated_projection', 'value', 'newsletter_signal']]
        for _, row in top_values.iterrows():
            signal_icon = {'target': '🎯', 'avoid': '⛔', 'volatile': '⚡', 'neutral': '  '}.get(row['newsletter_signal'], '  ')
            print(f"   {signal_icon} {row['name']:20} ${row['salary']:4,} {row['updated_projection']:5.1f}pts {row['value']:5.1f}")


class MMADataProcessor(BaseDataProcessor):
    """MMA-specific data processor."""
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load and parse MMA JSON data into standardized CSV format."""
        json_files = list(self.raw_path.glob("*.json"))
        main_json_file = None
        
        # Look for main data file (dk_pid.json or fd_pid.json)  
        for json_file in json_files:
            if json_file.name.startswith(f"{self.site}_{self.pid}") and not 'newsletter' in json_file.name.lower():
                main_json_file = json_file
                break
        
        if not main_json_file:
            raise FileNotFoundError(f"No main JSON file found matching {self.site}_{self.pid}.json in {self.raw_path}")
        
        print(f"   📄 Loading JSON: {main_json_file}")
        
        with open(main_json_file, 'r') as f:
            data = json.load(f)
        
        # Extract salary data from SalaryContainerJson (MMA-specific parsing)
        salary_data = json.loads(data['SalaryContainerJson'])
        players = []
        
        for player_data in salary_data['Salaries']:
            player_id = player_data['PID']
            player = {
                'player_id': player_id,
                'name': player_data['Name'],
                'salary': player_data['Salary'],
                'projection': player_data.get('AvgPointsPerGame', 0.0),
                'floor': player_data.get('FloorPointsPerGame', 0.0), 
                'ceiling': player_data.get('CeilingPointsPerGame', 0.0),
                'ownership': player_data.get('PercentOwned', 10.0),
                'std_dev': 25.0  # Default variance for MMA
            }
            players.append(player)
        
        # Create base dataframe
        df = pd.DataFrame(players)
        
        # Extract ML odds and additional data from MatchupData (MMA-specific)
        ml_odds_map = {}
        win_pct_map = {}
        itd_odds_map = {}
        
        for table in data.get('MatchupData', []):
            if table.get('Name') == 'Fight':
                columns = table['Columns']
                for match in table.get('PlayerMatchups', []):
                    player_id = match['PlayerId']
                    values = match['Values']
                    
                    # Map columns to values
                    col_data = dict(zip(columns, values))
                    
                    # Extract ML odds (Vegas Odds column)
                    if 'Vegas Odds' in col_data:
                        ml_odds_map[player_id] = float(col_data['Vegas Odds']) if col_data['Vegas Odds'] != '0' else 0
                    
                    # Extract win percentage
                    if 'Win Pct' in col_data:
                        win_pct_map[player_id] = float(col_data['Win Pct']) if col_data['Win Pct'] != '0' else 0
                    
                    # Extract full fight odds (ITD odds)
                    if 'Full Fight Odds' in col_data:
                        itd_odds_map[player_id] = float(col_data['Full Fight Odds']) if col_data['Full Fight Odds'] != '0' else 0
        
        # Add ML odds and win data to dataframe
        df['ml_odds'] = df['player_id'].map(ml_odds_map).fillna(0)
        df['win_pct'] = df['player_id'].map(win_pct_map).fillna(0)
        df['itd_odds'] = df['player_id'].map(itd_odds_map).fillna(0)
        
        print(f"   ✅ Parsed {len(df)} fighters from JSON")
        return df
    
    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MMA-specific metrics."""
        df = df.copy()
        
        # Calculate ITD probability from ML odds if not present
        if 'itd_probability' not in df.columns:
            df['itd_probability'] = 0.35  # Default
            
            # Estimate based on ML odds
            if 'ml_odds' in df.columns:
                # Favorites (negative odds) tend to have higher finishing rates
                df.loc[df['ml_odds'] <= -200, 'itd_probability'] = 0.45
                df.loc[df['ml_odds'] >= 200, 'itd_probability'] = 0.25
        
        # Calculate ITD-adjusted ceiling
        df['itd_adjusted_ceiling'] = df.apply(
            lambda row: row['ceiling'] * (1 + 0.4 * row['itd_probability']), axis=1
        )
        
        return df
    
    def get_position_from_data(self, row: pd.Series) -> Position:
        """MMA fighters are all FIGHTER position."""
        return Position.FIGHTER


def create_processor(sport: str, pid: str, site: str = 'dk') -> BaseDataProcessor:
    """Factory function to create sport-specific processor."""
    processors = {
        'mma': MMADataProcessor,
        # 'nfl': NFLDataProcessor,  # Future implementation
        # 'nba': NBADataProcessor,  # Future implementation
    }
    
    processor_class = processors.get(sport.lower())
    if not processor_class:
        raise ValueError(f"Sport '{sport}' not yet supported")
    
    return processor_class(sport, pid, site)


def main():
    """Main data processing workflow."""
    parser = argparse.ArgumentParser(
        description="⚙️ DFS Data Processor - Transform raw data into analysis-ready CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--sport', type=str, required=True,
                       choices=['mma', 'nfl', 'nba'],
                       help='Sport to process data for')
    parser.add_argument('--pid', type=str, required=True,
                       help='Contest/event identifier')
    parser.add_argument('--site', type=str, default='dk',
                       choices=['dk', 'fd'],
                       help='DFS site: dk=DraftKings, fd=FanDuel (default: dk)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only display data summary, skip processing')
    
    args = parser.parse_args()
    
    print("⚙️ DFS DATA PROCESSOR")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    print(f"Site: {args.site.upper()}")
    print("=" * 60)
    
    try:
        processor = create_processor(args.sport, args.pid, args.site)
        
        if args.summary_only:
            # Load existing processed data and show summary
            csv_file = processor.csv_path / f"dk_{args.pid}_extracted.csv"
            if not csv_file.exists():
                print(f"❌ No processed data found: {csv_file}")
                return 1
            
            df = pd.read_csv(csv_file)
            processor.display_data_summary(df)
            return 0
        
        # Full processing workflow
        df = processor.process_data()
        output_file = processor.save_processed_data(df)
        processor.display_data_summary(df)
        
        print(f"\n🎯 NEXT STEPS")
        print("=" * 40)
        print("1. Review the data summary above")
        print("2. Check processed CSV for any issues:")
        print(f"   {output_file}")
        print("3. Run lineup optimization when satisfied:")
        print(f"   python optimize.py --sport {args.sport} --pid {args.pid}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())