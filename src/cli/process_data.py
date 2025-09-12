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
    
    def __init__(self, sport: str, pid: str):
        self.sport = sport.lower()
        self.pid = pid
        self.base_path = Path(f"data/{self.sport}/{self.pid}")
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
    
    def load_newsletter_signals(self) -> Dict[str, Any]:
        """Load newsletter signals if available."""
        newsletter_files = ['newsletter_signals.json', 'signals.json', 'newsletter.json']
        
        for filename in newsletter_files:
            newsletter_path = self.raw_path / filename
            if newsletter_path.exists():
                print(f"📰 Loading newsletter signals: {newsletter_path}")
                with open(newsletter_path, 'r') as f:
                    return json.load(f)
        
        print("📰 No newsletter signals found (using neutral)")
        return {'targets': [], 'fades': [], 'volatile': []}
    
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
        """Load MMA raw data files."""
        # For now, assume we have the existing CSV format
        # In a real implementation, this would combine multiple sources
        csv_files = list(self.raw_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_path}")
        
        # Use the first CSV file found (in real implementation, would merge multiple)
        main_file = csv_files[0]
        print(f"   📄 Loading: {main_file}")
        df = pd.read_csv(main_file)
        
        # Standardize column names
        column_mapping = {
            'player_id': 'player_id',
            'name': 'name', 
            'salary': 'salary',
            'updated_projection': 'projection',
            'updated_floor': 'floor',
            'updated_ceiling': 'ceiling',
            'updated_ownership': 'ownership',
            'std_dev': 'std_dev'
        }
        
        # Rename columns to standard format
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Fill missing values
        df['std_dev'] = df.get('std_dev', 25.0)
        df['ownership'] = df.get('ownership', 10.0)
        
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


def create_processor(sport: str, pid: str) -> BaseDataProcessor:
    """Factory function to create sport-specific processor."""
    processors = {
        'mma': MMADataProcessor,
        # 'nfl': NFLDataProcessor,  # Future implementation
        # 'nba': NBADataProcessor,  # Future implementation
    }
    
    processor_class = processors.get(sport.lower())
    if not processor_class:
        raise ValueError(f"Sport '{sport}' not yet supported")
    
    return processor_class(sport, pid)


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
    parser.add_argument('--summary-only', action='store_true',
                       help='Only display data summary, skip processing')
    
    args = parser.parse_args()
    
    print("⚙️ DFS DATA PROCESSOR")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    print("=" * 60)
    
    try:
        processor = create_processor(args.sport, args.pid)
        
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