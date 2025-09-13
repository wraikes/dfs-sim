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
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from src.models.player import Player, Position
from src.data.llm_newsletter_processor import LLMNewsletterProcessor, NewsletterExtraction


class BaseDataProcessor(ABC):
    """Abstract base class for sport-specific data processing."""
    
    def __init__(self, sport: str, pid: str, site: str = 'dk'):
        self.sport = sport.lower()
        self.pid = pid
        self.site = site.lower()
        self.base_path = Path(f"data/{self.sport}/{self.pid}/{self.site}")
        self.json_path = self.base_path / 'json'
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
    
    def process_newsletter_with_llm(self) -> Optional[NewsletterExtraction]:
        """Process newsletter using LLM and create structured signals."""
        # Initialize LLM processor
        llm_processor = LLMNewsletterProcessor(sport=self.sport, model="gpt-4o")
        
        # Look for newsletter files in the /newsletters subdirectory
        newsletter_dir = self.base_path / 'newsletters'
        if not newsletter_dir.exists():
            print("📰 No newsletters directory found - creating empty signals")
            return None
        
        # Find newsletter files
        newsletter_files = list(newsletter_dir.glob("*.txt")) + list(newsletter_dir.glob("*.md"))
        if not newsletter_files:
            print("📰 No newsletter files found - creating empty signals")
            return None
        
        print(f"📰 Processing newsletter: {newsletter_files[0]}")
        
        try:
            # Process the first newsletter file found
            extraction = llm_processor.process_newsletter_file(
                str(newsletter_files[0]), 
                save_json=True  # Auto-save to /json directory
            )
            
            print(f"📰 Extracted {len(extraction.players)} player signals")
            return extraction
            
        except Exception as e:
            print(f"⚠️ LLM newsletter processing failed: {e}")
            return None
    
    def apply_newsletter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply newsletter signals to player data using LLM processor."""
        # Always initialize newsletter columns
        df['newsletter_signal'] = 'neutral'
        df['newsletter_confidence'] = 0.5
        df['newsletter_reason'] = ''
        df['ownership_delta'] = 0.0
        df['ceiling_delta'] = 0.0
        
        # Try to process newsletter
        extraction = self.process_newsletter_with_llm()
        
        if not extraction or not extraction.players:
            print("📰 No newsletter signals found - using default values")
            return df
        
        # Apply signals from LLM extraction
        applied_count = 0
        for signal in extraction.players:
            # Find matching players
            name_upper = signal.name.upper()
            mask = df['name'].str.upper().str.contains(name_upper, na=False)
            matched_players = df[mask]
            
            if len(matched_players) > 0:
                df.loc[mask, 'newsletter_signal'] = signal.signal
                df.loc[mask, 'newsletter_confidence'] = signal.confidence
                df.loc[mask, 'newsletter_reason'] = signal.reason
                df.loc[mask, 'ownership_delta'] = signal.ownership_delta
                df.loc[mask, 'ceiling_delta'] = signal.ceiling_delta
                applied_count += len(matched_players)
        
        # Count applied signals by type
        targets_count = len(df[df['newsletter_signal'] == 'target'])
        avoids_count = len(df[df['newsletter_signal'] == 'avoid'])
        
        print(f"📰 Applied newsletter signals:")
        print(f"  🎯 {targets_count} targets")
        print(f"  ⛔ {avoids_count} avoids")
        print(f"  📊 {applied_count} total players affected")
        
        return df
    
    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply newsletter-based projection adjustments using LLM signals."""
        df = df.copy()
        
        # Always add updated columns - copy originals as default
        df['updated_projection'] = df['projection']
        df['updated_floor'] = df['floor'] 
        df['updated_ceiling'] = df['ceiling']
        df['updated_ownership'] = df['ownership']
        
        # Only modify if newsletter signals exist
        newsletter_players = df[df['newsletter_signal'] != 'neutral']
        
        for idx, row in newsletter_players.iterrows():
            signal = row['newsletter_signal']
            confidence = row['newsletter_confidence']
            ceiling_delta = row['ceiling_delta']
            ownership_delta = row['ownership_delta']
            
            if signal == 'target':
                # Targets get projection boost based on confidence
                proj_mult = 1.02 + (0.08 * confidence)  # 1.02-1.10x
                
                df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
                df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult
                # Use LLM-calculated ceiling delta
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * (1 + ceiling_delta)
                # Use LLM-calculated ownership delta  
                df.loc[idx, 'updated_ownership'] = max(0.1, row['ownership'] + (ownership_delta * 100))
                
            elif signal == 'avoid':
                # Avoids get projection reduction based on confidence
                proj_mult = 0.96 - (0.06 * confidence)  # 0.90-0.96x
                
                df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
                df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult
                # Use LLM-calculated ceiling delta
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * (1 + ceiling_delta)
                # Use LLM-calculated ownership delta
                df.loc[idx, 'updated_ownership'] = row['ownership'] + (ownership_delta * 100)
        
        # Apply derived metric influences AFTER newsletter signals (positive only to avoid double-counting)
        # Target: 5% max total impact from our calculated metrics (reduced from 15% to let correlations handle style effects)
        
        # 1. Finishing rate influences ceiling (0-4% boost)
        finishing_boost = df['finishing_rate'] > 0
        df.loc[finishing_boost, 'updated_ceiling'] *= (1 + df.loc[finishing_boost, 'finishing_rate'] * 0.04)
        
        # 2. Style score influences projection (0-2.7% boost for well-rounded fighters)
        style_boost = df['style_score'] > 0
        df.loc[style_boost, 'updated_projection'] *= (1 + df.loc[style_boost, 'style_score'] * 0.0005)
        
        # 3. Matchup advantage influences all metrics (0-2% boost)
        positive_matchup = df['matchup_advantage'] > 0
        df.loc[positive_matchup, 'updated_projection'] *= (1 + df.loc[positive_matchup, 'matchup_advantage'] * 0.002)
        df.loc[positive_matchup, 'updated_floor'] *= (1 + df.loc[positive_matchup, 'matchup_advantage'] * 0.002)
        df.loc[positive_matchup, 'updated_ceiling'] *= (1 + df.loc[positive_matchup, 'matchup_advantage'] * 0.002)
        
        # 4. Takedown effectiveness influences floor (0-1.3% boost for strong grapplers)
        takedown_eff = df['takedowns_per_fight'] * (1 - df['takedown_defense'] / 100)
        strong_grapplers = takedown_eff > 0.5
        df.loc[strong_grapplers, 'updated_floor'] *= (1 + (takedown_eff.loc[strong_grapplers] - 0.5) * 0.027)
        
        # 5. Vegas odds adjustments (most important data source)
        print("   🎰 Applying Vegas odds adjustments...")
        
        for idx, row in df.iterrows():
            ml_odds = row['ml_odds']
            if ml_odds == 0:  # Skip if no odds data
                continue
                
            # Convert ML odds to win probability
            if ml_odds < 0:  # Favorite
                win_prob = abs(ml_odds) / (abs(ml_odds) + 100)
                # Higher floor for favorites (win more often)
                floor_boost = win_prob * 0.15  # Up to 15% boost for heavy favorites
                df.loc[idx, 'updated_floor'] *= (1 + floor_boost)
            else:  # Dog
                win_prob = 100 / (ml_odds + 100)
                # Higher ceiling for dogs (upset potential)
                ceiling_boost = (1 - win_prob) * 0.20  # Up to 20% boost for heavy dogs
                df.loc[idx, 'updated_ceiling'] *= (1 + ceiling_boost)
        
        # Apply bounds validation after all adjustments
        print("   🔒 Applying bounds validation...")
        
        # Clamp ownership to [0, 100]
        df['updated_ownership'] = np.clip(df['updated_ownership'], 0, 100)
        
        # Ensure ceiling >= projection
        mask = df['updated_ceiling'] < df['updated_projection']
        df.loc[mask, 'updated_ceiling'] = df.loc[mask, 'updated_projection'] * 1.1  # Min 10% ceiling buffer
        
        # Ensure floor <= projection
        mask = df['updated_floor'] > df['updated_projection']
        df.loc[mask, 'updated_floor'] = df.loc[mask, 'updated_projection'] * 0.8  # Max 80% floor
        
        # Ensure all values are non-negative
        df['updated_projection'] = np.maximum(df['updated_projection'], 0.1)
        df['updated_floor'] = np.maximum(df['updated_floor'], 0)
        df['updated_ceiling'] = np.maximum(df['updated_ceiling'], df['updated_projection'])
        
        # Apply salary-ownership correlation model (50/50 weighting)
        print("   💰 Applying salary-ownership correlation model...")
        df = self._apply_salary_ownership_correlation(df)
        
        return df
    
    def _apply_salary_ownership_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply salary-ownership correlation model with 50/50 weighting."""
        df = df.copy()
        
        # Calculate salary-based ownership expectations
        # Higher salary players should have higher ownership (positive correlation)
        min_salary = df['salary'].min()
        max_salary = df['salary'].max()
        salary_range = max_salary - min_salary
        
        if salary_range > 0:
            # Normalize salaries to 0-1 range
            normalized_salary = (df['salary'] - min_salary) / salary_range
            
            # Convert to ownership expectations (2% to 20% range based on salary)
            # Higher salary = higher ownership expectation (realistic for MMA)
            salary_based_ownership = 2 + (normalized_salary * 18)
        else:
            # If all salaries are the same, use uniform ownership
            salary_based_ownership = pd.Series([15.0] * len(df), index=df.index)
        
        # Create blended ownership with 50/50 weighting
        original_ownership = df['updated_ownership']
        blended_ownership = (0.5 * original_ownership) + (0.5 * salary_based_ownership)
        
        # Ensure bounds [0.1, 100]
        df['salary_ownership'] = np.clip(blended_ownership, 0.1, 100.0)
        
        # Calculate correlation improvement
        orig_corr = df['salary'].corr(df['updated_ownership'])
        new_corr = df['salary'].corr(df['salary_ownership'])
        
        print(f"      Original salary-ownership correlation: {orig_corr:.3f}")
        print(f"      Enhanced salary-ownership correlation: {new_corr:.3f}")
        print(f"      Improvement: {new_corr - orig_corr:+.3f}")
        
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
        output_file = self.csv_path / "extracted.csv"
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
        # Look for raw.json file in json directory
        main_json_file = self.json_path / "raw.json"
        
        if not main_json_file.exists():
            raise FileNotFoundError(f"No raw.json file found in {self.json_path}")
        
        print(f"   📄 Loading JSON: {main_json_file}")
        
        with open(main_json_file, 'r') as f:
            data = json.load(f)
        
        # Extract salary data from SalaryContainerJson (MMA-specific parsing)
        try:
            salary_data = json.loads(data['SalaryContainerJson'])
            
            # Extract ownership data from Ownership section
            ownership_map = {}
            if 'Ownership' in data and 'Projected' in data['Ownership']:
                projected = data['Ownership']['Projected']
                # Get the contest ID (should be one key in Projected)
                contest_ids = list(projected.keys())
                if contest_ids:
                    contest_id = contest_ids[0]
                    for ownership_player in projected[contest_id]:
                        salary_id = ownership_player.get('SalaryId')
                        owned_pct = ownership_player.get('Owned', 10.0)
                        if salary_id:
                            ownership_map[salary_id] = owned_pct
            
            players = []
            
            for player_data in salary_data['Salaries']:
                player_id = player_data['PID']
                salary_id = player_data['Id']  # Used to match with ownership
                
                # Get ownership from ownership_map using SalaryId
                ownership = ownership_map.get(salary_id, 10.0)
                
                player = {
                    'player_id': player_id,
                    'name': player_data['Name'],
                    'salary': player_data['SAL'],
                    'projection': player_data.get('PP', 0.0),  # PP = Projected Points
                    'floor': player_data.get('Floor', 0.0), 
                    'ceiling': player_data.get('Ceil', 0.0),
                    'ownership': ownership,
                    'std_dev': 25.0,  # Default variance for MMA
                    'hteam': player_data.get('HTEAM', ''),  # Home team
                    'oteam': player_data.get('OTEAM', ''),  # Opponent team
                    'htid': player_data.get('HTID', 0),  # Home team ID
                    'otid': player_data.get('OTID', 0),  # Opponent team ID 
                    'agg_proj': player_data.get('AggProj', 0.0),  # Aggregate projection
                    'confidence': player_data.get('Conf', 50),  # Projection confidence
                    'game_info': player_data.get('GI', ''),  # Game info with fight time
                    'opp_rank': player_data.get('OppRank', 0),  # Opponent ranking
                    'stars': player_data.get('Stars', 3),  # Player star rating
                    'alert_score': player_data.get('AlertScore', 0),  # Alert/attention score
                }
                players.append(player)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ❌ Error parsing SalaryContainerJson: {e}")
            raise ValueError(f"Invalid LineStar data format: {e}")
        
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
        
        # Extract opponent relationships using OTEAM (opponent team) data
        opponent_map = {}
        opponent_odds_map = {}
        
        # Group fighters by their OTEAM to find opponents
        for i, row1 in df.iterrows():
            if row1['player_id'] in opponent_map:
                continue  # Already matched
                
            player1_id = row1['player_id']
            player1_hteam = row1['hteam']
            player1_oteam = row1['oteam']
            
            # Skip if no team data
            if not player1_oteam or not player1_hteam:
                continue
                
            # Find the fighter whose HTEAM matches our OTEAM
            for j, row2 in df.iterrows():
                if i >= j or row2['player_id'] in opponent_map:
                    continue
                    
                player2_id = row2['player_id']
                player2_hteam = row2['hteam']
                player2_oteam = row2['oteam']
                
                # Skip if no team data
                if not player2_oteam or not player2_hteam:
                    continue
                
                # Check if they are opponents: player1's OTEAM = player2's HTEAM AND vice versa
                if player1_oteam == player2_hteam and player1_hteam == player2_oteam:
                    opponent_map[player1_id] = row2['name']
                    opponent_map[player2_id] = row1['name']
                    # Also store opponent odds for variance modeling
                    opponent_odds_map[player1_id] = row2['ml_odds']
                    opponent_odds_map[player2_id] = row1['ml_odds']
                    break
        
        # Add opponent data to dataframe
        df['opponent'] = df['player_id'].map(opponent_map).fillna('')
        df['opponent_ml_odds'] = df['player_id'].map(opponent_odds_map).fillna(0)
        
        print(f"   ✅ Parsed {len(df)} fighters from JSON")
        print(f"   🥊 Found {len(opponent_map)} opponent relationships using OTEAM/HTEAM data")
        return df
    
    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MMA-specific metrics and variability features."""
        df = df.copy()
        
        # Load the main JSON data to access MatchupData
        main_json_file = self.json_path / "raw.json"
        with open(main_json_file, 'r') as f:
            data = json.load(f)
        
        # Initialize feature maps for variability analysis
        feature_maps = {
            # Last 5 Fights offensive stats
            'strikes_per_fight': {},
            'strikes_per_min': {},
            'strike_accuracy': {},
            'control_per_fight': {},
            'takedowns_per_fight': {},
            'avg_rounds': {},
            'last5_win_pct': {},
            'last5_fppg': {},
            
            # Last 5 Fights defensive stats 
            'strikes_absorbed': {},
            'strike_defense': {},
            'takedown_defense': {},
            'fppg_allowed': {},
            
            # Opponent stats for matchup analysis
            'opp_strikes_per_fight': {},
            'opp_strike_accuracy': {},
            'opp_last5_fppg': {},
            'opp_takedowns_per_fight': {},
        }
        
        # Process MatchupData tables for detailed stats
        for table in data.get('MatchupData', []):
            table_name = table.get('Name', '').lower()
            columns = table.get('Columns', [])
            
            for match in table.get('PlayerMatchups', []):
                player_id = match['PlayerId']
                values = match['Values']
                col_data = dict(zip(columns, values))
                
                # Extract Last 5 Fights offensive stats
                if 'last 5 fights (off)' in table_name:
                    if 'SS Landed/F' in col_data:
                        feature_maps['strikes_per_fight'][player_id] = float(col_data['SS Landed/F']) if col_data['SS Landed/F'] != '0' else 0
                    if 'SS Landed/Min' in col_data:
                        feature_maps['strikes_per_min'][player_id] = float(col_data['SS Landed/Min']) if col_data['SS Landed/Min'] != '0' else 0
                    if 'Strike Acc%' in col_data:
                        feature_maps['strike_accuracy'][player_id] = float(col_data['Strike Acc%']) if col_data['Strike Acc%'] != '0' else 0
                    if 'Ctrl Secs/F' in col_data:
                        feature_maps['control_per_fight'][player_id] = float(col_data['Ctrl Secs/F']) if col_data['Ctrl Secs/F'] != '0' else 0
                    if 'Takedowns/F' in col_data:
                        feature_maps['takedowns_per_fight'][player_id] = float(col_data['Takedowns/F']) if col_data['Takedowns/F'] != '0' else 0
                    if 'Rounds/F' in col_data:
                        feature_maps['avg_rounds'][player_id] = float(col_data['Rounds/F']) if col_data['Rounds/F'] != '0' else 0
                    if 'Win%' in col_data:
                        feature_maps['last5_win_pct'][player_id] = float(col_data['Win%']) if col_data['Win%'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['last5_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0
                
                # Extract Last 5 Fights defensive stats
                elif 'last 5 fights (def)' in table_name:
                    if 'SS Taken/F' in col_data:
                        feature_maps['strikes_absorbed'][player_id] = float(col_data['SS Taken/F']) if col_data['SS Taken/F'] != '0' else 0
                    if 'Strike Def%' in col_data:
                        feature_maps['strike_defense'][player_id] = float(col_data['Strike Def%']) if col_data['Strike Def%'] != '0' else 0
                    if 'Takedown Def%' in col_data:
                        feature_maps['takedown_defense'][player_id] = float(col_data['Takedown Def%']) if col_data['Takedown Def%'] != '0' else 0
                    if 'FPPG Allowed' in col_data:
                        feature_maps['fppg_allowed'][player_id] = float(col_data['FPPG Allowed']) if col_data['FPPG Allowed'] != '0' else 0
                
                # Extract opponent stats for matchup analysis
                elif 'opp last 5 (off)' in table_name:
                    if 'SS Landed/F' in col_data:
                        feature_maps['opp_strikes_per_fight'][player_id] = float(col_data['SS Landed/F']) if col_data['SS Landed/F'] != '0' else 0
                    if 'Strike Acc%' in col_data:
                        feature_maps['opp_strike_accuracy'][player_id] = float(col_data['Strike Acc%']) if col_data['Strike Acc%'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['opp_last5_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0
                    if 'Takedowns/F' in col_data:
                        feature_maps['opp_takedowns_per_fight'][player_id] = float(col_data['Takedowns/F']) if col_data['Takedowns/F'] != '0' else 0
        
        # Add all features to dataframe
        for feature_name, feature_map in feature_maps.items():
            df[feature_name] = df['player_id'].map(feature_map).fillna(0)
        
        # Calculate derived metrics for variability analysis
        df['finishing_rate'] = df.apply(lambda row: max(0, 3 - row['avg_rounds']) / 3 if row['avg_rounds'] > 0 else 0.33, axis=1)
        df['style_score'] = df['strikes_per_min'] / (df['takedowns_per_fight'] + 0.1)  # Striker vs grappler indicator
        df['matchup_advantage'] = df['strike_accuracy'] - df['opp_strike_accuracy']  # Striking matchup edge
        df['takedown_matchup'] = df['takedowns_per_fight'] / (100 - df['takedown_defense'] + 1)  # Takedown vs TDD
        
        # Calculate ITD probability from ML odds and finishing rate
        df['itd_probability'] = 0.35  # Default
        if 'ml_odds' in df.columns:
            # Combine ML odds with finishing rate for better ITD estimates
            df.loc[df['ml_odds'] <= -200, 'itd_probability'] = 0.45 + df.loc[df['ml_odds'] <= -200, 'finishing_rate'] * 0.15
            df.loc[df['ml_odds'] >= 200, 'itd_probability'] = 0.25 + df.loc[df['ml_odds'] >= 200, 'finishing_rate'] * 0.1
        
        # Calculate ITD-adjusted ceiling
        df['itd_adjusted_ceiling'] = df.apply(
            lambda row: row['ceiling'] * (1 + 0.4 * row['itd_probability']), axis=1
        )
        
        
        print(f"   ✅ Added {len(feature_maps)} variability features to {len(df)} fighters")
        
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
            csv_file = processor.csv_path / "extracted.csv"
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