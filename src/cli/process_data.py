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
        """Base implementation with newsletter signals and salary-ownership correlation."""
        df = df.copy()

        # Always add updated columns - copy originals as default
        df['updated_projection'] = df['projection']
        df['updated_floor'] = df['floor']
        df['updated_ceiling'] = df['ceiling']
        df['updated_ownership'] = df['ownership']

        # Apply newsletter signals (universal across sports)
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

        # Apply bounds validation
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

        # Apply salary-ownership correlation model (universal for all sports)
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


class NASCARDataProcessor(BaseDataProcessor):
    """NASCAR-specific data processor."""

    def load_raw_data(self) -> pd.DataFrame:
        """Load and parse NASCAR JSON data into standardized CSV format."""
        # Look for raw.json file in json directory
        main_json_file = self.json_path / "raw.json"

        if not main_json_file.exists():
            raise FileNotFoundError(f"No raw.json file found in {self.json_path}")

        print(f"   📄 Loading JSON: {main_json_file}")

        with open(main_json_file, 'r') as f:
            data = json.load(f)

        # Extract salary data from SalaryContainerJson (NASCAR-specific parsing)
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

            drivers = []

            for player_data in salary_data['Salaries']:
                player_id = player_data['PID']
                salary_id = player_data['Id']  # Used to match with ownership

                # Get ownership from ownership_map using SalaryId
                ownership = ownership_map.get(salary_id, 10.0)

                driver = {
                    'player_id': player_id,
                    'name': player_data['Name'],
                    'salary': player_data['SAL'],
                    'projection': player_data.get('PP', 0.0),  # PP = Projected Points
                    'floor': player_data.get('Floor', 0.0),
                    'ceiling': player_data.get('Ceil', 0.0),
                    'ownership': ownership,
                    'std_dev': 35.0,  # Higher variance for NASCAR (crashes, mechanical issues)
                    'team': player_data.get('TeamName', ''),  # NASCAR team
                    'manufacturer': '',  # Will extract from matchup data
                    'starting_position': 20,  # Default, will extract from matchup data
                    'agg_proj': player_data.get('AggProj', 0.0),  # Aggregate projection
                    'confidence': player_data.get('Conf', 50),  # Projection confidence
                    'game_info': player_data.get('GI', ''),  # Game info with race details
                    'stars': player_data.get('Stars', 3),  # Player star rating
                    'alert_score': player_data.get('AlertScore', 0),  # Alert/attention score
                }
                drivers.append(driver)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ❌ Error parsing SalaryContainerJson: {e}")
            raise ValueError(f"Invalid LineStar data format: {e}")

        # Create base dataframe
        df = pd.DataFrame(drivers)

        # Extract additional NASCAR data from MatchupData (6 tables)
        vegas_odds_map = {}
        starting_pos_map = {}
        practice_speed_map = {}
        track_type_map = {}
        track_info_map = {}

        # Advanced NASCAR features for correlation/variance modeling
        avg_pass_diff_map = {}      # Avg positions gained/lost per race (consistency)
        quality_passes_map = {}     # Passing ability (correlates with advancement)
        fastest_laps_map = {}       # Speed consistency indicator
        practice_lap_time_map = {}  # Practice performance (variance predictor)
        bristol_avg_finish_map = {} # Track-specific performance
        bristol_fp_per_race_map = {}# Track-specific fantasy points
        short_track_avg_map = {}    # Short track category performance

        for table in data.get('MatchupData', []):
            table_name = table.get('Name', '').lower()
            table_id = table.get('Id', '')
            columns = table['Columns']

            for match in table.get('PlayerMatchups', []):
                player_id = match['PlayerId']
                values = match['Values']

                # Map columns to values
                col_data = dict(zip(columns, values))

                # Extract data based on table ID (more reliable than name)
                if table_id == 'pmatchup':  # Practice & Qualifying
                    if 'Qualifying Pos' in col_data and col_data['Qualifying Pos'] not in ['0', '', None]:
                        starting_pos_map[player_id] = int(col_data['Qualifying Pos'])
                    if 'Practice Best Lap Speed' in col_data and col_data['Practice Best Lap Speed'] not in ['0', '', None]:
                        practice_speed_map[player_id] = float(col_data['Practice Best Lap Speed'])
                    if 'Qualifying Best Lap Speed' in col_data and col_data['Qualifying Best Lap Speed'] not in ['0', '', None]:
                        practice_speed_map[player_id] = float(col_data['Qualifying Best Lap Speed'])
                    if 'Vegas Odds' in col_data and col_data['Vegas Odds'] not in ['0', '', None]:
                        vegas_odds_map[player_id] = float(col_data['Vegas Odds'])
                    if 'Practice Best Lap Time' in col_data and col_data['Practice Best Lap Time'] not in ['0', '', None]:
                        practice_lap_time_map[player_id] = float(col_data['Practice Best Lap Time'])

                elif table_id == 'timatchup':  # Track Info
                    if 'Surface' in col_data:
                        track_info_map[player_id] = {
                            'surface': col_data.get('Surface', 'Unknown'),
                            'miles': col_data.get('Miles', '0'),
                            'restrictor_plate': col_data.get('Restrictor Plate?', 'No')
                        }

                        # Determine track type from miles
                        try:
                            miles = float(col_data.get('Miles', '1.5'))
                            if miles >= 2.0:
                                track_type_map[player_id] = 'superspeedway'
                            elif miles <= 1.0:
                                track_type_map[player_id] = 'short'
                            elif 'road' in table_name:
                                track_type_map[player_id] = 'road'
                            else:
                                track_type_map[player_id] = 'intermediate'
                        except (ValueError, TypeError):
                            track_type_map[player_id] = 'intermediate'

                elif table_id == 'tmatchup':  # @ Bristol Motor Speedway table
                    if 'Avg. Place' in col_data and col_data['Avg. Place'] not in ['0', '', None]:
                        bristol_avg_finish_map[player_id] = float(col_data['Avg. Place'])
                    if 'FP/Race' in col_data and col_data['FP/Race'] not in ['0', '', None]:
                        bristol_fp_per_race_map[player_id] = float(col_data['FP/Race'])

                elif table_id == 'stmatchup':  # @ Short Tracks table
                    if 'Avg. Place' in col_data and col_data['Avg. Place'] not in ['0', '', None]:
                        short_track_avg_map[player_id] = float(col_data['Avg. Place'])

                elif 'vegas' in table_name or 'odds' in table_name:
                    # Vegas odds data (if present)
                    if 'Win Odds' in col_data and col_data['Win Odds'] not in ['0', '', None]:
                        vegas_odds_map[player_id] = float(col_data['Win Odds'])

        # Add extracted data to dataframe
        df['starting_position'] = df['player_id'].map(starting_pos_map).fillna(20)
        df['manufacturer'] = 'Unknown'  # Placeholder - not available in LineStar data
        df['win_odds'] = df['player_id'].map(vegas_odds_map).fillna(0)
        df['practice_speed'] = df['player_id'].map(practice_speed_map).fillna(0)

        # Add advanced NASCAR features
        df['practice_lap_time'] = df['player_id'].map(practice_lap_time_map).fillna(16.0)  # ~125mph default
        df['bristol_avg_finish'] = df['player_id'].map(bristol_avg_finish_map).fillna(20.0)
        df['bristol_fp_per_race'] = df['player_id'].map(bristol_fp_per_race_map).fillna(30.0)
        df['short_track_avg_finish'] = df['player_id'].map(short_track_avg_map).fillna(20.0)

        # Add track type (same for all drivers in a race)
        if track_type_map:
            track_type = list(track_type_map.values())[0]  # All drivers have same track type
            df['track_type'] = track_type
        else:
            df['track_type'] = 'intermediate'  # Default fallback

        print(f"   ✅ Parsed {len(df)} drivers from JSON")
        print(f"   🏁 Starting positions: {df['starting_position'].min():.0f} - {df['starting_position'].max():.0f}")
        print(f"   🏭 Manufacturers: {df['manufacturer'].value_counts().to_dict()}")

        return df

    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate NASCAR-specific metrics and variability features."""
        df = df.copy()

        # Load the main JSON data to access MatchupData
        main_json_file = self.json_path / "raw.json"
        with open(main_json_file, 'r') as f:
            data = json.load(f)

        # Initialize feature maps for NASCAR stats
        avg_pass_diff_map = {}
        quality_passes_map = {}
        fastest_laps_map = {}

        feature_maps = {
            # Season stats
            'avg_finish': {},
            'top5_pct': {},
            'top10_pct': {},
            'dnf_pct': {},
            'laps_led_avg': {},
            'season_fppg': {},

            # Advanced season stats (correlation/variance modeling)
            'avg_pass_diff': avg_pass_diff_map,
            'quality_passes_per_race': quality_passes_map,
            'fastest_laps_per_race': fastest_laps_map,

            # Track history
            'track_avg_finish': {},
            'track_top5_pct': {},
            'track_fppg': {},

            # Recent form
            'recent_avg_finish': {},
            'recent_fppg': {},
        }

        # Process MatchupData tables for detailed stats
        for table in data.get('MatchupData', []):
            table_name = table.get('Name', '').lower()
            table_id = table.get('Id', '')
            columns = table.get('Columns', [])

            for match in table.get('PlayerMatchups', []):
                player_id = match['PlayerId']
                values = match['Values']
                col_data = dict(zip(columns, values))

                # Extract season stats (Driver Season table)
                if table_id == 'dSeason':
                    if 'Avg Finish' in col_data:
                        feature_maps['avg_finish'][player_id] = float(col_data['Avg Finish']) if col_data['Avg Finish'] != '0' else 20
                    if 'Top Fives' in col_data and 'Races' in col_data:
                        try:
                            top5_pct = (float(col_data['Top Fives']) / float(col_data['Races'])) * 100
                            feature_maps['top5_pct'][player_id] = top5_pct
                        except (ValueError, ZeroDivisionError):
                            feature_maps['top5_pct'][player_id] = 0
                    if 'Top Tens' in col_data and 'Races' in col_data:
                        try:
                            top10_pct = (float(col_data['Top Tens']) / float(col_data['Races'])) * 100
                            feature_maps['top10_pct'][player_id] = top10_pct
                        except (ValueError, ZeroDivisionError):
                            feature_maps['top10_pct'][player_id] = 0
                    if 'Laps Led/Race' in col_data:
                        feature_maps['laps_led_avg'][player_id] = float(col_data['Laps Led/Race']) if col_data['Laps Led/Race'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['season_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0

                    # Advanced season stats for correlation/variance modeling
                    if 'Avg Pass Diff' in col_data:
                        avg_pass_diff_map[player_id] = float(col_data['Avg Pass Diff']) if col_data['Avg Pass Diff'] != '0' else 0
                    if 'Quality Passes/Race' in col_data:
                        quality_passes_map[player_id] = float(col_data['Quality Passes/Race']) if col_data['Quality Passes/Race'] != '0' else 0
                    if 'Fastest Laps/Race' in col_data:
                        fastest_laps_map[player_id] = float(col_data['Fastest Laps/Race']) if col_data['Fastest Laps/Race'] != '0' else 0

                # Extract track-specific stats (@ Bristol Motor Speedway table)
                elif table_id == 'tmatchup':
                    if 'Avg Finish' in col_data:
                        feature_maps['track_avg_finish'][player_id] = float(col_data['Avg Finish']) if col_data['Avg Finish'] != '0' else 20
                    if 'Top 5s' in col_data and 'Races' in col_data:
                        try:
                            track_top5_pct = (float(col_data['Top 5s']) / float(col_data['Races'])) * 100
                            feature_maps['track_top5_pct'][player_id] = track_top5_pct
                        except (ValueError, ZeroDivisionError):
                            feature_maps['track_top5_pct'][player_id] = 0
                    if 'FPPG' in col_data:
                        feature_maps['track_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0

                # Extract recent form (Driver Last Season table as proxy)
                elif table_id == 'dLastSeason':
                    if 'Avg Finish' in col_data:
                        feature_maps['recent_avg_finish'][player_id] = float(col_data['Avg Finish']) if col_data['Avg Finish'] != '0' else 20
                    if 'FPPG' in col_data:
                        feature_maps['recent_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['season_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0

                # Extract track history
                elif 'track history' in table_name or 'track stats' in table_name:
                    if 'Avg Finish' in col_data:
                        feature_maps['track_avg_finish'][player_id] = float(col_data['Avg Finish']) if col_data['Avg Finish'] != '0' else 20
                    if 'Top 5%' in col_data:
                        feature_maps['track_top5_pct'][player_id] = float(col_data['Top 5%']) if col_data['Top 5%'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['track_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0

                # Extract recent form
                elif 'last 5' in table_name or 'recent' in table_name:
                    if 'Avg Finish' in col_data:
                        feature_maps['recent_avg_finish'][player_id] = float(col_data['Avg Finish']) if col_data['Avg Finish'] != '0' else 20
                    if 'FPPG' in col_data:
                        feature_maps['recent_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0

        # Add all features to dataframe
        for feature_name, feature_map in feature_maps.items():
            df[feature_name] = df['player_id'].map(feature_map).fillna(0)

        # Calculate derived metrics for NASCAR
        # Position differential potential (key NASCAR metric)
        df['position_differential'] = df['starting_position'] - 15  # Target: finish in top 15
        df['pd_upside'] = np.maximum(0, df['position_differential']) * 0.5  # Bonus for advancing

        # Dominator potential (drivers who lead laps and finish well)
        df['dominator_score'] = (
            (df['laps_led_avg'] / 10) +  # Normalize laps led
            (1 - df['avg_finish'] / 40) * 10  # Inverse of avg finish
        )

        # Track type detection (simplified - would need actual track data)
        # For now, use proxy based on typical speeds/characteristics
        track_type = 'intermediate'  # Default
        df['track_type'] = track_type

        # Variance adjustments based on starting position
        df['variance_multiplier'] = df.apply(
            lambda row: 0.9 if row['starting_position'] <= 5 else  # Front runners more consistent
                        1.2 if row['starting_position'] >= 25 else  # Back of pack more volatile
                        1.0,  # Mid-pack standard
            axis=1
        )

        # Calculate ceiling adjustments for NASCAR
        # Drivers starting outside top 10 have higher ceiling (position differential upside)
        df['ceiling_adjustment'] = df.apply(
            lambda row: 1.15 if row['starting_position'] > 15 else
                        1.05 if row['starting_position'] > 10 else
                        1.0,
            axis=1
        )

        # 🛡️ SYNTHESIZED FLOOR CALCULATION (NASCAR-specific)
        # Real floors based on DNF risk, starting position, and track reliability
        def calculate_nascar_floor(row):
            base_projection = row['projection']
            starting_pos = row['starting_position']

            # DNF risk by starting position (empirical NASCAR data)
            if starting_pos <= 5:
                dnf_risk = 0.08  # Front runners: 8% DNF risk
            elif starting_pos <= 15:
                dnf_risk = 0.12  # Mid-pack: 12% DNF risk
            elif starting_pos <= 25:
                dnf_risk = 0.18  # Back-mid: 18% DNF risk
            else:
                dnf_risk = 0.25  # Back-pack: 25% DNF risk (equipment/lapped)

            # Track-specific adjustments (Bristol is relatively safe)
            track_safety_factor = 0.85  # Bristol has fewer multi-car crashes
            adjusted_dnf_risk = dnf_risk * track_safety_factor

            # More realistic NASCAR floor calculation
            # Base floor from finish position points (42 pts for 1st down to 4 pts for 39th)
            if starting_pos <= 5:
                # Front runners: Even with issues, likely 15th-25th finish
                base_finish_floor = 25  # 15-20 pts from finish + some laps/fastest laps
            elif starting_pos <= 15:
                # Mid-pack: Issues likely mean 25th-35th finish
                base_finish_floor = 15  # 8-15 pts from finish
            elif starting_pos <= 25:
                # Back-mid: Issues mean 30th+ finish
                base_finish_floor = 8   # 4-12 pts from finish
            else:
                # Back-pack: High risk of being lapped/DNF
                base_finish_floor = 3   # Minimum points if they finish

            # Apply DNF risk
            reliability_factor = 1 - adjusted_dnf_risk
            calculated_floor = reliability_factor * base_finish_floor

            # Add small bonus for drivers with track history (reduces floor risk)
            if hasattr(row, 'bristol_fp_per_race') and row['bristol_fp_per_race'] > 35:
                calculated_floor *= 1.2  # Track specialists have higher floors

            # Floor can't exceed 30% of projection (prevents unrealistic floors)
            max_floor = base_projection * 0.30

            return min(calculated_floor, max_floor)

        df['synthesized_floor'] = df.apply(calculate_nascar_floor, axis=1)

        # Replace the zero floors with synthesized floors
        df['floor'] = df['synthesized_floor']

        print(f"   ✅ Added {len(feature_maps)} NASCAR features to {len(df)} drivers")

        return df

    def get_position_from_data(self, row: pd.Series) -> Position:
        """NASCAR drivers are all DRIVER position."""
        return Position.DRIVER

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NASCAR-specific projection adjustments."""
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

        # Apply NASCAR-specific adjustments
        print("   🏁 Applying NASCAR-specific adjustments...")

        # 1. Position differential upside influences ceiling
        pd_boost = df['pd_upside'] > 0
        df.loc[pd_boost, 'updated_ceiling'] *= (1 + df.loc[pd_boost, 'pd_upside'] * 0.02)

        # 2. Dominator score influences projection
        dom_boost = df['dominator_score'] > 5
        df.loc[dom_boost, 'updated_projection'] *= (1 + (df.loc[dom_boost, 'dominator_score'] - 5) * 0.01)

        # 3. Track history influences floor
        track_experts = df['track_avg_finish'] < 10
        df.loc[track_experts, 'updated_floor'] *= 1.1

        # 4. DNF risk influences floor negatively
        high_dnf = df['dnf_pct'] > 15
        df.loc[high_dnf, 'updated_floor'] *= (1 - df.loc[high_dnf, 'dnf_pct'] / 200)

        # 5. Apply ceiling adjustments from calculate_sport_metrics
        df['updated_ceiling'] *= df['ceiling_adjustment']

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

        return df


class MMADataProcessor(BaseDataProcessor):
    """MMA-specific data processor."""

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MMA-specific projection adjustments."""
        # First apply base adjustments (newsletter + salary-ownership)
        df = super().apply_projection_adjustments(df)

        # Then apply MMA-specific derived metric influences
        print("   🥊 Applying MMA-specific adjustments...")

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

        # Apply takedown matchup ceiling adjustments (MMA-specific)
        print("   🤼 Applying takedown matchup ceiling adjustments...")
        df = self._apply_takedown_ceiling_adjustments(df)

        # Final bounds validation after MMA-specific adjustments
        df['updated_ownership'] = np.clip(df['updated_ownership'], 0, 100)
        mask = df['updated_ceiling'] < df['updated_projection']
        df.loc[mask, 'updated_ceiling'] = df.loc[mask, 'updated_projection'] * 1.1
        mask = df['updated_floor'] > df['updated_projection']
        df.loc[mask, 'updated_floor'] = df.loc[mask, 'updated_projection'] * 0.8

        return df

    def _apply_takedown_ceiling_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ceiling adjustments based on takedown matchup disadvantages."""
        df = df.copy()

        for idx, row in df.iterrows():
            takedown_matchup = row['takedown_matchup']

            # If opponent will dominate takedowns (high score), suppress ceiling
            if takedown_matchup > 5.0:  # Opponent gets 5+ takedowns vs player's defense
                # Severe ceiling suppression for terrible matchups
                ceiling_reduction = min(0.40, (takedown_matchup - 5) * 0.08)  # Up to 40% reduction
                df.loc[idx, 'updated_ceiling'] *= (1 - ceiling_reduction)

            elif takedown_matchup > 3.0:  # Moderate takedown disadvantage
                # Moderate ceiling suppression
                ceiling_reduction = min(0.25, (takedown_matchup - 3) * 0.10)  # Up to 25% reduction
                df.loc[idx, 'updated_ceiling'] *= (1 - ceiling_reduction)

            elif takedown_matchup > 1.5:  # Slight takedown disadvantage
                # Minor ceiling suppression
                ceiling_reduction = min(0.15, (takedown_matchup - 1.5) * 0.08)  # Up to 15% reduction
                df.loc[idx, 'updated_ceiling'] *= (1 - ceiling_reduction)

        # Display notable adjustments
        severe_disadvantages = df[df['takedown_matchup'] > 5.0]
        if not severe_disadvantages.empty:
            print(f"      🚨 Severe takedown disadvantages: {len(severe_disadvantages)} fighters")
            for _, fighter in severe_disadvantages.iterrows():
                print(f"         {fighter['name']}: {fighter['takedown_matchup']:.1f} TD matchup")

        return df

    def _calculate_takedown_matchup(self, df: pd.DataFrame) -> pd.Series:
        """Calculate proper opponent-based takedown matchup scores."""
        takedown_scores = []

        for _, row in df.iterrows():
            player_td_defense = row['takedown_defense']
            opponent_team = row.get('oteam', '')  # Use opponent team field

            if opponent_team:
                # Find opponent using team-based matching
                opponent_row = df[df['hteam'] == opponent_team]
                if not opponent_row.empty:
                    opp_td_per_fight = opponent_row.iloc[0]['takedowns_per_fight']

                    # Calculate actual matchup: opponent TD rate vs player TD defense
                    # Higher score = opponent will dominate takedowns (bad for player ceiling)
                    matchup_score = opp_td_per_fight / (player_td_defense / 100 + 0.1)
                    takedown_scores.append(matchup_score)
                else:
                    # No opponent found, use self-referential as fallback
                    takedown_scores.append(row['takedowns_per_fight'] / (100 - player_td_defense + 1))
            else:
                # No opponent team listed, use self-referential as fallback
                takedown_scores.append(row['takedowns_per_fight'] / (100 - player_td_defense + 1))

        return pd.Series(takedown_scores, index=df.index)

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
        
        # Extract opponent relationships using OTEAM-to-lastname matching
        opponent_map = {}
        opponent_odds_map = {}

        # Create lastname-to-fighter mapping for OTEAM matching
        lastname_to_fighter = {}
        for _, row in df.iterrows():
            # Extract last name (handle multi-part names like "Jose Daniel Medina")
            last_name = row['name'].split()[-1].upper()
            lastname_to_fighter[last_name] = {
                'player_id': row['player_id'],
                'name': row['name'],
                'ml_odds': row['ml_odds']
            }

        # Match fighters to opponents using OTEAM values
        matched_fighters = set()
        for _, row in df.iterrows():
            player_id = row['player_id']
            player_oteam = row['oteam']

            # Skip if already matched or no OTEAM data
            if player_id in matched_fighters or not player_oteam:
                continue

            # Find opponent by matching OTEAM to fighter lastname
            # Special handling for multi-word names like "DANIEL MEDINA"
            potential_opponents = []

            # First try exact OTEAM match
            if player_oteam in lastname_to_fighter:
                potential_opponents.append(lastname_to_fighter[player_oteam])

            # For multi-word OTEAM like "DANIEL MEDINA", try to find "Jose Daniel Medina"
            elif ' ' in player_oteam:
                for fighter_name, fighter_data in lastname_to_fighter.items():
                    full_fighter_name = fighter_data['name'].upper()
                    if player_oteam in full_fighter_name:
                        potential_opponents.append(fighter_data)

            # Match with the first valid opponent found
            for opponent_data in potential_opponents:
                opponent_id = opponent_data['player_id']

                # Don't match a fighter to themselves and ensure unique matching
                if opponent_id != player_id and opponent_id not in matched_fighters:
                    opponent_map[player_id] = opponent_data['name']
                    opponent_odds_map[player_id] = opponent_data['ml_odds']
                    opponent_map[opponent_id] = row['name']
                    opponent_odds_map[opponent_id] = row['ml_odds']

                    matched_fighters.add(player_id)
                    matched_fighters.add(opponent_id)
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
        # Calculate proper opponent-based takedown matchup
        df['takedown_matchup'] = self._calculate_takedown_matchup(df)
        
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


class NFLDataProcessor(BaseDataProcessor):
    """NFL-specific data processor."""

    def load_raw_data(self) -> pd.DataFrame:
        """Load and parse NFL JSON data into standardized CSV format."""
        # Look for raw.json file in json directory
        main_json_file = self.json_path / "raw.json"

        if not main_json_file.exists():
            raise FileNotFoundError(f"No raw.json file found in {main_json_file}")

        print(f"   📄 Loading JSON: {main_json_file}")

        with open(main_json_file, 'r') as f:
            data = json.load(f)

        # Extract salary data from SalaryContainerJson
        try:
            salary_data = json.loads(data['SalaryContainerJson'])

            # Extract ownership data from Ownership section
            ownership_map = {}
            if 'Ownership' in data and 'Projected' in data['Ownership']:
                projected = data['Ownership']['Projected']
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
                salary_id = player_data['Id']

                # Get ownership from ownership_map
                ownership = ownership_map.get(salary_id, 10.0)

                player = {
                    'player_id': player_id,
                    'name': player_data['Name'],
                    'salary': player_data['SAL'],
                    'projection': player_data.get('PP', 0.0),
                    'floor': player_data.get('Floor', 0.0),
                    'ceiling': player_data.get('Ceil', 0.0),
                    'ownership': ownership,
                    'std_dev': 25.0,  # NFL variance (lower than NASCAR/MMA)
                    'position': player_data.get('POS', ''),
                    'team': player_data.get('PTEAM', ''),
                    'opponent': player_data.get('OTEAM', ''),
                    'game_id': player_data.get('GID', ''),
                    'game_info': player_data.get('GI', ''),
                    'home_team': player_data.get('HTEAM', ''),
                    'agg_proj': player_data.get('AggProj', 0.0),
                    'confidence': player_data.get('Conf', 50),
                    'stars': player_data.get('Stars', 3),
                    'alert_score': player_data.get('AlertScore', 0),
                    'opponent_rank': player_data.get('OppRank', 16),
                    'points_per_game': player_data.get('PPG', 0.0),
                }
                players.append(player)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ❌ Error parsing SalaryContainerJson: {e}")
            raise ValueError(f"Invalid NFL data format: {e}")

        # Create base dataframe
        df = pd.DataFrame(players)

        # Filter out players with zero salary (inactive/IR players)
        original_count = len(df)
        df = df[df['salary'] > 0]
        filtered_count = len(df)

        if original_count > filtered_count:
            print(f"   ⚠️  Filtered out {original_count - filtered_count} players with $0 salary")

        # Extract critical NFL features from MatchupData for correlation/variance modeling
        self._extract_nfl_matchup_features(df, data)

        print(f"   ✅ Loaded {len(df)} active NFL players with advanced metrics")
        return df

    def _extract_nfl_matchup_features(self, df: pd.DataFrame, raw_data: dict):
        """Extract NFL-specific features from MatchupData for correlation modeling."""

        # Initialize feature maps
        target_share_map = {}           # WR/TE target volume
        snap_pct_map = {}              # Playing time correlation
        rz_usage_map = {}              # Red zone TD correlation
        matchup_grade_map = {}         # Defensive matchup quality
        pace_metrics_map = {}          # Game flow indicators
        home_away_map = {}             # Venue variance adjustments

        for table in raw_data.get('MatchupData', []):
            table_name = table.get('Name', '').lower()
            columns = table.get('Columns', [])

            for player_matchup in table.get('PlayerMatchups', []):
                player_id = player_matchup.get('PlayerId')
                values = player_matchup.get('Values', [])

                if not player_id or len(values) != len(columns):
                    continue

                player_stats = dict(zip(columns, values))

                # Extract Flex usage patterns (critical for WR/RB/TE correlation)
                if 'flex' in table_name and 'season' in table_name:
                    target_share_map[player_id] = {
                        'targets_per_game': self._safe_float(player_stats.get('Targets/G', 0)),
                        'targets_per_salary': self._safe_float(player_stats.get('Targets/Salary', 0)),
                        'receptions_per_game': self._safe_float(player_stats.get('Receptions/G', 0)),
                        'rush_attempts_per_game': self._safe_float(player_stats.get('Rushes/G', 0)),
                        'total_tds': self._safe_float(player_stats.get('Touchdowns', 0)),
                        'home_away': player_stats.get('Home/Away', '0.0')
                    }

                # Extract snap counts and red zone usage (last game for recency)
                elif 'snaps & redzone' in table_name:
                    snap_pct_map[player_id] = {
                        'snap_pct': self._safe_float(player_stats.get('Snap%', 0)),
                        'snaps_per_salary': self._safe_float(player_stats.get('Snaps/Sal', 0)),
                        'rz_targets': self._safe_float(player_stats.get('RZ Targets', 0)),
                        'rz_catches': self._safe_float(player_stats.get('RZ Catches', 0)),
                        'rz_rushes': self._safe_float(player_stats.get('RZ Rushes', 0))
                    }

                # Extract WR-specific matchup advantages (shadow coverage, speed)
                elif 'wr matchup' in table_name:
                    matchup_grade_map[player_id] = {
                        'speed_advantage': self._safe_float(player_stats.get('Speed Adv', 0)),
                        'shadow_coverage': player_stats.get('Shadow', '') != '',  # Boolean
                        'slot_route_pct': self._safe_float(player_stats.get('SlotRoute%', 0)),
                        'def_grade': self._safe_float(player_stats.get('Def Grade', 0))
                    }

                # Extract opponent allowed FPPG (matchup leverage for all positions)
                elif 'opp vs' in table_name and 'season' in table_name:
                    pos_type = 'rb' if 'rb' in table_name else 'wr' if 'wr' in table_name else 'te'
                    if player_id not in pace_metrics_map:
                        pace_metrics_map[player_id] = {}
                    pace_metrics_map[player_id][f'{pos_type}_fppg_allowed'] = self._safe_float(
                        player_stats.get('FPPG Allowed', 0)
                    )
                    pace_metrics_map[player_id][f'{pos_type}_opp_rank'] = self._safe_float(
                        player_stats.get('Opp Rank', 16)
                    )

        # Apply extracted features to dataframe
        df['targets_per_game'] = df['player_id'].map(
            lambda x: target_share_map.get(x, {}).get('targets_per_game', 0)
        )
        df['snap_pct'] = df['player_id'].map(
            lambda x: snap_pct_map.get(x, {}).get('snap_pct', 50.0)  # Default 50%
        )
        df['rz_targets'] = df['player_id'].map(
            lambda x: snap_pct_map.get(x, {}).get('rz_targets', 0)
        )
        df['speed_advantage'] = df['player_id'].map(
            lambda x: matchup_grade_map.get(x, {}).get('speed_advantage', 0)
        )
        df['shadow_coverage'] = df['player_id'].map(
            lambda x: matchup_grade_map.get(x, {}).get('shadow_coverage', False)
        )

        # Position-specific matchup metrics
        df['matchup_fppg_allowed'] = 0.0
        for pos in ['rb', 'wr', 'te']:
            pos_mask = df['position'].str.lower() == pos
            df.loc[pos_mask, 'matchup_fppg_allowed'] = df.loc[pos_mask, 'player_id'].map(
                lambda x: pace_metrics_map.get(x, {}).get(f'{pos}_fppg_allowed', 15.0)
            )

        return df

    def _safe_float(self, value, default=0.0):
        """Safely convert value to float."""
        try:
            return float(value) if value != '' else default
        except (ValueError, TypeError):
            return default

    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate NFL-specific metrics."""
        df = df.copy()

        # NFL position variance multipliers (QB most consistent, DST most volatile)
        position_variance = {
            'QB': 0.8,   # Low variance - most predictable scoring
            'RB': 1.0,   # Baseline variance
            'WR': 1.1,   # Slightly higher variance
            'TE': 1.0,   # Similar to RB
            'K': 1.3,    # Higher variance - weather/game script dependent
            'DST': 1.5   # Highest variance - boom/bust potential
        }

        # Apply position-based variance adjustments
        df['variance_multiplier'] = df['position'].map(position_variance).fillna(1.0)
        df['adjusted_std_dev'] = df['std_dev'] * df['variance_multiplier']

        # Calculate ceiling adjustments based on position and matchup
        df['ceiling_adjustment'] = 1.0

        # QB ceiling boost for high-total games (extract from game_info if available)
        qb_mask = df['position'] == 'QB'
        df.loc[qb_mask, 'ceiling_adjustment'] = 1.1

        # WR/TE ceiling boost vs weak pass defense (simplified - use opponent rank)
        pass_catcher_mask = df['position'].isin(['WR', 'TE'])
        weak_defense_mask = df['opponent_rank'] >= 20  # Bottom defenses
        df.loc[pass_catcher_mask & weak_defense_mask, 'ceiling_adjustment'] = 1.15

        # RB ceiling boost vs weak run defense
        rb_mask = df['position'] == 'RB'
        df.loc[rb_mask & weak_defense_mask, 'ceiling_adjustment'] = 1.1

        # DST ceiling boost vs weak offense (lower opponent rank = weaker offense)
        dst_mask = df['position'] == 'DST'
        weak_offense_mask = df['opponent_rank'] <= 10
        df.loc[dst_mask & weak_offense_mask, 'ceiling_adjustment'] = 1.2

        # Apply ceiling adjustments
        df['synthesized_floor'] = df['floor']
        df['adjusted_ceiling'] = df['ceiling'] * df['ceiling_adjustment']

        # Engineer NFL-specific advanced features
        df = self._engineer_nfl_features(df)

        return df

    def _engineer_nfl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer NFL-specific advanced features from existing data."""
        print("   🔧 Engineering NFL value efficiency metrics...")

        # Value Efficiency Metrics
        df['ceiling_per_dollar'] = (df['adjusted_ceiling'] / df['salary']) * 1000  # Ceiling points per $1K
        df['projection_per_ownership'] = df['projection'] / df['ownership']  # Leverage ratio
        df['salary_vs_projection_ratio'] = df['salary'] / (df['projection'] * 1000)  # Pricing efficiency

        print("   📊 Engineering NFL risk-adjusted metrics...")

        # Risk-Adjusted Metrics (avoid division by zero)
        df['sharpe_ratio'] = np.where(df['adjusted_std_dev'] > 0,
                                      (df['projection'] - df['floor']) / df['adjusted_std_dev'], 0)
        df['ceiling_volatility'] = np.where(df['projection'] > 0,
                                            (df['adjusted_ceiling'] - df['projection']) / df['projection'], 0)
        df['confidence_weighted_proj'] = df['projection'] * (df['confidence'] / 100.0)

        print("   🎯 Engineering NFL position-specific features...")

        # Position-Specific Features
        # RB: Goal line value (RZ targets × snap share)
        df['goal_line_value'] = np.where(df['position'] == 'RB',
                                         df['rz_targets'] * (df['snap_pct'] / 100.0), 0)

        # WR/TE: Target efficiency (targets per snap)
        df['target_efficiency'] = np.where(df['position'].isin(['WR', 'TE']) & (df['snap_pct'] > 0),
                                           df['targets_per_game'] / (df['snap_pct'] / 100.0), 0)

        # QB: Passing volume indicator
        df['qb_volume_indicator'] = np.where(df['position'] == 'QB',
                                             df['targets_per_game'] * df['snap_pct'] / 100.0, 0)

        print("   🏈 Engineering NFL game context features...")

        # Game Context Features
        df['is_home'] = (df['team'] == df['home_team']).astype(int)

        # Calculate game totals for pace proxy
        game_totals = df.groupby('game_id')['projection'].sum().to_dict()
        df['game_pace_proxy'] = df['game_id'].map(game_totals)

        # Team stack potential (sum of ceiling for same team)
        team_ceilings = df.groupby('team')['adjusted_ceiling'].sum().to_dict()
        df['team_stack_ceiling'] = df['team'].map(team_ceilings)

        print("   🎪 Engineering NFL matchup leverage...")

        # Matchup & Leverage Features
        df['defensive_mismatch'] = df['matchup_fppg_allowed'] - (df['opponent_rank'] * 0.5)
        df['ownership_mispricing'] = df['ownership'] - (df['projection'] * 2)

        # Newsletter signal strength (if available)
        if 'newsletter_signal' in df.columns and 'newsletter_confidence' in df.columns:
            signal_multipliers = {'target': 1.5, 'fade': -1.0, 'volatile': 1.2, 'neutral': 0.0}
            df['signal_strength'] = df['newsletter_signal'].map(signal_multipliers).fillna(0) * df['newsletter_confidence']

        # Contrarian boost (if ownership_delta exists)
        if 'ownership_delta' in df.columns:
            df['contrarian_boost'] = df['ownership_delta'] * -1
        else:
            df['contrarian_boost'] = 0.0

        print(f"   ✅ Engineered 14 NFL-specific advanced features")

        return df

    def get_position_from_data(self, row: pd.Series) -> Position:
        """Extract NFL position from raw data."""
        pos_str = row.get('position', '').upper()

        position_map = {
            'QB': Position.QB,
            'RB': Position.RB,
            'WR': Position.WR,
            'TE': Position.TE,
            'K': Position.K,
            'DST': Position.DST,
            'D': Position.DST,
            'DEF': Position.DST
        }

        return position_map.get(pos_str, Position.FLEX)


def create_processor(sport: str, pid: str, site: str = 'dk') -> BaseDataProcessor:
    """Factory function to create sport-specific processor."""
    processors = {
        'mma': MMADataProcessor,
        'nascar': NASCARDataProcessor,
        'nfl': NFLDataProcessor,
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
                       choices=['mma', 'nascar', 'nfl', 'nba'],
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