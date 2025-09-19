"""NASCAR-specific data processor for DFS lineup optimization."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import BaseDataProcessor
from ..models.player import Position


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

    def get_position_from_data(self, _row: pd.Series) -> Position:
        """NASCAR drivers are all DRIVER position."""
        return Position.DRIVER

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NASCAR-specific projection adjustments."""
        # Use base class implementation which handles all newsletter signals
        df = super().apply_projection_adjustments(df)

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