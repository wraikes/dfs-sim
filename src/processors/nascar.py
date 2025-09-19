"""NASCAR-specific data processor for DFS lineup optimization."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseDataProcessor
from ..models.player import Position
from ..config.nascar_config import NASCAR_CONFIG


class NASCARDataProcessor(BaseDataProcessor):
    """NASCAR-specific data processor."""

    def load_raw_data(self) -> pd.DataFrame:
        """Load and parse NASCAR JSON data into standardized CSV format."""
        # Look for raw.json file in json directory
        main_json_file = self.json_path / "raw.json"

        if not main_json_file.exists():
            raise FileNotFoundError(f"No raw.json file found in {self.json_path}")

        logger.info(f"Loading NASCAR JSON: {main_json_file}")

        with open(main_json_file, 'r') as f:
            data = json.load(f)

        # Extract salary data from SalaryContainerJson
        try:
            salary_data = json.loads(data['SalaryContainerJson'])

            # Extract ownership data (could be moved to base class helper)
            ownership_map = self._extract_ownership_data(data)

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
                    'projection': player_data.get('PP'),  # NULL if not present
                    'floor': player_data.get('Floor'),  # NULL if not present
                    'ceiling': player_data.get('Ceil'),  # NULL if not present
                    'ownership': ownership,
                    'team': player_data.get('TeamName'),  # NASCAR team
                    'manufacturer': None,  # Will extract from matchup data
                    'starting_position': None,  # Will extract from matchup data
                    'agg_proj': player_data.get('AggProj'),  # Aggregate projection
                    'confidence': player_data.get('Conf'),  # Projection confidence (0-100 scale)
                    'game_info': player_data.get('GI'),  # Game info with race details
                    'stars': player_data.get('Stars'),  # Player star rating
                    'alert_score': player_data.get('AlertScore'),  # Alert/attention score
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
                    if 'Qualifying Pos' in col_data:
                        starting_pos_map[player_id] = int(self._safe_float(col_data['Qualifying Pos'], 20))
                    if 'Practice Best Lap Speed' in col_data:
                        practice_speed_map[player_id] = self._safe_float(col_data['Practice Best Lap Speed'])
                    if 'Qualifying Best Lap Speed' in col_data:
                        practice_speed_map[player_id] = self._safe_float(col_data['Qualifying Best Lap Speed'])
                    if 'Vegas Odds' in col_data:
                        vegas_odds_map[player_id] = self._safe_float(col_data['Vegas Odds'])
                    if 'Practice Best Lap Time' in col_data:
                        practice_lap_time_map[player_id] = self._safe_float(col_data['Practice Best Lap Time'], 16.0)

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

                elif table_id == 'tmatchup':  # Track-specific table (varies by race)
                    # Extract track-specific data dynamically
                    if 'Avg. Place' in col_data or 'Avg Finish' in col_data:
                        avg_key = 'Avg. Place' if 'Avg. Place' in col_data else 'Avg Finish'
                        track_info_map[player_id] = track_info_map.get(player_id, {})
                        track_info_map[player_id]['track_avg_finish'] = self._safe_float(col_data[avg_key], 20.0)
                    if 'FP/Race' in col_data or 'FPPG' in col_data:
                        fppg_key = 'FP/Race' if 'FP/Race' in col_data else 'FPPG'
                        track_info_map[player_id] = track_info_map.get(player_id, {})
                        track_info_map[player_id]['track_fppg'] = self._safe_float(col_data[fppg_key], 30.0)

                elif 'vegas' in table_name or 'odds' in table_name:
                    # Vegas odds data (if present)
                    if 'Win Odds' in col_data:
                        vegas_odds_map[player_id] = self._safe_float(col_data['Win Odds'])

        # Add extracted data to dataframe - keep as NULL if not found
        df['starting_position'] = df['player_id'].map(starting_pos_map)
        df['manufacturer'] = None  # Not available in LineStar data
        df['win_odds'] = df['player_id'].map(vegas_odds_map)
        df['practice_speed'] = df['player_id'].map(practice_speed_map)
        df['practice_lap_time'] = df['player_id'].map(practice_lap_time_map)

        # Add track type (same for all drivers in a race)
        if track_type_map:
            track_type = list(track_type_map.values())[0]  # All drivers have same track type
            df['track_type'] = track_type
        else:
            df['track_type'] = None  # No track type data available

        print(f"   ✅ Parsed {len(df)} drivers from JSON")
        print(f"   🏁 Starting positions: {df['starting_position'].min():.0f} - {df['starting_position'].max():.0f}")
        print(f"   🏭 Manufacturers: {df['manufacturer'].value_counts().to_dict()}")

        return df

    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate NASCAR-specific metrics from already extracted data."""
        df = df.copy()

        # First, validate core fields and filter out incomplete players
        valid_players = df.dropna(subset=['projection'])
        invalid_players = df[df['projection'].isna()]

        if not invalid_players.empty:
            logger.warning(f"Omitting {len(invalid_players)} players due to missing projections:")
            for _, player in invalid_players.iterrows():
                logger.warning(f"  - {player['name']}: missing projection data")

        df = valid_players.copy()
        if df.empty:
            logger.error("No valid players found - all missing projection data")
            return df

        # Re-parse JSON only for additional stats not captured in load_raw_data
        # This is necessary because load_raw_data focuses on basic player info
        # while this method extracts detailed performance metrics
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

            # Track history (extracted dynamically from current track table)
            'track_avg_finish': {},
            'track_top5_pct': {},
            'track_fppg': {},

            # Recent form
            'recent_avg_finish': {},
            'recent_fppg': {},

            # DNF risk calculation (derive from available data)
            'total_races': {},
            'dnf_pct': {},
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
                        feature_maps['avg_finish'][player_id] = self._safe_float(col_data['Avg Finish'], 20.0)
                    if 'Top Fives' in col_data and 'Races' in col_data:
                        top_fives = self._safe_float(col_data['Top Fives'])
                        races = self._safe_float(col_data['Races'], 1.0)  # Avoid division by zero
                        feature_maps['top5_pct'][player_id] = (top_fives / races) * 100 if races > 0 else 0
                    if 'Top Tens' in col_data and 'Races' in col_data:
                        top_tens = self._safe_float(col_data['Top Tens'])
                        races = self._safe_float(col_data['Races'], 1.0)
                        feature_maps['top10_pct'][player_id] = (top_tens / races) * 100 if races > 0 else 0
                    if 'Laps Led/Race' in col_data:
                        feature_maps['laps_led_avg'][player_id] = self._safe_float(col_data['Laps Led/Race'])
                    if 'FPPG' in col_data:
                        feature_maps['season_fppg'][player_id] = self._safe_float(col_data['FPPG'])

                    # Calculate DNF risk from race participation vs finishes
                    if 'Races' in col_data:
                        races = self._safe_float(col_data['Races'])
                        feature_maps['total_races'][player_id] = races
                        # Estimate DNF rate: drivers with very low FPPG relative to their skill level
                        # This is a proxy since LineStar doesn't provide direct DNF data
                        avg_finish = self._safe_float(col_data.get('Avg Finish', 20))
                        fppg = self._safe_float(col_data.get('FPPG', 0))
                        if races > 5 and avg_finish > 0:  # Enough sample size
                            # Expected FPPG based on average finish (rough estimate)
                            expected_fppg = max(0, 50 - avg_finish * 1.2)  # Linear relationship
                            if expected_fppg > 0:
                                fppg_deficit = max(0, expected_fppg - fppg) / expected_fppg
                                # Convert deficit to estimated DNF rate (capped at 30%)
                                estimated_dnf = min(30, fppg_deficit * 25)
                                feature_maps['dnf_pct'][player_id] = estimated_dnf

                    # Advanced season stats for correlation/variance modeling
                    if 'Avg Pass Diff' in col_data:
                        avg_pass_diff_map[player_id] = self._safe_float(col_data['Avg Pass Diff'])
                    if 'Quality Passes/Race' in col_data:
                        quality_passes_map[player_id] = self._safe_float(col_data['Quality Passes/Race'])
                    if 'Fastest Laps/Race' in col_data:
                        fastest_laps_map[player_id] = self._safe_float(col_data['Fastest Laps/Race'])

                # Extract track-specific stats (dynamic track table)
                elif table_id == 'tmatchup':
                    if 'Avg Finish' in col_data or 'Avg. Place' in col_data:
                        avg_key = 'Avg Finish' if 'Avg Finish' in col_data else 'Avg. Place'
                        feature_maps['track_avg_finish'][player_id] = self._safe_float(col_data[avg_key], 20.0)
                    if ('Top 5s' in col_data or 'Top Fives' in col_data) and 'Races' in col_data:
                        top5_key = 'Top 5s' if 'Top 5s' in col_data else 'Top Fives'
                        top_5s = self._safe_float(col_data[top5_key])
                        races = self._safe_float(col_data['Races'], 1.0)
                        feature_maps['track_top5_pct'][player_id] = (top_5s / races) * 100 if races > 0 else 0
                    if 'FPPG' in col_data or 'FP/Race' in col_data:
                        fppg_key = 'FPPG' if 'FPPG' in col_data else 'FP/Race'
                        feature_maps['track_fppg'][player_id] = self._safe_float(col_data[fppg_key])

                # Extract recent form (Driver Last Season table as proxy)
                elif table_id == 'dLastSeason':
                    if 'Avg Finish' in col_data:
                        feature_maps['recent_avg_finish'][player_id] = self._safe_float(col_data['Avg Finish'], 20.0)
                    if 'FPPG' in col_data:
                        feature_maps['recent_fppg'][player_id] = self._safe_float(col_data['FPPG'])
                        feature_maps['season_fppg'][player_id] = self._safe_float(col_data['FPPG'])

                # Extract track history
                elif 'track history' in table_name or 'track stats' in table_name:
                    if 'Avg Finish' in col_data:
                        feature_maps['track_avg_finish'][player_id] = self._safe_float(col_data['Avg Finish'], 20.0)
                    if 'Top 5%' in col_data:
                        feature_maps['track_top5_pct'][player_id] = self._safe_float(col_data['Top 5%'])
                    if 'FPPG' in col_data:
                        feature_maps['track_fppg'][player_id] = self._safe_float(col_data['FPPG'])

                # Extract recent form
                elif 'last 5' in table_name or 'recent' in table_name:
                    if 'Avg Finish' in col_data:
                        feature_maps['recent_avg_finish'][player_id] = self._safe_float(col_data['Avg Finish'], 20.0)
                    if 'FPPG' in col_data:
                        feature_maps['recent_fppg'][player_id] = self._safe_float(col_data['FPPG'])

        # Add all features to dataframe - use None for missing data instead of 0
        for feature_name, feature_map in feature_maps.items():
            if feature_name in ['total_races', 'dnf_pct']:  # Keep these as 0 for calculations
                df[feature_name] = df['player_id'].map(feature_map).fillna(0)
            else:
                df[feature_name] = df['player_id'].map(feature_map)  # Allow None values

        # Calculate derived metrics for NASCAR
        # Position differential potential (key NASCAR metric) - only for drivers with starting position
        df['position_differential'] = None
        df['pd_upside'] = None

        valid_pos = df['starting_position'].notna()
        if valid_pos.any():
            # Calculate PD for drivers starting outside top 15 (potential for advancement)
            df.loc[valid_pos, 'position_differential'] = df.loc[valid_pos, 'starting_position'] - 15
            df.loc[valid_pos, 'pd_upside'] = np.maximum(0, df.loc[valid_pos, 'position_differential'])

        # Dominator potential (drivers who lead laps and finish well)
        df['dominator_score'] = 0.0  # Default

        # Only calculate for drivers with valid data
        valid_laps = df['laps_led_avg'].notna() & (df['laps_led_avg'] > 0)
        valid_finish = df['avg_finish'].notna() & (df['avg_finish'] > 0)

        if valid_laps.any() or valid_finish.any():
            laps_component = (df['laps_led_avg'].fillna(0) / 10)  # Normalize laps led
            finish_component = (1 - df['avg_finish'].fillna(40) / 40) * 10  # Inverse of avg finish
            df['dominator_score'] = laps_component + finish_component

        # Track type is already extracted from JSON data during load_raw_data
        # No need to override here - keep the extracted value

        # Variance adjustments based on starting position (only for drivers with position data)
        df['variance_multiplier'] = 1.0  # Default
        valid_pos = df['starting_position'].notna()
        if valid_pos.any():
            df.loc[valid_pos & (df['starting_position'] <= 5), 'variance_multiplier'] = 0.9   # Front runners more consistent
            df.loc[valid_pos & (df['starting_position'] >= 25), 'variance_multiplier'] = 1.2  # Back of pack more volatile

        # Calculate ceiling adjustments for NASCAR (only for drivers with position data)
        df['ceiling_adjustment'] = 1.0  # Default
        valid_pos = df['starting_position'].notna()
        if valid_pos.any():
            df.loc[valid_pos & (df['starting_position'] > 15), 'ceiling_adjustment'] = 1.15
            df.loc[valid_pos & (df['starting_position'] > 10) & (df['starting_position'] <= 15), 'ceiling_adjustment'] = 1.05

        # 🛡️ SYNTHESIZED FLOOR CALCULATION (NASCAR-specific)
        # Real floors based on DNF risk, starting position, and track reliability
        def calculate_nascar_floor(row):
            base_projection = row['projection']
            starting_pos = row['starting_position']

            # DNF risk by starting position (empirical NASCAR data)
            if pd.isna(starting_pos):  # No starting position data
                dnf_risk = 0.15  # Average DNF risk
            elif starting_pos <= 5:
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
            if hasattr(row, 'track_fppg') and pd.notna(row.get('track_fppg')) and row['track_fppg'] > 35:
                calculated_floor *= 1.2  # Track specialists have higher floors

            # Floor can't exceed 30% of projection (prevents unrealistic floors)
            max_floor = base_projection * 0.30

            return min(calculated_floor, max_floor)

        df['synthesized_floor'] = df.apply(calculate_nascar_floor, axis=1)

        # Use synthesized floors only if original floor is NULL
        df['floor'] = df['floor'].fillna(df['synthesized_floor'])

        print(f"   ✅ Added {len(feature_maps)} NASCAR features to {len(df)} drivers")

        return df

    def _extract_ownership_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract ownership data from JSON with error handling."""
        ownership_map = {}
        try:
            ownership_data = json.loads(data.get('OwnershipJson', '{}'))
            for player in ownership_data.get('Players', []):
                salary_id = player.get('SalaryId')
                ownership = self._safe_float(player.get('Ownership', 10.0))
                if salary_id:
                    ownership_map[salary_id] = ownership
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse ownership data: {e}")
        return ownership_map

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float with fallback."""
        if value in ['0', '', None]:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_position_from_data(self, row: pd.Series) -> Position:
        """NASCAR drivers are all DRIVER position."""
        # pylint: disable=unused-argument
        return Position.DRIVER

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NASCAR-specific projection adjustments."""
        # Use base class implementation which handles all newsletter signals
        df = super().apply_projection_adjustments(df)

        # Apply NASCAR-specific adjustments
        logger.info("Applying NASCAR-specific adjustments...")

        # 1. Position differential upside influences ceiling
        pd_boost = df['pd_upside'] > 0
        df.loc[pd_boost, 'updated_ceiling'] *= (1 + df.loc[pd_boost, 'pd_upside'] * NASCAR_CONFIG.PD_CEILING_MULTIPLIER)

        # 2. Dominator score influences projection
        dom_boost = df['dominator_score'] > NASCAR_CONFIG.DOMINATOR_THRESHOLD
        df.loc[dom_boost, 'updated_projection'] *= (1 + (df.loc[dom_boost, 'dominator_score'] - NASCAR_CONFIG.DOMINATOR_THRESHOLD) * NASCAR_CONFIG.DOMINATOR_MULTIPLIER)

        # 3. Track history influences floor
        track_experts = df['track_avg_finish'] < NASCAR_CONFIG.TRACK_EXPERT_THRESHOLD
        df.loc[track_experts, 'updated_floor'] *= NASCAR_CONFIG.TRACK_EXPERT_FLOOR_BOOST

        # 4. DNF risk influences floor negatively (only for drivers with DNF data)
        valid_dnf = df['dnf_pct'].notna() & (df['dnf_pct'] > 0)
        high_dnf = valid_dnf & (df['dnf_pct'] > NASCAR_CONFIG.HIGH_DNF_THRESHOLD)
        if high_dnf.any():
            df.loc[high_dnf, 'updated_floor'] *= NASCAR_CONFIG.DNF_FLOOR_PENALTY

        # 5. Apply ceiling adjustments from calculate_sport_metrics
        df['updated_ceiling'] *= df['ceiling_adjustment']

        # Apply bounds validation after all adjustments
        logger.debug("Applying bounds validation...")

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