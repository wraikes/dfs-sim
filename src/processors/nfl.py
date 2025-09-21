"""NFL-specific data processor."""

import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import BaseDataProcessor
from ..models.player import Position


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
                        owned_pct = ownership_player.get('Owned')
                        if salary_id:
                            ownership_map[salary_id] = owned_pct

            players = []

            for player_data in salary_data['Salaries']:
                player_id = player_data['PID']
                salary_id = player_data['Id']

                # Get ownership from ownership_map (preserve None if not found)
                ownership = ownership_map.get(salary_id)

                player = {
                    'player_id': player_id,
                    'name': player_data['Name'],
                    'salary': player_data['SAL'],
                    'projection': player_data.get('PP', 0.0),
                    'floor': player_data.get('Floor', 0.0),
                    'ceiling': player_data.get('Ceil', 0.0),
                    'ownership': ownership,
                    'std_dev': 25.0,  # NFL baseline variance placeholder
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
                    'game_timestamp': player_data.get('GT', ''),  # Add raw timestamp
                }
                players.append(player)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ❌ Error parsing SalaryContainerJson: {e}")
            raise ValueError(f"Invalid NFL data format: {e}")

        # Create base dataframe
        df = pd.DataFrame(players)

        # Parse game timestamps to datetime
        def parse_game_timestamp(ts_str):
            """Parse /Date(1757858400000-0400)/ format to EST datetime."""
            if not ts_str or '/Date(' not in ts_str:
                return None
            match = re.search(r'/Date\((\d+)([-+]\d{4})\)/', ts_str)
            if match:
                timestamp_ms = int(match.group(1))
                timestamp_s = timestamp_ms / 1000
                # The timestamp is already in UTC, just convert directly
                dt = datetime.fromtimestamp(timestamp_s)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            return None

        df['game_datetime'] = df['game_timestamp'].apply(parse_game_timestamp)

        # Extract critical NFL features from MatchupData for correlation/variance modeling
        self._extract_nfl_matchup_features(df, data)

        print(f"   ✅ Loaded {len(df)} active NFL players with advanced metrics")
        return df

    def _extract_nfl_matchup_features(self, df: pd.DataFrame, raw_data: dict):
        """Extract NFL-specific features from MatchupData for correlation modeling."""

        # Initialize feature maps
        target_share_map = {}           # WR/TE target volume
        snap_pct_map = {}              # Playing time correlation
        matchup_grade_map = {}         # Defensive matchup quality
        pace_metrics_map = {}          # Game flow indicators

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
            lambda x: snap_pct_map.get(x, {}).get('snap_pct')  # Preserve None if not available
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
                lambda x: pace_metrics_map.get(x, {}).get(f'{pos}_fppg_allowed')
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

        # Skip variance adjustments since std_dev is just a placeholder
        df['variance_multiplier'] = 1.0
        df['adjusted_std_dev'] = df['std_dev']  # Keep placeholder as-is

        # Calculate ceiling adjustments based on position and matchup
        df['ceiling_adjustment'] = 1.0

        # QB ceiling boost for high-total games (extract from game_info if available)
        qb_mask = df['position'] == Position.QB.value
        df.loc[qb_mask, 'ceiling_adjustment'] = 1.1

        # WR/TE ceiling boost vs weak pass defense (simplified - use opponent rank)
        pass_catcher_mask = df['position'].isin([Position.WR.value, Position.TE.value])
        weak_defense_mask = df['opponent_rank'] >= 20  # Bottom defenses
        df.loc[pass_catcher_mask & weak_defense_mask, 'ceiling_adjustment'] = 1.15

        # RB ceiling boost vs weak run defense
        rb_mask = df['position'] == Position.RB.value
        df.loc[rb_mask & weak_defense_mask, 'ceiling_adjustment'] = 1.1

        # DST ceiling boost vs weak offense (lower opponent rank = weaker offense)
        dst_mask = df['position'] == Position.DST.value
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

        # Skip risk-adjusted metrics since std_dev is placeholder
        df['sharpe_ratio'] = 0  # Disable since std_dev is not real data
        df['ceiling_volatility'] = np.where(df['projection'] > 0,
                                            (df['adjusted_ceiling'] - df['projection']) / df['projection'], 0)
        df['confidence_weighted_proj'] = df['projection'] * (df['confidence'] / 100.0)

        print("   🎯 Engineering NFL position-specific features...")

        # Position-Specific Features
        # RB: Goal line value (RZ targets × snap share)
        df['goal_line_value'] = np.where(df['position'] == Position.RB.value,
                                         df['rz_targets'] * (df['snap_pct'] / 100.0), 0)

        # WR/TE: Target efficiency (targets per snap)
        df['target_efficiency'] = np.where(df['position'].isin([Position.WR.value, Position.TE.value]) & (df['snap_pct'] > 0),
                                           df['targets_per_game'] / (df['snap_pct'] / 100.0), 0)

        # QB: Passing volume indicator
        df['qb_volume_indicator'] = np.where(df['position'] == Position.QB.value,
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